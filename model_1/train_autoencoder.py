import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm

import argparse
import torch.optim as optim
import atomInSmiles as ais
from utils import *
from configs import *
from model import *
import logging
from tqdm import tqdm
import gc
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn


def setup_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    logging_file = os.path.join(
        config.ckpt_dir,
        "logs",
        f"trans_{config.training_id}_batch_size_{config.trans_batch_size}.log",
    )
    if os.path.exists(logging_file):
        os.remove(logging_file)
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(
    model, accelerator, optimizer, epoch, global_step, best_pct_match, file_path
):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_pct_match": best_pct_match,
    }
    accelerator.save(state_dict, file_path)


def train(config):
    print(f"\n\n{config.training_id=}\n\n")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.ckpt_dir, "logs"),
        mixed_precision=config.mixed_precision,
    )

    logger = setup_logger(config)
    device = accelerator.device
    torch.manual_seed(config.seed)

    # load the datasets
    print("Loading dataloaders...")
    # create the training dataset
    dataset = SMILESDataset(
        data_path=config.training_data_path, real_dataset=config.real_dataset
    )
    train_dataloader, _ = create_batches(
        dataset, batch_size=config.trans_batch_size, ratio=1.0
    )
    # create the validation dataset
    dataset = SMILESDataset(
        data_path=config.val_data_path, real_dataset=config.real_dataset
    )
    _, val_dataloader = create_batches(
        dataset, batch_size=config.trans_batch_size, ratio=0.0
    )

    # load the model
    model = TranslationModel(config).to(device)

    n_epoch = sum(
        config.lr_n_period * (config.lr_n_mult**i) for i in range(config.lr_n_restarts)
    )
    print(f"\nTotal number of epochs: {n_epoch}\n")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr_start,
    )
    kl_annealer = KLAnnealer(n_epoch, config)
    lr_annealer = CosineAnnealingLRWithRestart(optimizer, config)
    # for i in range(config.start_epoch):
    #     lr_annealer.step()

    (
        model,
        optimizer,
        kl_annealer,
        lr_annealer,
        train_dataloader,
        val_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer,
        kl_annealer,
        lr_annealer,
        train_dataloader,
        val_dataloader,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if config.resume_from_checkpoint is not None:
        logger.info(
            f"Resuming training from checkpoint: {config.resume_from_checkpoint}"
        )
        checkpoint = torch.load(
            f"{config.ckpt_dir}/{config.resume_from_checkpoint}",
            map_location=config.device,
            weights_only=False,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        config.best_pct_match = checkpoint["best_pct_match"]
        logger.info(f"Loaded checkpoint from epoch {start_epoch-1}")

    for epoch in range(start_epoch, n_epoch):
        model.train()
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{n_epoch}",
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                rand_smi, cano_smi = batch
                _, recon_loss = model(rand_smi, cano_smi)
                # logger.info(f"current smiles are {cano_smi}")
                # print(f"{kl_loss=},{recon_loss=}")
                # kl_weight = kl_annealer(epoch,config)
                loss = recon_loss

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()
                lr_annealer.step()

            progress_bar.update(1)
            logs = {
                "train_loss": loss.detach().item(),
                "model_lr": optimizer.param_groups[0]["lr"],
                # "transformer_lr": trans_scheduler.get_last_lr()[0],
                # "transformer_lr": trans_optimizer._get_lr(),
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            logger.info(
                f"Step {global_step}: train_loss={logs['train_loss']:.4f}, model_lr={logs['model_lr']:.6f}"
            )
            global_step += 1

            # Free up memory
            del recon_loss, rand_smi, cano_smi
            torch.cuda.empty_cache()

        # run validation
        # val_loss = validation_loop(model, val_dataloader)
        # if val_loss < best_val_loss:
        # best_val_loss = val_loss

        # if epoch != 0:
        # save the model every epoch
        save_checkpoint(
            model,
            accelerator,
            optimizer,
            epoch,
            global_step,
            config.best_pct_match,
            f"{config.ckpt_dir}/{config.training_id}_batch_size_{config.trans_batch_size}_epoch_saved.pth",
        )
        logger.info(f"Model saved at epoch {epoch}")

        pct_match, avg_score = compute_reconstruction_trans(model, val_dataloader)
        print(f"{pct_match=},{avg_score=}")
        logger.info(f"\nEpoch {epoch}, Step {step}: Percent Match = {pct_match:.4f}\n")

        # if model outputs a higher percent match, save the model
        if pct_match > config.best_pct_match:
            config.best_pct_match = pct_match
            print("\nNew best pct_match found\n")
            # del previous saved .pth file
            common_string = f"{config.ckpt_dir}/{config.training_id}_batch_size_{config.trans_batch_size}_num_epoch_{n_epoch}_val_pct_*_best*.pth"
            removed, _ = remove_matching_files(common_string)
            print(f"\nSuccessfully removed {removed} files.")

            # update the latest best pct_match
            save_checkpoint(
                model,
                accelerator,
                optimizer,
                epoch,
                global_step,
                config.best_pct_match,
                f"{config.ckpt_dir}/{config.training_id}_batch_size_{config.trans_batch_size}_num_epoch_{n_epoch}_val_pct_{pct_match}_best_avg_score_{avg_score}.pth",
            )
            print(f"New best percent match saved")
            logger.info(f"New best percent match: {config.best_pct_match:.4f}")

        if avg_score > config.best_avg_score:
            config.best_avg_score = avg_score
            print("\nNew best avg_score found\n")
            # update the latest best pct_match

            # del previous saved .pth file
            common_string = f"{config.ckpt_dir}/{config.training_id}_batch_size_{config.trans_batch_size}_num_epoch_{n_epoch}_val_avg_score_*_best*.pth"
            removed, _ = remove_matching_files(common_string)
            print(f"\nSuccessfully removed {removed} files.")

            save_checkpoint(
                model,
                accelerator,
                optimizer,
                epoch,
                global_step,
                config.best_pct_match,
                f"{config.ckpt_dir}/{config.training_id}_batch_size_{config.trans_batch_size}_num_epoch_{n_epoch}_val_avg_score_{avg_score}_best_pct_{pct_match}.pth",
            )
            print(f"New best avg_score saved")
            logger.info(f"New best avg_score : {config.best_avg_score:.4f}")

        # print(f"\nWe have a new best val loss: {best_val_loss:.4f}")

        # if epoch % 4 == 0 or epoch == n_epoch - 1:

    logger.info("Training completed")


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    train(config)
