import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
import argparse
import torch.optim as optim
import atomInSmiles as ais
import numpy as np

from spectra_configs import *
from model import *
from spectra_model import *

import logging
from tqdm import tqdm
import gc
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import random
from rdkit import Chem
import rdkit
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs
import sentencepiece as spm
import glob
import pandas as pd
import math


def setup_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    logging_file = os.path.join(
        config.ckpt_dir,
        "logs",
        f"spectra_{config.training_id}_batch_size_{config.spectra_batch_size}.log",
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


class KLAnnealer:
    def __init__(self, n_epoch, config):
        self.i_start = config.kl_start
        self.w_start = config.kl_w_start
        self.w_max = config.kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer, config):
        self.n_period = config.lr_n_period
        self.n_mult = config.lr_n_mult
        self.lr_end = config.lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        super().__init__(optimizer, -1)

    def get_lr(self):
        return [
            self.lr_end
            + (base_lr - self.lr_end)
            * (1 + math.cos(math.pi * self.current_epoch / self.t_end))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


def remove_matching_files(file_pattern):
    """
    Remove files matching the given pattern.

    Args:
    file_pattern (str): The file pattern to match, including wildcards.

    Returns:
    tuple: A tuple containing two lists - successfully removed files and failed removals.
    """
    removed_files = []
    failed_removals = []

    # Find all files matching the pattern
    matching_files = glob.glob(file_pattern)

    # Remove each matching file
    for file_path in matching_files:
        try:
            os.remove(file_path)
            removed_files.append(file_path)
            print(f"Removed file: {file_path}")
        except OSError as e:
            failed_removals.append((file_path, str(e)))
            print(f"Error removing file {file_path}: {e}")

    return removed_files, failed_removals


def CalcFpTc(truth_smi, pred_smi, metric=Chem.DataStructs.TanimotoSimilarity):
    RDLogger.DisableLog("rdApp.*")
    try:
        pred_mol = Chem.MolFromSmiles(pred_smi, sanitize=True)
        if pred_mol is None:
            return 0
        truth_mol = Chem.MolFromSmiles(truth_smi, sanitize=True)
        if truth_mol is None:
            return 0
    except:
        return 0
    #     return Chem.DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(truth_mol), Chem.RDKFingerprint(pred_mol), metric=metric)
    return DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(truth_mol, 2, nBits=1024),
        AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=1024),
    )


def save_checkpoint(model, accelerator, optimizer, epoch, global_step, file_path):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    accelerator.save(state_dict, file_path)


def AisDecoder(new_tokens_list, rdLogger=False):

    RDLogger.EnableLog("rdApp.*") if rdLogger else RDLogger.DisableLog("rdApp.*")
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smarts = ""
    for new_token in new_tokens_list.split():
        try:
            token = regex.findall(new_token)[0]
        except:
            token = ""
        if "[[" in token:
            smarts += token[1:]
        else:
            if "[" in token:
                sym = token[1:-1].split(";")
                if "H" in sym[0]:
                    # print(sym[0])
                    try:
                        smarts += regex.findall(sym[0])[0]
                    except:
                        pass
                else:
                    smarts += sym[0]
            else:
                smarts += token
    try:
        mol_sma = Chem.MolFromSmarts(smarts, mergeHs=True)
        if mol_sma != None:
            return Chem.CanonSmiles(Chem.MolToSmiles(mol_sma))
        else:
            return "invalid"
    except:
        return "invalid"


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


# Metrics for evaluation
def FpSimilarity(
    smiles1,
    smiles2,
    metric=DataStructs.TanimotoSimilarity,  # DiceSimilarity
    fingerprint=Chem.rdMolDescriptors.GetHashedMorganFingerprint,
    rdLogger=False,  # RDKit logger
    **kwargs,
):
    RDLogger.EnableLog("rdApp.*") if rdLogger else RDLogger.DisableLog("rdApp.*")
    try:
        mol1 = Chem.MolFromSmiles(smiles1, sanitize=True)
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
        if mol2 is not None and mol1 is not None:
            fp1 = fingerprint(mol1, **kwargs)
            fp2 = fingerprint(mol2, **kwargs)
            return metric(fp1, fp2)
        else:
            if rdLogger:
                warnings.warn(f"{smiles1=}, {smiles2=}")
            return 0
    except:
        return 0


def validation_loop(model, dataloader):
    errors = []
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            spectra, embedding, _ = batch
            spectra = spectra.to(device).to(torch.bfloat16)
            embedding = embedding.to(device).squeeze(1)
            preds = model(spectra)
            error = torch.mean((preds - embedding) ** 2)
            errors.append(error.item())
    model.train()
    return np.mean(errors)


class SpectraDataset(Dataset):
    def __init__(self, csv_path, model):

        data = pd.read_csv(csv_path, low_memory=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.negative_spectra_low = data["negative_spectra_low"].tolist()
        self.negative_spectra_high = data["negative_spectra_high"].tolist()
        self.positive_spectra_low = data["positive_spectra_low"].tolist()
        self.positive_spectra_high = data["positive_spectra_high"].tolist()
        # self.negative_mz_l = data["negative_mz"].tolist()
        # self.positive_mz_l = data["positive_mz"].tolist()
        self.smi_l = data["cano_smi"].tolist()

        self.resolution = 2
        self.minmass = 50.0
        self.maxmass = 499.97

    def __len__(self):
        return len(self.smi_l)

    def __getitem__(self, idx):
        positive_spectra_low = self.positive_spectra_low[idx]
        positive_spectra_high = self.positive_spectra_high[idx]
        negative_spectra_low = self.negative_spectra_low[idx]
        negative_spectra_high = self.negative_spectra_high[idx]
        vec_poslow = spec2vec(
            positive_spectra_low, self.minmass, self.maxmass, self.resolution
        )
        vec_poshigh = spec2vec(
            positive_spectra_high, self.minmass, self.maxmass, self.resolution
        )
        vec_neglow = spec2vec(
            negative_spectra_low, self.minmass, self.maxmass, self.resolution
        )
        vec_neghigh = spec2vec(
            negative_spectra_high, self.minmass, self.maxmass, self.resolution
        )
        spectra = np.stack((vec_poslow, vec_poshigh, vec_neglow, vec_neghigh))
        spectra = torch.from_numpy(spectra)
        smi = self.smi_l[idx]
        smiles_tensor_list = [self.model.string2tensor(smiles) for smiles in [smi]]
        h = self.model.forward_encoder(smiles_tensor_list)

        return spectra, h, smi


def spec2vec(spectrum, minmass, maxmass, resolution=2):
    mult = pow(10, resolution)
    length = int((maxmass - minmass) * mult)
    vec = np.zeros(length)
    if "0.0:0.0" in spectrum:
        return vec
    else:
        for peak in spectrum.split(" "):
            p_l = peak.split(":")
            mass = float(p_l[0])
            abund = float(p_l[1])
            mass = round(float(mass), resolution) * mult
            try:
                vec[int(mass) - int(minmass * mult)] = (
                    vec[int(mass) - int(minmass * mult)] + abund
                )
            except:
                raise ValueError(f"mass: {mass}")
        return vec / np.max(vec)


def mz2vec(mz_mass, minmass, maxmass, resolution=2):
    mult = pow(10, resolution)
    length = int((maxmass - minmass) * mult)
    vec = np.zeros(length)
    vec[0] = round(mz_mass / 1000, 4)
    return vec


def min_max_mass(spectrum):
    """
    it finds the starting and the ending mass of a spectrum
    Args:
        A spectrum as it is formatted in the NIST sdf files
    Output:
        a tuple (starting mass, ending mass)
    """
    for peak in spectrum.split(" "):
        mass, abund = peak.split(":")
        mass = round(float(mass), 2)
        start = mass
        break
    for peak in spectrum.split(" "):
        mass, abund = peak.split(":")
        mass = round(float(mass), 2)
    end = mass
    return (start, end)


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

    ckpt = torch.load(
        config.trans_ckpt,
        map_location=device,
        weights_only=False,
    )
    trans_model = TranslationModel(config).to(device)
    trans_model.load_state_dict(ckpt["model"])

    print("Loading dataloaders...")
    train_set = SpectraDataset(config.spectra_train_path, trans_model)
    train_dataloader = DataLoader(
        train_set,
        batch_size=config.spectra_batch_size,
        shuffle=True,
    )

    val_set = SpectraDataset(config.spectra_val_path, trans_model)
    val_dataloader = DataLoader(
        val_set,
        batch_size=config.spectra_batch_size,
        shuffle=True,
    )
    print("Loading model...")
    model = Net1D(config.spectra_length)

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
    best_val_error = float("inf")
    criterion = nn.MSELoss()

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
                spectra, embedding, _ = batch
                spectra = spectra.to(device).to(torch.bfloat16)
                embedding = embedding.to(device).squeeze(1)
                # print(f"{spectra.shape=},{embedding.shape=}")
                preds = model(spectra)
                loss = criterion(preds, embedding)
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
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            logger.info(
                f"Step {global_step}: train_loss={logs['train_loss']:.4f}, model_lr={logs['model_lr']:.6f}"
            )
            global_step += 1

            # Free up memory
            del spectra, embedding, preds, loss
            torch.cuda.empty_cache()

        # if epoch != 0:
        # save the model every epoch
        # del previous saved .pth file
        common_string = f"{config.ckpt_dir}/{config.training_id}_epoch_*_saved.pth"
        removed, _ = remove_matching_files(common_string)
        print(f"\nSuccessfully removed {removed} files.")

        save_checkpoint(
            model,
            accelerator,
            optimizer,
            epoch,
            global_step,
            file_path=f"{config.ckpt_dir}/{config.training_id}_epoch_{epoch}_saved.pth",
        )
        logger.info(f"Model saved at epoch {epoch}")

        # # custom valiation function
        # custom_validation_fn(model, val_dataloader, logger=logger,src_sp=src_sp, trg_sp=trg_sp, method="greedy", model_type="ais2")

        # run validation
        val_error = validation_loop(model, val_dataloader)
        logger.info(f"\nEpoch {epoch}, Step {global_step} || {val_error=}")
        if val_error < best_val_error:

            common_string = f"{config.ckpt_dir}/{config.training_id}_num_epoch_*_best_val_error_*.pth"
            removed, _ = remove_matching_files(common_string)
            print(f"\nSuccessfully removed {removed} files.")

            best_val_error = val_error

            print("\nNew best_val_error found\n")
            # del previous saved .pth file
            try:
                common_string = f"{config.ckpt_dir}/{config.training_id}_num_epoch_{n_epoch}_best_val_error_*.pth"
                removed, _ = remove_matching_files(common_string)
                print(f"\nSuccessfully removed {removed} files.")
            except:
                pass

            # update the latest best pct_match
            save_checkpoint(
                model,
                accelerator,
                optimizer,
                epoch,
                global_step,
                f"{config.ckpt_dir}/{config.training_id}_num_epoch_{epoch}_best_val_error_{best_val_error}.pth",
            )
            logger.info(f"New best_val_error model saved {best_val_error=}")

    logger.info("Training completed")


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    train(config)

