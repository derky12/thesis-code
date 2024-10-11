import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset, Sampler
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import _LRScheduler
import argparse
import torch.optim as optim
import atomInSmiles as ais
from utils import *
from configs import *
import logging
from tqdm import tqdm
import gc
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import random
from rdkit import Chem
import rdkit
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs
import sentencepiece as spm
import glob


def greedy_search(model, e_output, e_mask, trg_sp, config):
    last_words = torch.LongTensor([config.pad_id] * config.seq_len).to(
        config.device
    )  # (L)
    last_words[0] = config.sos_id  # (L)
    cur_len = 1

    for i in range(config.seq_len):
        d_mask = (
            (last_words.unsqueeze(0) != config.pad_id).unsqueeze(1).to(config.device)
        )  # (1, 1, L)
        nopeak_mask = torch.ones(
            [1, config.seq_len, config.seq_len], dtype=torch.bool
        ).to(
            config.device
        )  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = model.positional_encoder(trg_embedded)
        # print(f"{i=}, greedy_search, {trg_positional_encoded.shape=}")
        decoder_output = model.decoder(
            trg_positional_encoded, e_output, e_mask, d_mask
        )  # (1, L, d_model)

        output = model.softmax(
            model.output_linear(decoder_output)
        )  # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1)  # (1, L)
        last_word_id = output[0][i].item()

        if i < config.seq_len - 1:
            last_words[i + 1] = last_word_id
            cur_len += 1

        if last_word_id == config.eos_id:
            break

    if last_words[-1].item() == config.pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)

    return decoded_output


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


def setup_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    logging_file = os.path.join(
        config.ckpt_dir,
        "logs",
        f"trans_{config.training_id}_batch_size_{config.train_batch_size}.log",
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
    model,
    accelerator,
    optimizer,
    epoch,
    global_step,
    best_pct_match,
    best_avg_score,
    file_path,
):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_pct_match": best_pct_match,
        "best_avg_score": best_avg_score,
    }
    accelerator.save(state_dict, file_path)


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


def smiles_to_ais_tokens(smiles_list):
    ais_tokens_list = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            random_smiles = Chem.MolToSmiles(mol)

            tmp = Chem.MolFromSmiles(random_smiles)
            for atom in tmp.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            mapped_smiles = Chem.MolToSmiles(tmp)

            ais_tokens = ais.encode(mapped_smiles, with_atomMap=True)
            ais_tokens_list.append(ais_tokens)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            ais_tokens_list.append(None)
    return ais_tokens_list


class CSVRowIterator:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.reader = pd.read_csv(csv_path, low_memory=False, chunksize=1)

    def __iter__(self):
        self.reader = pd.read_csv(self.csv_path, low_memory=False, chunksize=1)
        return self

    def __next__(self):
        chunk = next(self.reader)
        return chunk.iloc[0]


class SMILESDataset(Dataset):
    def __init__(self, csv_path, src_sp, trg_sp, config):
        self.csv_iterator = CSVRowIterator(csv_path)
        self.src_sp = src_sp
        self.trg_sp = trg_sp
        self.config = config
        self.eos_id = self.config.eos_id
        self.sos_id = self.config.sos_id
        self.length = sum(
            1 for _ in CSVRowIterator(csv_path)
        )  # Get the length of the CSV file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of range")

        # Reset the iterator if we reach the end
        if idx == 0:
            self.csv_iterator = CSVRowIterator(self.csv_iterator.csv_path)

        # Move the iterator to the correct position
        for _ in range(idx):
            next(self.csv_iterator)

        row = next(self.csv_iterator)
        ais_tokens = row["ais_tokens"]
        smiles = row["smiles"]

        src_encoded = self.src_sp.EncodeAsIds(ais_tokens)
        trg_encoded = self.trg_sp.EncodeAsIds(ais_tokens)

        src_input = pad_or_truncate(src_encoded + [self.eos_id], self.config)
        trg_input = pad_or_truncate([self.sos_id] + trg_encoded, self.config)
        trg_output = pad_or_truncate(trg_encoded + [self.eos_id], self.config)

        return (
            torch.LongTensor(src_input),
            torch.LongTensor(trg_input),
            torch.LongTensor(trg_output),
            smiles,
        )


class ShuffledSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)


def make_mask(src_input, trg_input, config):
    e_mask = (src_input != config.pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != config.pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones(
        [1, config.seq_len, config.seq_len], dtype=torch.bool
    )  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(
        config.device
    )  # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

    return e_mask, d_mask


def pad_or_truncate(tokenized_text, config):
    if len(tokenized_text) < config.seq_len:
        left = config.seq_len - len(tokenized_text)
        padding = [config.pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[: config.seq_len]

    return tokenized_text


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


def compute_reconstruction_trans(model, val_dataloader, config, src_sp, trg_sp):
    samples = []
    all_samples = pd.DataFrame()
    model.eval()
    device = config.device

    # Get all datapoints from the dataloader
    all_datapoints = []
    for batch in val_dataloader:
        all_datapoints.extend(list(zip(*batch)))

    # Randomly choose 200 datapoints
    sample_size = min(config.val_smi_num, len(all_datapoints))
    sampled_datapoints = random.sample(all_datapoints, sample_size)

    # for i, batch in enumerate(val_dataloader):
    #     src_input, trg_input, trg_output, _ = batch
    exact = 0
    tanimoto = 0
    num_invalid = 0
    # no gradeint calculation
    with torch.no_grad():
        i = 1
        for src_input, trg_input, trg_output, cano_smi in sampled_datapoints:
            src_input = src_input.to(device)
            trg_input = trg_input.to(device)
            trg_output = trg_output.to(device)

            src_j = src_input.unsqueeze(0).to(device)  # (L) => (1, L)
            encoder_mask = (
                (src_j != config.pad_id).unsqueeze(1).to(device)
            )  # (1, L) => (1, 1, L)
            encoder_output = model.forward_encoder(
                src_j, encoder_mask
            )  # (1, L, d_model)
            s_src = src_sp.decode_ids(src_input.tolist())
            # s_truth = trg_sp.decode_ids(trg_output.tolist())
            # print(f"{encoder_output.shape=},{encoder_mask.shape=}")

            s_pred = greedy_search(model, encoder_output, encoder_mask, trg_sp, config)
            # print(f"{s_truth=},{s_pred=}")
            decod_truth = cano_smi  # AisDecoder(s_truth)
            # print(f"{decod_truth=}")
            decod_pred = AisDecoder(s_pred)
            # print(f"{decod_pred=}")
            try:
                decod_pred = normalize_smiles(decod_pred, canonical=True, isomeric=True)
            except:
                pass

            if decod_pred != "invalid":
                if decod_truth == decod_pred:
                    exact += 1
                    tanimoto += 1.0
                else:
                    tanimoto_ = FpSimilarity(
                        decod_truth, decod_pred, radius=2, nBits=2048
                    )
                    tanimoto += tanimoto_
            else:
                num_invalid += 1

            pct_match = exact / sample_size
            avg_score = tanimoto / sample_size
            invalid_ratio = num_invalid / sample_size
            print(
                f"Sample {i}/{sample_size}: Truth: {decod_truth} || Pred: {decod_pred}"
            )
            i += 1
            print(
                f"Current pct_match ratio: {pct_match} || Current tanimoto: {avg_score} || Invalid ratio: {invalid_ratio}\n\n"
            )
    model.train()
    return exact, pct_match, avg_score, invalid_ratio


def train(config):
    print(f"\n\n{config.training_id=} \n\n")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.ckpt_dir, "logs"),
        mixed_precision=config.mixed_precision,
    )

    logger = setup_logger(config)
    device = accelerator.device
    torch.manual_seed(config.seed)

    # load the model
    model = Transformer(config).to(device)
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load("./data/sp/ais_sp.model")
    trg_sp.Load("./data/sp/ais_sp.model")

    # load the datasets
    print("Loading dataloaders...")
    # create the training dataset
    train_dataset = SMILESDataset(
        "/root/docker/monash/master_thesis/codes/final/ais_spec_2/data/val_1000.csv",
        src_sp,
        trg_sp,
        config,
    )
    # shuffled_sampler = ShuffledSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    # train_dataset = SMILESDataset(config.training_data_path, src_sp, trg_sp, config)
    # shuffled_sampler = ShuffledSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size)
    # Usage for validation dataset
    val_dataset = SMILESDataset(config.val_data_path, src_sp, trg_sp, config)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.train_batch_size, shuffle=False
    )

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
    criterion = nn.NLLLoss()

    (
        model,
        optimizer,
        kl_annealer,
        lr_annealer,
        train_dataloader,
        val_dataloader,
        src_sp,
        trg_sp,
    ) = accelerator.prepare(
        model,
        optimizer,
        kl_annealer,
        lr_annealer,
        train_dataloader,
        val_dataloader,
        src_sp,
        trg_sp,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_pct_match = 0.0
    best_avg_score = 0.0

    if config.resume_from_checkpoint is not None:
        logger.info(
            f"Resuming training from checkpoint: {config.resume_from_checkpoint}"
        )
        checkpoint = torch.load(
            f"{config.ckpt_dir}/{config.resume_from_checkpoint}",
            map_location=device,
            weights_only=False,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        config.best_pct_match = checkpoint["best_pct_match"]
        config.best_avg_score = checkpoint["best_avg_score"]
        logger.info(f"Loaded checkpoint from epoch {start_epoch-1}")
        # run the validation function for the saved model
        # equals to using --not_in_training
        if not config.in_training:
            exact, pct_match, avg_score, invalid_ratio = compute_reconstruction_trans(
                model, val_dataloader, config, src_sp, trg_sp
            )
            logger.info(
                f"Total: {config.val_smi_num} || Matched: {exact} || Matched Ratio {pct_match*100:.2f}% || Invalid Ratio {invalid_ratio*100:.2f}%\n"
            )
            logger.info(f"Average tanimoto score: {avg_score}")
            # after running the validation function, quit the training
            return

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
                src_input, trg_input, trg_output, cano_smi = batch
                src_input = src_input.to(device)
                trg_input = trg_input.to(device)
                trg_output = trg_output.to(device)

                e_mask, d_mask = make_mask(src_input, trg_input, config)

                output = model(
                    src_input, trg_input, e_mask, d_mask
                )  # (B, L, vocab_size)
                trg_output_shape = trg_output.shape

                loss = criterion(
                    output.view(-1, config.trg_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1]),
                )

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
            del (
                src_input,
                trg_input,
                trg_output,
                e_mask,
                d_mask,
                output,
                loss,
                cano_smi,
                trg_output_shape,
            )
            torch.cuda.empty_cache()

        # if epoch != 0:
        # save the model every epoch
        # del previous saved .pth file
        common_string = f"{config.ckpt_dir}/{config.training_id}*_saved_epoch_*.pth"
        removed, _ = remove_matching_files(common_string)
        print(f"\nSuccessfully removed {removed} files.")

        save_checkpoint(
            model,
            accelerator,
            optimizer,
            epoch,
            global_step,
            best_pct_match,
            best_avg_score,
            f"{config.ckpt_dir}/{config.training_id}_saved_epoch_{epoch}.pth",
        )
        logger.info(f"Model saved at epoch {epoch}")

        # # custom valiation function
        # custom_validation_fn(model, val_dataloader, logger=logger,src_sp=src_sp, trg_sp=trg_sp, method="greedy", model_type="ais2")

        exact, pct_match, avg_score, invalid_ratio = compute_reconstruction_trans(
            model, val_dataloader, config, src_sp, trg_sp
        )
        logger.info(
            f"Total: {config.val_smi_num} || Matched: {exact} || Matched Ratio {pct_match*100:.2f}% || Invalid Ratio {invalid_ratio*100:.2f}%\n"
        )
        logger.info(f"Average tanimoto score: {avg_score}")

        # if model outputs a higher percent match, save the model
        if pct_match > best_pct_match:
            best_pct_match = pct_match
            print("\nNew best pct_match found\n")
            # del previous saved .pth file
            common_string = (
                f"{config.ckpt_dir}/{config.training_id}*_best_pct_match_*.pth"
            )
            removed, _ = remove_matching_files(common_string)
            print(f"\nSuccessfully removed {removed} files.")

            # update the latest best pct_match
            save_checkpoint(
                model,
                accelerator,
                optimizer,
                epoch,
                global_step,
                best_pct_match,
                best_avg_score,
                f"{config.ckpt_dir}/{config.training_id}_best_pct_match_{pct_match}_avg_score_{avg_score}_num_epoch_{epoch}.pth",
            )
            print(f"New best percent match saved")
            logger.info(f"New best percent match: {best_pct_match:.4f}")

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            print("\nNew best avg_score found\n")
            # update the latest best pct_match

            # del previous saved .pth file
            common_string = (
                f"{config.ckpt_dir}/{config.training_id}*_best_avg_score_*.pth"
            )
            removed, _ = remove_matching_files(common_string)
            print(f"\nSuccessfully removed {removed} files.")

            save_checkpoint(
                model,
                accelerator,
                optimizer,
                epoch,
                global_step,
                best_pct_match,
                best_avg_score,
                f"{config.ckpt_dir}/{config.training_id}_best_avg_score_{avg_score}_pct_match_{pct_match}_num_epoch_{epoch}.pth",
            )
            print(f"New best avg_score saved")
            logger.info(f"New best avg_score : {best_avg_score:.4f}")

        # print(f"\nWe have a new best val loss: {best_val_loss:.4f}")

        # if epoch % 4 == 0 or epoch == n_epoch - 1:

    logger.info("Training completed")


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    train(config)

