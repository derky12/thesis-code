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
from configs_spectra import *
from transformer import *
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
import copy
import csv
import pandas as pd
from utils_eval import *
import math


def beam_search(model, e_output, e_mask, trg_sp, config):
    device = config.device
    cur_queue = PriorityQueue()
    # for k in range(beam_size):
    cur_queue.put(BeamNode(config.sos_id, -0.0, [config.sos_id]))

    finished_count = 0
    for pos in range(config.seq_len):
        new_queue = PriorityQueue()
        for k in range(config.beam_size):
            if pos == 0 and k > 0:
                continue
            else:
                node = cur_queue.get()

            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(
                    node.decoded
                    + [config.pad_id] * (config.seq_len - len(node.decoded))
                ).to(
                    device
                )  # (L)
                d_mask = (
                    (trg_input.unsqueeze(0) != config.pad_id).unsqueeze(1).to(device)
                )  # (1, 1, L)
                nopeak_mask = torch.ones(
                    [1, config.seq_len, config.seq_len], dtype=torch.bool
                ).to(device)
                nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.positional_encoder(trg_embedded)
                decoder_output = model.decoder(
                    trg_positional_encoded, e_output, e_mask, d_mask
                )  # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                )  # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=config.beam_size)
                last_word_ids = output.indices.tolist()  # (k)
                last_word_prob = output.values.tolist()  # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(
                        idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx]
                    )
                    if idx == config.eos_id:
                        # new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

    all_candidates = list()
    scores = []
    for _ in range(config.beam_size):
        node = cur_queue.get()
        decoded_output = node.decoded
        scores.append(node.prob)
        all_candidates.append(trg_sp.decode_ids(decoded_output))
    # print("\n", all_candidates, "\n")
    # print("\n", scores, "\n")

    return all_candidates, scores


def setup_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    logging_file = os.path.join(
        config.ckpt_dir,
        "logs",
        f"spectra_{config.training_id}_batch_size_{config.train_batch_size}.log",
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


def compute_reconstruction_trans(
    spectra_model, trans_model, val_dataloader, config, src_sp, trg_sp
):
    samples = []

    spectra_model.eval()
    trans_model.eval()
    device = config.device

    # Get all datapoints from the dataloader
    all_datapoints = []
    for batch in val_dataloader:
        all_datapoints.extend(list(zip(*batch)))

    with torch.no_grad():
        if config.in_training:
            i = 1
            df = pd.DataFrame()
            print(f"{config.in_training=}")
            # Randomly choose 200 datapoints
            sample_size = min(config.val_smi_num, len(all_datapoints))
            sampled_datapoints = random.sample(all_datapoints, sample_size)
            exact = 0
            tanimoto = 0
            num_invalid = 0
            for src_input, _, trg_output, smi, spectra in sampled_datapoints:
                # print("in compute_reconstruction_trans iteration now")
                src_input = src_input.to(device)
                # print(f"{src_input.shape=}")
                spectra = spectra.unsqueeze(0).to(device).to(torch.bfloat16)
                # print(f"{spectra.shape=}")
                src_j = src_input.unsqueeze(0).to(device)  # (L) => (1, L)
                # print(f"{src_j.shape=}")
                encoder_mask = (
                    (src_j != config.pad_id).unsqueeze(1).to(device)
                )  # (1, L) => (1, 1, L)
                # print(f"{encoder_mask.shape=}")
                pred_encoder_output = spectra_model(spectra)
                # print(f"{pred_encoder_output.shape=}")
                candidates_l, _ = beam_search(
                    trans_model, pred_encoder_output, encoder_mask, trg_sp, config
                )
                print(f"{candidates_l=}")
                decod_truth = smi  # AisDecoder(s_truth)
                # print(f"{decod_truth=}")
                tani_ = 0
                pred_smi = "invalid"
                for s_pred in candidates_l:
                    # among the candidates list, find the highest tanimoto score carrying smiles
                    decod_pred = AisDecoder(s_pred)
                    print(f"i, {decod_pred=}")
                    # print(f"{decod_pred=}")
                    # try:
                    #     decod_pred = normalize_smiles(
                    #         decod_pred, canonical=True, isomeric=True
                    #     )
                    #     pred_smi = decod_pred
                    # except:
                    #     pass

                    if decod_pred != "invalid":
                        tanimoto_ = FpSimilarity(
                            decod_truth, decod_pred, radius=2, nBits=1024
                        )
                    else:
                        tanimoto_ = 0

                    if tanimoto_ > tani_:
                        tani_ = tanimoto_
                        pred_smi = decod_pred

                if pred_smi != "invalid":
                    if decod_truth == pred_smi:
                        exact += 1
                        tanimoto += 1.0
                    else:
                        tanimoto_ = FpSimilarity(
                            decod_truth, pred_smi, radius=2, nBits=1024
                        )
                        print(f"{pred_smi}'s {tanimoto_=}")
                        tanimoto += tanimoto_
                else:
                    num_invalid += 1

                pct_match = exact / sample_size
                avg_score = tanimoto / sample_size
                invalid_ratio = num_invalid / sample_size
                print(
                    f"Sample {i}/{sample_size}: Truth: {decod_truth} || Pred: {pred_smi}"
                )
                i += 1
                print(
                    f"Current pct_match ratio: {pct_match} || Current tanimoto: {avg_score} || Invalid ratio: {invalid_ratio}\n\n"
                )
                df = df._append(
                    {"true_smiles": decod_truth, "predict_smiles": pred_smi},
                    ignore_index=True,
                )
                df.to_csv("./pred_smiles_.csv")
            spectra_model.train()
            return exact, pct_match, avg_score, invalid_ratio
        elif not config.in_training:
            print(f"{config.in_training=}")
            for i in range(3):
                j = 1
                df = pd.DataFrame()
                sample_size = min(config.val_smi_num, len(all_datapoints))
                sampled_datapoints = random.sample(all_datapoints, sample_size)
                exact = 0
                tanimoto = 0
                num_invalid = 0
                for src_input, _, trg_output, smi, spectra in sampled_datapoints:
                    # print("in compute_reconstruction_trans iteration now")
                    src_input = src_input.to(device)
                    # print(f"{src_input.shape=}")
                    spectra = spectra.unsqueeze(0).to(device).to(torch.bfloat16)
                    # print(f"{spectra.shape=}")
                    src_j = src_input.unsqueeze(0).to(device)  # (L) => (1, L)
                    # print(f"{src_j.shape=}")
                    encoder_mask = (
                        (src_j != config.pad_id).unsqueeze(1).to(device)
                    )  # (1, L) => (1, 1, L)
                    # print(f"{encoder_mask.shape=}")
                    pred_encoder_output = spectra_model(spectra)
                    # print(f"{pred_encoder_output.shape=}")
                    candidates_l, _ = beam_search(
                        trans_model, pred_encoder_output, encoder_mask, trg_sp, config
                    )
                    print(f"{candidates_l=}")
                    decod_truth = smi  # AisDecoder(s_truth)
                    # print(f"{decod_truth=}")
                    tani_ = 0
                    pred_smi = "invalid"
                    for s_pred in candidates_l:
                        # among the candidates list, find the highest tanimoto score carrying smiles
                        decod_pred = AisDecoder(s_pred)
                        print(f"i, {decod_pred=}")
                        # print(f"{decod_pred=}")
                        # try:
                        #     decod_pred = normalize_smiles(
                        #         decod_pred, canonical=True, isomeric=True
                        #     )
                        #     pred_smi = decod_pred
                        # except:
                        #     pass

                        if decod_pred != "invalid":
                            tanimoto_ = FpSimilarity(
                                decod_truth, decod_pred, radius=2, nBits=1024
                            )
                        else:
                            tanimoto_ = 0

                        if tanimoto_ > tani_:
                            tani_ = tanimoto_
                            pred_smi = decod_pred

                    if pred_smi != "invalid":
                        if decod_truth == pred_smi:
                            exact += 1
                            tanimoto += 1.0
                        else:
                            tanimoto_ = FpSimilarity(
                                decod_truth, pred_smi, radius=2, nBits=1024
                            )
                            print(f"{pred_smi}'s {tanimoto_=}")
                            tanimoto += tanimoto_
                    else:
                        num_invalid += 1

                    pct_match = exact / sample_size
                    avg_score = tanimoto / sample_size
                    invalid_ratio = num_invalid / sample_size
                    print(
                        f"Sample {j}/{sample_size}: Truth: {decod_truth} || Pred: {pred_smi}"
                    )
                    print(
                        f"Current pct_match ratio: {pct_match} || Current tanimoto: {avg_score} || Invalid ratio: {invalid_ratio}\n\n"
                    )
                    df = df._append(
                        {"true_smiles": decod_truth, "predict_smiles": pred_smi},
                        ignore_index=True,
                    )
                    j += 1
                df.to_csv(
                    f"./pred_smiles_{config.training_id}_{config.beam_size}_{config.val_smi_num}_{i}.csv"
                )
                # remove predict_smiles where value is 'invalid'
                df = df[df["predict_smiles"] != "invalid"]
                result_dict = {}
                truth_smi_l = df["true_smiles"].tolist()
                pred_smi_l = df["predict_smiles"].tolist()
                for i in range(len(truth_smi_l)):
                    truth_smi = truth_smi_l[i]
                    pred_smi = pred_smi_l[i]
                    result_dict[truth_smi] = pred_smi

                mcs_ratio_l = []
                mcs_tan_l = []
                mcs_dict = {}
                mcs_coef_l = []
                for k, v in iter(result_dict.items()):
                    closest_smiles, mcs_ratio, mcs_tan, mcs_coef = get_max_mcs([v], k)
                    mcs_dict[k] = {
                        "closest_smiles": closest_smiles,
                        "mcs_ratio": mcs_ratio,
                        "mcs_tan": mcs_tan,
                        "mcs_coef": mcs_coef,
                    }
                    mcs_ratio_l.append(mcs_ratio)
                    mcs_tan_l.append(mcs_tan)
                    mcs_coef_l.append(mcs_coef)

                ref_smi_l = df["predict_smiles"].tolist()
                len_count = 0
                for smi in ref_smi_l:
                    len_count += len(smi)
                avg_smi_length = len_count / len(ref_smi_l)

                avg_cosine_l = []

                for k, v in iter(result_dict.items()):
                    t_ = get_avg_cosine(v, k, 2, 1024)
                    avg_cosine_l.append(t_)
                avg_cosine_l_filtered = [v for v in avg_cosine_l if not math.isnan(v)]
                avg_cosine_l_filtered_is_nan = [
                    v for v in avg_cosine_l if math.isnan(v)
                ]

                avg_mcs_raito = np.sum(mcs_ratio_l) / config.val_smi_num
                avg_mcs_tan = np.sum(mcs_tan_l) / config.val_smi_num
                avg_mcs_coef = np.sum(mcs_coef_l) / config.val_smi_num
                avg_cosine = np.mean(avg_cosine_l_filtered)
                print(
                    f"{avg_mcs_raito=}\n{avg_mcs_tan=}\n{avg_mcs_coef=}\n{avg_cosine=}\n{avg_smi_length=}\n"
                )
                # write above 5 variables to a csv
                # write the header first
                # check if the file eval_results.csv exists
                if not os.path.exists("eval_results.csv"):
                    with open("eval_results.csv", "w") as f:
                        f.write(
                            "avg_mcs_raito,avg_mcs_tan,avg_mcs_coef,avg_cosine,avg_smi_length\n"
                        )
                # write the values
                with open("eval_results.csv", "a+") as f:
                    f.write(
                        f"{avg_mcs_raito},{avg_mcs_tan},{avg_mcs_coef},{avg_cosine},{avg_smi_length}\n"
                    )

                # spectra_model.train()
            return exact, pct_match, avg_score, invalid_ratio


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


class SpectraDataset(Dataset):
    def __init__(self, csv_path, src_sp, trg_sp, config):
        self.src_sp = src_sp
        self.trg_sp = trg_sp
        self.config = config
        df = pd.read_csv(csv_path, low_memory=False)
        # self.model = model.to(self.config.device)
        self.negative_spectra_low = df["negative_spectra_low"].tolist()
        self.negative_spectra_high = df["negative_spectra_high"].tolist()
        self.positive_spectra_low = df["positive_spectra_low"].tolist()
        self.positive_spectra_high = df["positive_spectra_high"].tolist()
        self.ais_data = df["ais_tokens"].tolist()
        self.smiles_list = df["cano_smi"].tolist()

        self.eos_id = self.config.eos_id
        self.sos_id = self.config.sos_id
        self.resolution = 2
        self.minmass = 50.0
        self.maxmass = 499.97

    def __len__(self):
        return len(self.ais_data)

    def __getitem__(self, idx):
        ais_tokens = self.ais_data[idx]

        src_encoded = self.src_sp.EncodeAsIds(ais_tokens)
        trg_encoded = self.trg_sp.EncodeAsIds(ais_tokens)

        src_input = pad_or_truncate(src_encoded + [self.eos_id], self.config)
        trg_input = pad_or_truncate([self.sos_id] + trg_encoded, self.config)
        trg_output = pad_or_truncate(trg_encoded + [self.eos_id], self.config)

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

        smi = self.smiles_list[idx]
        # smiles_tensor_list = [self.model.string2tensor(smiles) for smiles in [smi]]
        # h, _, _, _ = self.model.forward_encoder(smiles_tensor_list)

        return (
            torch.LongTensor(src_input),
            torch.LongTensor(trg_input),
            torch.LongTensor(trg_output),
            smi,
            spectra,
        )


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

    trans_model = Transformer(config).to(device)
    trans_model.load_state_dict(ckpt["model"])
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load("./data/sp/ais_sp.model")
    trg_sp.Load("./data/sp/ais_sp.model")

    # load the datasets
    print("Loading dataloaders...")
    train_set = SpectraDataset(config.spectra_train_path, src_sp, trg_sp, config)
    train_dataloader = DataLoader(
        train_set,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    val_set = SpectraDataset(config.spectra_val_path, src_sp, trg_sp, config)
    val_dataloader = DataLoader(
        val_set,
        batch_size=config.train_batch_size,
        shuffle=False,
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
    # best_val_error = float("inf")
    best_pct_match = 0
    best_avg_score = 0
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
        if not config.in_training:
            exact, pct_match, avg_score, invalid_ratio = compute_reconstruction_trans(
                model, trans_model, val_dataloader, config, src_sp, trg_sp
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
                src_input, trg_input, trg_output, smi, spectra = batch
                src_input = src_input.to(device)
                trg_input = trg_input.to(device)
                trg_output = trg_output.to(device)
                spectra = spectra.to(device).to(torch.bfloat16)

                e_mask, d_mask = make_mask(src_input, trg_input, config)
                encoder_output = (
                    trans_model.forward_encoder(src_input, e_mask).to(device).squeeze(1)
                )
                # print(
                #     f"{encoder_output.shape=},{spectra.shape=},{encoder_output.dtype=},{spectra.dtype=}"
                # )
                preds = model(spectra)
                # print(f"{preds.shape=},{preds.dtype=},{trg_output.dtype=}")
                loss = criterion(preds, encoder_output)
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
                smi,
                spectra,
                e_mask,
                d_mask,
                loss,
                encoder_output,
                preds,
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
            model, trans_model, val_dataloader, config, src_sp, trg_sp
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
