from torch.optim.lr_scheduler import _LRScheduler
import random
import re
import torch
import numpy as np
import pandas as pd
from collections import UserList, defaultdict

import math
from bisect import bisect_right


import atomInSmiles
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, random_split, Dataset
from rdkit import RDLogger
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    DataStructs,
    MolFromSmiles,
    MolToSmiles,
)
from rdkit import Chem
import os
import glob

# REGEX_SML = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""


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


class SMILESDataset(Dataset):
    def __init__(self, data_path, real_dataset):
        if not real_dataset:
            data = pd.read_csv(data_path, low_memory=False)[:128]
        else:
            data = pd.read_csv(data_path, low_memory=False)
        rand_dataset = data["rand_smi"]
        cano_dataset = data["cano_smi"]
        self.dataset = []
        for i in range(len(rand_dataset)):
            item = [rand_dataset[i], cano_dataset[i]]
            self.dataset.append(item)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rand_smi, cano_smi = self.dataset[idx]
        return rand_smi, cano_smi


# Function to create batches for training and testing
def create_batches(dataset, batch_size, ratio=1.0):
    # batch_size = config.trans_batch_size
    train_ratio = ratio
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    if ratio == 1.0:
        train_dataset = dataset
        test_dataset = dataset[1]
    elif ratio == 0.0:
        train_dataset = dataset[1]
        test_dataset = dataset
    else:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    def collate_fn(batch):
        randomized_smiles_list = []
        canonical_smiles_list = []

        for item in batch:
            randomized_smiles_list.append(item[0])
            canonical_smiles_list.append(item[1])

        # Tokenize SMILES strings and get their lengths
        smiles_lengths = [
            len(smiles_tokenize(smiles)) for smiles in randomized_smiles_list
        ]

        # Sort all elements based on the lengths of tokenized SMILES strings (long to short)
        # it has be done otherwise forward_encoder will fail
        sorted_indices = sorted(
            range(len(smiles_lengths)), key=lambda i: -smiles_lengths[i]
        )

        randomized_smiles_list = [randomized_smiles_list[i] for i in sorted_indices]
        canonical_smiles_list = [canonical_smiles_list[i] for i in sorted_indices]
        return (
            randomized_smiles_list,
            canonical_smiles_list,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


class SMILESDataset_regression(Dataset):
    def __init__(self, data_path, real_dataset):
        if not real_dataset:
            data = pd.read_csv(data_path, low_memory=False)[:128]
        else:
            data = pd.read_csv(data_path, low_memory=False)
        rand_dataset = data["rand_smi"]  # no need unique, it's wrangled
        cano_dataset = data["cano_smi"]
        mol_wt_dataset = data["mol_wt"]
        logp_dataset = data["log_p"]
        num_hbd_dataset = data["num_hbd"]
        num_hba_dataset = data["num_hba"]
        tpsa_dataset = data["tpsa"]
        num_rot_bonds_dataset = data["num_rot_bonds"]
        qed_dataset = data["qed"]
        sa_score_dataset = data["sa_score"]
        druglike_dataset = data["druglike"]
        self.dataset = []
        for i in range(len(rand_dataset)):
            item = [
                rand_dataset[i],
                cano_dataset[i],
                mol_wt_dataset[i],
                logp_dataset[i],
                num_hbd_dataset[i],
                num_hba_dataset[i],
                tpsa_dataset[i],
                num_rot_bonds_dataset[i],
                qed_dataset[i],
                sa_score_dataset[i],
                druglike_dataset[i],
            ]
            self.dataset.append(item)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (
            rand_smi,
            cano_smi,
            mol_wt,
            logp,
            num_hbd,
            num_hba,
            tpsa,
            num_rot,
            qed,
            sa_score,
            druglike,
        ) = self.dataset[idx]
        return (
            rand_smi,
            cano_smi,
            mol_wt,
            logp,
            num_hbd,
            num_hba,
            tpsa,
            num_rot,
            qed,
            sa_score,
            druglike,
        )


# Function to create batches for training and testing
def create_batches_regression(dataset, batch_size, ratio=1.0):
    # batch_size = config.trans_batch_size
    train_ratio = ratio  # config.train_val_ratio
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    def collate_fn(batch):
        randomized_smiles_list = []
        canonical_smiles_list = []
        mol_wt = []
        logp = []
        num_hbd = []
        num_hba = []
        tpsa = []
        num_rot = []
        qed = []
        sa_score = []
        druglike = []

        for item in batch:
            randomized_smiles_list.append(item[0])
            canonical_smiles_list.append(item[1])
            mol_wt.append(item[2])
            logp.append(item[3])
            num_hbd.append(item[4])
            num_hba.append(item[5])
            tpsa.append(item[6])
            num_rot.append(item[7])
            qed.append(item[8])
            sa_score.append(item[9])
            druglike.append(item[10])

        # Tokenize SMILES strings and get their lengths
        smiles_lengths = [
            len(smiles_tokenize(smiles)) for smiles in randomized_smiles_list
        ]

        # Sort all elements based on the lengths of tokenized SMILES strings (long to short)
        # it has be done otherwise forward_encoder will fail
        sorted_indices = sorted(
            range(len(smiles_lengths)), key=lambda i: -smiles_lengths[i]
        )

        randomized_smiles_list = [randomized_smiles_list[i] for i in sorted_indices]
        canonical_smiles_list = [canonical_smiles_list[i] for i in sorted_indices]
        return (
            randomized_smiles_list,
            canonical_smiles_list,
            mol_wt,
            logp,
            num_hbd,
            num_hba,
            tpsa,
            num_rot,
            qed,
            sa_score,
            druglike,
        )

    if ratio == 1.0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = None
    elif ratio == 0.0:
        train_loader = None
        val_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


def compute_reconstruction_trans(model, dataloader):
    samples = []
    all_samples = pd.DataFrame()
    model.eval()

    for i, batch in enumerate(dataloader):
        rand_smi, cano_smi = batch
        h, recon_loss = model(rand_smi, cano_smi)
        current_samples = model.generate_smiles(h, n_batch=len(cano_smi))

        samples = pd.DataFrame(
            {
                "REAL_CAN": cano_smi,  # real_smiles,
                "GENERATED": current_samples,
                # "RANDOMIZED_CONVERTED": rand_smiles_converted,
            }
        )
        samples["MATCH"] = samples["REAL_CAN"] == samples["GENERATED"]

        score_list = []
        # for j in range(len(cano_smi)):
        #     score = CalcFpTc(cano_smi[j], current_samples[j])
        #     score_list.append(score)
        for j in range(len(cano_smi)):
            try:
                score = CalcFpTc(cano_smi[j], current_samples[j])
            except Exception as e:
                print(
                    f"Error calculating score for SMILES pair ({cano_smi[i]}, {current_samples[i]}): {e}"
                )
                score = 0.0
            score_list.append(score)

        samples["score"] = score_list

        # Concatenate the new dataframe to the existing dataframe
        all_samples = pd.concat([all_samples, samples], ignore_index=True)

    # Ensure all_samples has the desired columns structure
    all_samples = all_samples[["REAL_CAN", "GENERATED", "MATCH", "score"]]

    # Save all_samples dataframe to a csv file
    all_samples.to_csv("./checkpoints/val_result.csv", index=False)
    # print the avg_score and pct_match
    total = len(all_samples)
    match = all_samples["MATCH"].sum()
    pct_match = all_samples["MATCH"].mean()
    # average score
    score_list = all_samples["score"].tolist()
    avg_score = sum(score_list) / len(score_list)
    # make avg_score round to 2 decimal places
    avg_score = round(avg_score, 2)
    # make pct_match round to 4 decimal places
    pct_match = round(pct_match, 4)
    print(f"Total: {total} || Matched: {match} || Matched Ratio {pct_match*100:.2f}%\n")
    print(f"Average Score: {avg_score}")
    del rand_smi, cano_smi, h, recon_loss, current_samples, samples, score_list
    torch.cuda.empty_cache()
    model.train()

    return pct_match, avg_score


def compute_reconstruction_trans_regression(model, dataloader):
    samples = []
    all_samples = pd.DataFrame()
    model.eval()

    for i, batch in enumerate(dataloader):
        (rand_smi, cano_smi, _, _, _, _, _, _, _, _, _) = batch
        # true_properties_tensor = torch.stack(                    [
        #                 mol_wt,
        #                 logp,
        #                 num_hbd,
        #                 num_hba,
        #                 tpsa,
        #                 num_rot,
        #                 qed,
        #                 sa_score,
        #                 druglike,])

        h, _ = model(rand_smi, cano_smi)
        current_samples = model.generate_smiles(h, n_batch=len(cano_smi))

        samples = pd.DataFrame(
            {
                "REAL_CAN": cano_smi,  # real_smiles,
                "GENERATED": current_samples,
                # "RANDOMIZED_CONVERTED": rand_smiles_converted,
            }
        )
        samples["MATCH"] = samples["REAL_CAN"] == samples["GENERATED"]

        score_list = []
        for i in range(len(cano_smi)):
            score = CalcFpTc(cano_smi[i], current_samples[i])
            score_list.append(score)

        samples["score"] = score_list

        # Concatenate the new dataframe to the existing dataframe
        all_samples = pd.concat([all_samples, samples], ignore_index=True)

    # Ensure all_samples has the desired columns structure
    all_samples = all_samples[["REAL_CAN", "GENERATED", "MATCH", "score"]]

    # Save all_samples dataframe to a csv file
    all_samples.to_csv("./checkpoints/val_result.csv", index=False)
    # print the avg_score and pct_match
    total = len(all_samples)
    match = all_samples["MATCH"].sum()
    pct_match = all_samples["MATCH"].mean()
    # average score
    score_list = all_samples["score"].tolist()
    avg_score = sum(score_list) / len(score_list)
    # make avg_score round to 2 decimal places
    avg_score = round(avg_score, 2)
    # make pct_match round to 4 decimal places
    pct_match = round(pct_match, 4)
    print(f"Total: {total} || Matched: {match} || Matched Ratio {pct_match*100:.2f}%\n")
    print(f"Average Score: {avg_score}")
    model.train()

    return pct_match, avg_score


def set_torch_seed_to_all_gens(_):
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


class SS:
    bos = "<bos>"
    eos = "<eos>"
    pad = "<pad>"
    unk = "<unk>"


def smiles_tokenize(smiles):
    # return smiles
    try:
        smiles = normalize_smiles(smiles, True, True)
        tmp = MolFromSmiles(smiles)
        for atom in tmp.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        smi_1 = MolToSmiles(tmp)
        ais_tokens = atomInSmiles.encode(smi_1, with_atomMap=True)
        return ais_tokens.split(" ")
    except Exception as e:
        print(e)
        print(f"Error in {smiles}")


# this class is integrated into the Vocab.nb
# after loading vacab.nb into the model, the change of this class has no effect
# to make a change, we need to regenerate a new vocab.nb with the updated codes
class SmilesVocab:
    @classmethod
    def from_data(
        cls, data, tokenizer=smiles_tokenize, max_smiles=1000000, *args, **kwargs
    ):
        chars = set()
        for string in data[:max_smiles]:
            # print(string)
            chars.update(tokenizer(string))

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, tokenizer=smiles_tokenize, ss=SS):
        if (
            (ss.bos in chars)
            or (ss.eos in chars)
            or (ss.pad in chars)
            or (ss.unk in chars)
        ):
            raise ValueError("SS in chars")

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in self.tokenizer(string)]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ""
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = " ".join([self.id2char(id) for id in ids])

        return string


class SmilesOneHotVocab(SmilesVocab):
    def __init__(self, *args, **kwargs):
        super(SmilesOneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))


class Logger(UserList):
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in data or []:
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return Logger(self.data[key])
        else:
            ldata = self.sdata[key]
            if isinstance(ldata[0], dict):
                return Logger(ldata)
            else:
                return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)

    def save(self, path):
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)


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
