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


import logging
from tqdm import tqdm
import gc
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import random

import sentencepiece as spm
import glob
import copy
import csv
import pandas as pd


class SpectraEncoder(nn.Module):
    def __init__(
        self,
        precursor_mass_mask_ratio=0.5,
        hidden_size=1024,
        vocab_size=1000,
        embed_size=256,
    ):
        super(SpectraEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.precursor_mass_mask_ratio = precursor_mass_mask_ratio

        # Precursor mass embedding

        self.integer_parts_embedding = nn.Linear(self.vocab_size, self.embed_size)
        self.fractional_parts_embedding = nn.Linear(self.vocab_size, self.embed_size)
        self.intensity_embedding = nn.Linear(self.vocab_size, self.embed_size)

        # Fully connected layer
        self.fc1 = nn.Linear(self.embed_size * 3, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        spec_integer_parts,
        spec_fractional_parts,
        spec_intensity_parts,
        precursor_mass,
    ):
        # Embed precursor mass

        precursor_embedded_0 = precursor_mass.clone().detach().unsqueeze(-1)
        precursor_embedded_0 = F.pad(precursor_embedded_0, (0, self.hidden_size - 1))
        precursor_embedded_1 = torch.zeros_like(precursor_embedded_0)
        # use randon mask to mask the precursor mass embedding, the mask ratio is precursor_mass_mask_ratio
        # the choices are precursor_embedded_0 and precursor_embedded_1
        precursor_embedded = random.choices(
            [precursor_embedded_0, precursor_embedded_1],
            weights=[
                self.precursor_mass_mask_ratio,
                1 - self.precursor_mass_mask_ratio,
            ],
        )[0]

        integer_parts_embedded = self.integer_parts_embedding(spec_integer_parts[0])
        fractional_parts_embedded = self.fractional_parts_embedding(
            spec_fractional_parts[0]
        )
        intensity_embedded = self.intensity_embedding(spec_intensity_parts[0])
        # Concatenate the embeddings
        embedded = torch.cat(
            (integer_parts_embedded, fractional_parts_embedded, intensity_embedded),
            dim=-1,
        )

        embedded = F.relu(self.fc1(embedded))
        embedded = torch.tanh(self.fc2(embedded))
        embedded_with_precursor = torch.stack((precursor_embedded, embedded), dim=1)

        return embedded, embedded_with_precursor
