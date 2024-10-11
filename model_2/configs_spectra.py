from dataclasses import dataclass, field
import torch
import argparse
import random


@dataclass
class TrainingConfig:
    training_data_path: str = "./data/training_1068070.csv"
    val_data_path: str = "./data/val_1000.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # From device
    train_batch_size: int = 128
    train_val_ratio: float = 0.8
    kl_weight: float = 1.0
    gradient_accumulation_steps: int = 1
    seed: int = 38  # random.randint(0, 1000)
    training_id: str = "default"
    resume_from_checkpoint: str = None
    src_vocab_size: int = 3365  # 5933
    trg_vocab_size: int = 3365  # 5933
    d_model: int = 1024
    clip_grad: int = 50
    kl_start: int = 0
    kl_w_start: float = 0
    kl_w_end: float = 0.05
    lr_start: float = 0.0001  # 0.0002
    lr_n_period: int = 10
    lr_n_restarts: int = 100  # 30 --> n_epoch=300
    lr_n_mult: int = 1
    lr_end: float = 0.00005
    ckpt_dir: str = "checkpoints"
    mixed_precision: str = "bf16"  # or 'fp16' for older GPUs
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    seq_len: int = 150
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    # d_k: int = d_model // num_heads
    drop_out_rate: int = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    SP_DIR: str = "./data/sp"
    val_smi_num: int = 200
    compute_val: str = None
    in_training: bool = True
    spectra_val_path: str = "./data/spectra/spectra_val.csv"
    spectra_train_path: str = "./data/spectra/spectra_train.csv"
    spectra_test_path: str = "./data/spectra/spectra_test.csv"
    spec_ckpt: str = (
        "./checkpoints/autoencoder-1m-spectra-translation-model-1_num_epoch_56_best_val_error_0.15165482461452484.pth"
    )
    # trans_ckpt: str = (
    #     "./checkpoints/no-vae-5m_batch_size_256_num_epoch_1000_val_pct_0.744_best_avg_score_0.91.pth"
    # )
    trans_ckpt: str = (
        "./checkpoints/no-vae-1m-training-3_batch_size_128_num_epoch_1000_val_avg_score_0.92_best_pct_0.674.pth"
    )
    spectra_vocab_size: int = 1000
    spectra_embed_size: int = 256
    precursor_mass_mask_ratio: float = 1.0  # use all the precursor mass when 1
    # vocab_path: str = "./data/vocab-5m-filtered.nb"
    vocab_path: str = "./data/vocab_1213383.nb"
    use_tanh: bool = False
    ignore_vae: bool = False
    freeze_embeddings: bool = False
    q_cell: str = "gru"
    q_bidir: bool = True
    q_d_h: int = 512
    q_n_layers: int = 3
    q_dropout: float = 0.0
    d_cell: str = "gru"
    d_n_layers: int = 3
    d_dropout: float = 0.0
    d_z: int = 512
    d_d_h: int = 512
    fc_h: int = 512
    fc_use_relu: bool = False
    fc_use_bn: bool = False
    number_of_variants: int = 100

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Training configuration")
        parser.add_argument("--config", type=str, help="Path to configuration file")

        for field in cls.__dataclass_fields__.values():
            if field.type == bool:
                parser.add_argument(
                    f"--{field.name}", action="store_true", default=field.default
                )
                parser.add_argument(
                    f"--not_{field.name}", action="store_false", dest=field.name
                )
            else:
                parser.add_argument(
                    f"--{field.name}", type=field.type, default=field.default
                )

        args = parser.parse_args()

        config_dict = {}
        if args.config:
            with open(args.config, "r") as f:
                config_dict = json.load(f)

        # Update with command line arguments, only if they're not None
        config_dict.update(
            {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        )

        # Fill in any missing values with class defaults
        for field in cls.__dataclass_fields__.values():
            if field.name not in config_dict:
                config_dict[field.name] = field.default

        return cls(**config_dict)
