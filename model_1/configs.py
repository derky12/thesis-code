from dataclasses import dataclass, field
import torch
import argparse
import random

# generate a random int


@dataclass
class TrainingConfig:
    # Model parameters
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
    freeze_embeddings: bool = False
    fc_h: int = 512
    fc_use_relu: bool = False
    fc_use_bn: bool = False
    ignore_vae: bool = False
    use_tanh: bool = False
    clip_grad: int = 50
    kl_start: int = 0
    kl_w_start: float = 0
    kl_w_end: float = 0.05
    n_last: int = 1000
    n_jobs: int = 1
    n_workers: int = 1
    real_dataset: bool = (
        True  # for testing purposes, used to switch between real and shortened datasets
    )
    resume_from_ckpt: bool = False
    # model_load: Optional[str] = None
    # config_load: Optional[str] = None
    start_epoch: int = 0
    save_frequency: int = 2048
    # vocab_load: Optional[str] = None
    n_regression_outputs: int = 10
    context_d: int = 512
    best_pct_match: float = 0.0
    best_avg_score: float = 0.0
    save_pct_frequency: int = 2048
    training_data_path: str = "./data/df_train.csv"
    val_data_path: str = "./data/df_val_1k.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # From device
    trans_batch_size: int = 256
    train_val_ratio: float = 0.8
    kl_weight: float = 1.0
    gradient_accumulation_steps: int = 1
    ckpt_dir: str = "checkpoints"
    mixed_precision: str = "bf16"  # or 'fp16' for older GPUs
    seed: int = 38  # random.randint(0, 1000)
    lr_start: float = (
        0.00008  # 0.00004  # 0.00008  # 0.00012  # 0.00018  # 细化 0.00015 # 初始 0.0003
    )
    lr_n_period: int = 10
    lr_n_restarts: int = 100  # 30 --> n_epoch=300
    lr_n_mult: int = 1
    lr_end: float = (
        0.00001  # 0.00001  # 0.00003  # 0.00007  # 细化 0.00003 # 初始 0.0001
    )
    training_id: str = "default"
    vocab_path: str = "./data/vocab-filtered.nb"
    resume_from_checkpoint: str = None
    regression_loss_weight: float = 1.0

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
                    f"--no_{field.name}", action="store_false", dest=field.name
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
