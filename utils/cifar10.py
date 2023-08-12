import argparse
import os
import torch
import numpy as np
import random
from lightning.fabric import seed_everything


def init_envir():
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="DSVDD")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--normal_class', type=int, default=0)
    parser.add_argument('--pre_epochs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--radio', type=float, default=0.0)
    parser.add_argument('--objective', type=str, default="one-class")
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--log_path", type=str, default=os.getcwd())
    parser.add_argument("--bash_log_name", type=str, default="bash-logv3")

    args = parser.parse_args()
    seed_everything(args.seed, workers=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    return args
