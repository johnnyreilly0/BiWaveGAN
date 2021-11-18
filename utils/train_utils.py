import torch
import argparse
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('logdir')
    parser.add_argument('--slice_len', default=32768, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_iters', default=20000, type=int)
    parser.add_argument('--lambda_gp', default=10, type=float)
    parser.add_argument('--d_iters', default=5, type=int)
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--model_size', default=32, type=int)
    parser.add_argument('--discrim_filters', default=512, type=int)
    parser.add_argument('--z_discrim_depth', default=2, type=int)
    parser.add_argument('--joint_discrim_depth', default=3, type=int)
    parser.add_argument('--phaseshuffle_rad', default=2, type=int)
    parser.add_argument('--sample_rate', default=250000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--val_split', default=0.15, type=int)
    return parser.parse_args()


def get_next_batch(iter, loader, device):
    try:
        batch = next(iter).to(device)
    except StopIteration:
        iter = iter(loader)
        batch = next(iter).to(device)
    return batch
