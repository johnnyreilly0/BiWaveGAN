import torch
import argparse
from torch.utils.data import DataLoader


# a mess

# class Train:
#     """
#     General purpose object for training, checkpointing, loading, saving, logging.
#     Needs model, optimiser, train set, anything else?
#     """
#     def __init__(self, model, optimEG, optimD, train_loader, val_loader, args):
#         self.model = model
#         self.optimEG = self.optimEG
#         self.optimD = optimD
#         self.train_loader = train_loader,
#         self.val_loader = val_loader
#         self.args = args
#
#     def train(self, iters):
#         model.train()
#         one = torch.tensor(1, dtype=torch.float, device=device)
#         neg_one = one * -1
#
#         for _ in range(D_ITERS):
#             model.D.zero_grad()
#             # grab next batch
#             try:
#                 real = next(train_iter).to(device)
#             except StopIteration:
#                 train_iter = iter(train_loader)
#                 real = next(train_iter).to(device)
#             cur_batch_size = real.shape[0]
#
#             # generate real and fake latent vectors
#             z_real = model.encoder(real)
#             z_fake = torch.Tensor(cur_batch_size, LATENT_DIM).uniform_(-1, 1).to(device)  # TODO: incl. normal noise
#             fake = model.generator(z_fake)
#
#             # compute D loss
#             D_real = model.discriminator(real, z_real).reshape(-1)
#             D_fake = model.discriminator(fake, z_fake).reshape(-1)
#             grad_penalty = model.D.gradient_penalty(model.discriminator, real, z_real, fake, z_fake, device=device)
#             loss_D = -1 * (torch.mean(D_real) - torch.mean(D_fake)) + LAMBDA_GP * grad_penalty
#             # update D
#             loss_D.backward(retain_graph=True)
#             optimD.step()
#
#         # stop D from updating
#         # TODO: this is ugly, abstract away
#         for p in model.G.parameters():
#             p.requires_grad = True
#         for p in model.E.parameters():
#             p.requires_grad = True
#         for p in model.D.parameters():
#             p.requires_grad = False
#
#         # G iteration
#         model.E.zero_grad()
#         model.G.zero_grad()
#
#     def get_next_batch(self, iter):
#         try:
#             batch = next(iter).to(self.device)
#         except StopIteration:
#             iter = iter(self.train_loader)
#             batch = next(train_iter).to(self.device)
#         return batch


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
    parser.add_argument('--val_size', default=1000, type=int)
    return parser.parse_args()


def get_next_batch(iter, loader, device):
    try:
        batch = next(iter).to(device)
    except StopIteration:
        iter = iter(loader)
        batch = next(iter).to(device)
    return batch
