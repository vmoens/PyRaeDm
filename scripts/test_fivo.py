from raedm.fivo import FIVO
import tqdm
from raedm.ddm import TwoBoundariesFPT
from torch import nn, autograd
import torch

if __name__ == '__main__':
    device = 'cuda'
    batch = 128
    sequence_length = 40

    dimx = 7
    hidden = 64
    latent = 64
    niter = 10

    x = torch.randn(batch, sequence_length, dimx)
    y = TwoBoundariesFPT(*(torch.randn(4)/10)).sample((batch, sequence_length))

    embedder = nn.Sequential(nn.Linear(dimx, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
    recurrent = nn.GRUCell(hidden+latent+2, hidden)
    encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2*latent))
    decoder = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, 4))
    prior = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, 2*latent))

    f = FIVO(embedder, encoder, recurrent, decoder, prior, latent).to(device)
    # f(x, y)

    optim = torch.optim.Adam(f.parameters())
    ## TODO: implement mini-batching
    pbar = tqdm.tqdm(range(niter))
    for _ in pbar:
        x, y = x.to(device), y.to(device)
        log_prob = f(x, y)
        (-log_prob.mean()).backward()
        optim.step()
        optim.zero_grad()
        pbar.set_description(f'loss: {log_prob.mean().item(): 4.2f}')