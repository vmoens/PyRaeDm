from raedm import FIVO, TwoBoundariesFPT, RaeDm
from torch.nn.utils import clip_grad_norm_
import tqdm
from torch import nn, autograd
import torch
from raedm.utils import init_weights

if __name__ == '__main__':
    device = 'cuda'
    batch = 128
    sequence_length = 40

    dimx = 7
    hidden = 64
    latent = 64
    niter = 10

    x = torch.randn(batch, sequence_length, dimx)
    a, v, w, t0 = torch.randn(4)
    t0 = abs(t0)
    y = TwoBoundariesFPT(a, v, w, t0).sample((batch, sequence_length))

    embedder = nn.Sequential(nn.Linear(dimx, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
    recurrent = nn.GRUCell(hidden+latent+2, hidden)
    encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2*latent))
    decoder = nn.Sequential(nn.Linear(latent-1, hidden), nn.Tanh(), nn.Linear(hidden, 3))
    prior = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, 2*latent))

    f = RaeDm(embedder, encoder, recurrent, decoder, prior, latent).to(device)
    f.apply(init_weights)
    f.encoder[-1].bias.data[-1] -= 1
    # f(x, y)

    optim = torch.optim.Adam(f.parameters())
    ## TODO: implement mini-batching
    pbar = tqdm.tqdm(range(niter))
    for _ in pbar:
        x, y = x.to(device), y.to(device)
        log_prob = f(x, y)
        (-log_prob.mean()).backward()
        gn = clip_grad_norm_(f.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(f'loss: {log_prob.mean().item(): 4.2f}, grad norm: {gn}')