from raedm import FIVO, TwoBoundariesFPT, RaeDm
from torch.nn.utils import clip_grad_norm_
import tqdm
from torch import nn, autograd
import torch
from raedm.utils import init_weights

if __name__ == '__main__':
    device = 'cuda' # 'cpu', 'cuda:1' ...

    # Fake data
    # in real settings, it might be interesting to slice batches of data when sequences are too long
    batch = 128
    sequence_length = 40

    dimx = 7 # regressor size
    hidden = 64 # hidden size
    latent = 64 # latent size (z variable, last dim = t0)
    niter = 10 # number of iterations for optimisation

    x = torch.randn(batch, sequence_length, dimx) # random regressors
    a, v, w, t0 = torch.randn(4) # parameters used for simulation
    t0 = abs(t0)
    y = TwoBoundariesFPT(a, v, w, t0).sample((batch, sequence_length))

    embedder = nn.Sequential(nn.Linear(dimx, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
    recurrent = nn.GRUCell(hidden+latent+2, hidden)
    # encoder (and prior) outputs loc and scale of approximate posterior (and prior)
    encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2*latent))
    prior = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, 2*latent))
    # decoder outputs a, v, w but not t0, which is taken directly from the posterior because it is encoded as a trucated Gaussian
    decoder = nn.Sequential(nn.Linear(latent-1, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    raedm = RaeDm(embedder, encoder, recurrent, decoder, prior, latent).to(device)
    # init weights and biases
    raedm.apply(init_weights)
    # make sure that t0 mean is on average = -1
    raedm.encoder[-1].bias.data[latent] -= 1
    # f(x, y)

    optim = torch.optim.Adam(raedm.parameters())
    ## TODO: implement mini-batching
    pbar = tqdm.tqdm(range(niter))
    for _ in pbar:
        x, y = x.to(device), y.to(device)
        log_prob = raedm(x, y)
        (-log_prob.mean()).backward()
        gn = clip_grad_norm_(raedm.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(f'loss: {log_prob.mean().item(): 4.2f}, grad norm: {gn}')