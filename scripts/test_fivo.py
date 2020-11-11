from raedm import FIVO, TwoBoundariesFPT, RaeDm
from torch.nn.utils import clip_grad_norm_
import tqdm
from torch import nn, autograd
import torch
from raedm.utils import init_weights
from matplotlib import pyplot as plt


if __name__ == '__main__':
    device = 'cuda' # 'cpu', 'cuda:1' ...

    # Fake data
    # in real settings, it might be interesting to slice batches of data when sequences are too long
    n_data = 200
    batch = 16
    sequence_length = 40

    dimx = 7 # regressor size
    hidden = 64 # hidden size
    latent = 3 # latent size (z variable, last dim = t0)
    niter = 1000 # number of iterations for optimisation

    def mavg(x, coef):
        y = torch.empty_like(x)
        ym1 = 0.0
        corr = 1-coef ** torch.arange(1, x.shape[1]+1, dtype=x.dtype, device=x.device)
        corr = corr.view(1, corr.numel(), 1)
        for i in range(x.shape[1]):
            ym1 = y[:,i] = coef * ym1 + (1-coef)*x[:,i]
        return y/corr

    x = torch.randn(n_data, sequence_length, dimx) # random regressors
    a, v, w, t0 = (x @ torch.randn(dimx, 4)).chunk(4, -1) # parameters used for simulation

    t0 = t0/10+0.3
    a -= 0.5
    v = v.sign() * (abs(v)+1)
    w /= 3

    a = mavg(a, 0.7)
    # v = mavg(v, 0.9)
    w = mavg(w, 0.7)
    orig_params = (a, v, w, t0)
    # a += 1 # numerical stability
    y = TwoBoundariesFPT(a.double(), v.double(), w.double(), t0.double()).sample().squeeze(-2).float()
    print(y.shape)

    # embedder = nn.Identity()
    embedder = nn.Sequential(nn.Linear(dimx, hidden), nn.Tanh(),
                             nn.Linear(hidden, hidden), nn.Tanh(),
                             nn.Linear(hidden, dimx))
    recurrent = nn.GRUCell(dimx+latent+2, hidden)
    # encoder (and prior) outputs loc and scale of approximate posterior (and prior)
    encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                            nn.Linear(hidden, hidden), nn.Tanh(),
                            nn.Linear(hidden, 2*latent))
    prior = nn.Sequential(nn.Linear(latent + dimx, hidden), nn.Tanh(),
                          nn.Linear(hidden, hidden), nn.Tanh(),
                          nn.Linear(hidden, 2*latent))
    # decoder outputs a, v, w but not t0, which is taken directly from the posterior because it is encoded as a trucated Gaussian
    decoder = nn.Sequential(
        nn.Linear(latent - 1, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 3))

    raedm = RaeDm(embedder, encoder, recurrent, decoder, prior, latent).to(device)
    # init weights and biases
    raedm.apply(init_weights)
    # make sure that t0 mean is on average = -1
    raedm.encoder[-1].bias.data[latent] -= 1
    # f(x, y)

    optim = torch.optim.Adam(raedm.parameters(), weight_decay=1e-2)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, )
    dataloader_iter = iter(dataloader)
    ## TODO: implement mini-batching
    pbar = tqdm.tqdm(range(niter))
    for i in pbar:
        try:
            _x, _y = next(dataloader_iter)
        except:
            dataloader_iter = iter(dataloader)
            _x, _y = next(dataloader_iter)
        _x, _y = _x.to(device), _y.to(device)
        log_prob = raedm(_x, _y)
        (-log_prob.mean()/sequence_length).backward()
        gn = clip_grad_norm_(raedm.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(f'log_prob: {log_prob.mean().item(): 4.2f}, grad norm: {gn}')

        if (i % 10) == 0:
            x = x[:1].to(device)
            y = y[:1].to(device)
            reg = raedm.fivo_chain(x, y)
            log_w = reg['log_w']
            params = reg['params']
            params = (log_w.exp().unsqueeze(-1)*params).sum(1)
            # print(params.shape)
            # print(orig_params[0].shape)
            f = plt.figure(figsize=(20, 20))
            for i in range(4):
                plt.subplot(2, 4, i + 1)
                p = params[..., i]
                # if i==3:
                #     p = p/10
                plt.plot(p[0].detach().cpu().t(), label=i)
                plt.plot(orig_params[i][0].detach().cpu(), ls='--', label='true value')

                plt.subplot(2, 4, 5+i)
                plt.scatter(p.view(-1).detach().cpu().t(), orig_params[i][0].view(-1).detach().cpu())

            plt.savefig('/tmp/ddm.png')
            plt.close(f)
