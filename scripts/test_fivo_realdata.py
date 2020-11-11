from raedm import FIVO, TwoBoundariesFPT, RaeDm
from torch.nn.utils import clip_grad_norm_
import tqdm
from torch import nn, autograd
import torch
from raedm.utils import init_weights
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':
    device = 'cuda' # 'cpu', 'cuda:1' ...

    data = pd.read_csv('scripts/data/ILTrialData.csv')
    subj_data_x = []
    subj_data_y = []
    for i in range(20):
        _subj_data = data[data.sbj==i+1]
        _y = torch.tensor(_subj_data[['rt', 'perf']].to_numpy())
        _y[..., 0] /= 1000
        _y[..., 1] *= 2
        _y[..., 1] -= 1
        subj_data_y.append(_y)
        _x = torch.tensor(_subj_data[['cond', 'block']].to_numpy(), dtype=torch.float)
        subj_data_x.append(_x)

    subj_data_y = torch.stack(subj_data_y, 0).to(torch.float)
    subj_data_y[~torch.isfinite(subj_data_y)] = 1.0 # quick fix
    subj_data_x = torch.stack(subj_data_x, 0).to(torch.float)
    rand_input = torch.empty_like(subj_data_x[..., -1]).bernoulli_().mul_(2).add_(-1)
    subj_data_y[..., 1] *= rand_input # if correct, then y[1] is = to rand_input
    subj_data_x = torch.cat([subj_data_x, rand_input.unsqueeze(-1)], -1)
    subj_data_x -= subj_data_x.view(-1, subj_data_x.shape[-1]).mean(0)
    subj_data_x /= subj_data_x.view(-1, subj_data_x.shape[-1]).std(0).clamp_min(1e-6)
    dataset = torch.utils.data.TensorDataset(subj_data_x, subj_data_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, )

    # Fake data
    # in real settings, it might be interesting to slice batches of data when sequences are too long

    n_epochs = 1000
    dimx = subj_data_x.shape[-1] # regressor size
    hidden = 64 # hidden size
    latent = 3 # latent size (z variable, last dim = t0)
    niter = 10 # number of iterations for optimisation

    embedder = nn.Sequential(nn.Linear(dimx, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
    recurrent = nn.GRUCell(hidden+latent+2, hidden)
    # encoder (and prior) outputs loc and scale of approximate posterior (and prior)
    encoder = nn.Sequential(
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 2*latent))
    prior = nn.Sequential(
        nn.Linear(latent+hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 2*latent))
    # decoder outputs a, v, w but not t0, which is taken directly from the posterior because it is encoded as a trucated Gaussian
    decoder = nn.Sequential(nn.Linear(latent-1, hidden),
                            nn.Tanh(),
                            nn.Linear(hidden, 3))

    raedm = RaeDm(embedder, encoder, recurrent, decoder, prior, latent).to(device)
    # init weights and biases
    raedm.apply(init_weights)
    # make sure that t0 mean is on average = -1
    # raedm.encoder[-1].bias.data[latent] -= 1
    # f(x, y)

    optim = torch.optim.Adam(raedm.parameters(), weight_decay=1e-2)
    for j in range(n_epochs):
        pbar = tqdm.tqdm(enumerate(dataloader))
        for i, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            idx = torch.randint(x.shape[1]-32, (1,)).item()
            x = x[:, idx:idx+32]
            y = y[:, idx:idx+32]
            sequence_length = x.shape[1]
            log_prob = raedm(x, y)
            (-log_prob.mean()/sequence_length).backward()
            gn = clip_grad_norm_(raedm.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
            pbar.set_description(f'epoch{j} // log_prob: {log_prob.mean().item(): 4.2f}, grad norm: {gn}')

        reg = raedm.fivo_chain(x, y)
        log_w = reg['log_w']
        params = reg['params']
        params = (log_w.exp().unsqueeze(-1)*params).sum(1)
        f = plt.figure(figsize=(20, 20))
        for i in range(4):
            plt.subplot(2, 4, i + 1)
            p = params[..., i]
            # if i==3:
            #     p = p/10
            plt.plot(p[0].detach().cpu().t(), label=i)
            # plt.plot(orig_params[i][0].detach().cpu(), ls='--', label='true value')

            # plt.subplot(2, 4, 5+i)
            # plt.scatter(p.view(-1).detach().cpu().t(), orig_params[i].view(-1).detach().cpu())

        plt.savefig('/tmp/ddm.png')
        plt.close(f)
