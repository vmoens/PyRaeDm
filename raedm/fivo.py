import numpy as np
import torch
from torch import distributions as d, nn, autograd
from torch.nn import functional as F
from .ddm import TwoBoundariesFPT
from .truncated_gaussian import TruncatedGaussian
from .utils import StackedDistributions

softplus_bias = 0.5413248538970947

class FIVO(nn.Module):
    def __init__(self,
                 embedder,
                 encoder,
                 recurrent_model,
                 decoder,
                 prior,
                 latent_dim,
                 data_dim=2,
                 data_dist=TwoBoundariesFPT,
                 n_particles=50,
                 ):
        super().__init__()
        self.embedder = embedder
        self.recurrent_model = recurrent_model
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.data_dist = data_dist
        self.n_particles = n_particles

    def _make_prior(self, z):
        prior_params = self.prior(z)  # add _r ?
        mu_prior, sig_prior = prior_params.chunk(2, -1)
        sig_prior = F.softplus(sig_prior + softplus_bias)
        prior = d.Independent(d.Normal(mu_prior, sig_prior), 1)
        return prior

    def _make_posterior(self, _r, _y):
        posterior_params = self.encoder(_r)
        mu_post, sig_post = posterior_params.chunk(2, -1)
        sig_post = F.softplus(sig_post + softplus_bias)
        posterior = d.Independent(d.Normal(mu_post, sig_post), 1)
        return posterior

    def _make_data_dist(self, z):
        data_params = self.decoder(z)
        data_params = data_params.unbind(-1)  # TODO: make compatible with other data distribution using chunk or split
        data_dist = self.data_dist(*data_params)
        return data_dist

    def forward(self, x, y):
        embed = self.embedder(x)
        x_split = embed.unbind(-2)
        y_split = y.unbind(-2)

        z = torch.zeros(x.shape[0], self.n_particles, self.latent_dim, device=x.device)
        log_ws = []
        log_alphas = []

        log_w = 0.0
        log_p = 0.0
        _r = None
        for i, (_x, _y) in enumerate(zip(x_split, y_split)):
            _x = _x.unsqueeze(0).expand(self.n_particles, *_x.shape).permute(1,0,2)
            _y = _y.unsqueeze(0).expand(self.n_particles, *_y.shape).permute(1,0,2)

            prior = self._make_prior(z)

            _x = torch.cat([_x, z, _y], -1) # Add _r ?
            _r = self.recurrent_model(_x.view(-1, _x.shape[-1]), _r)
            _rview = _r.view(*_x.shape[:2], -1)

            posterior = self._make_posterior(_rview, _y)

            z = posterior.rsample()
            data_dist = self._make_data_dist(z)

            log_alpha = prior.log_prob(z)+data_dist.log_prob(_y) - posterior.log_prob(z)
            _log_p = log_w + log_alpha
            log_p = log_p + _log_p.logsumexp(-1, True)
            log_w = _log_p - _log_p.logsumexp(-1, True)

            z, log_w = self.resample(log_w, z, )

            log_alphas.append(log_alpha)
            log_ws.append(log_w)
        # log_ws = torch.stack(log_w, -1)
        # log_alphas = torch.stack(log_alphas, -1)
        return log_p

    def resample(self, log_w, z):
        log_ess = -(2*log_w).logsumexp(-1, True)
        # resample if ess < N/2
        resample = log_ess.exp()<self.n_particles//2
        if resample.any():
            z_out = torch.empty_like(z)
            log_w_out = torch.full_like(log_w, -np.log(self.n_particles))
            idx_z = resample.unsqueeze(-1).expand_as(z)
            idx_logw = resample.expand_as(log_w)
            log_w_out.masked_scatter(~idx_logw, log_w[~idx_logw])

            ## TODO: systematic resampling
            new_idx = torch.multinomial(log_w[resample.squeeze(-1)].exp(), self.n_particles, True)
            z_out_to_replace = z[resample.squeeze(-1)]
            new_idx = new_idx.unsqueeze(-1).expand(z_out_to_replace.shape[0], self.n_particles, z_out_to_replace.shape[-1])
            z_out_to_replace = z_out_to_replace.gather(index=new_idx, dim=-2)

            z_out.masked_scatter_(~idx_z, z[~idx_z])
            z_out.masked_scatter_(idx_z, z_out_to_replace)
            z = z_out
            log_w = log_w_out

        return z, log_w

class RaeDm(FIVO):
    rt_margin = 0.0

    def _make_posterior(self, _r, _y):
        posterior_params = self.encoder(_r)
        mu_post, sig_post = posterior_params.chunk(2, -1)
        sig_post = F.softplus(sig_post + softplus_bias)
        mu_post_t0 = mu_post[..., -1:]
        sig_post_t0 = sig_post[..., -1:]
        t0_dist = d.TransformedDistribution(
            TruncatedGaussian(mu_post_t0, sig_post_t0, lim_sup=10*_y[..., :1]-self.rt_margin), # upper bound of t0 dist is RT
            d.AffineTransform(0, 0.1, cache_size=1))
        # print(t0_dist.rsample((10,)).mean(0) - _y[..., :1])

        normals = d.Normal(mu_post[..., :-1], sig_post[..., :-1])
        posterior = d.Independent(StackedDistributions([normals, t0_dist],
                                                       univariates=[False, False],
                                                       sort_index=[list(range(mu_post.shape[-1]-1)), [mu_post.shape[-1]-1,]]), 1)
        return posterior

    def _make_data_dist(self, z):
        t0 = z[..., -1]
        z = z[..., :-1]
        data_params = self.decoder(z)
        data_params = data_params.unbind(-1)  # TODO: make compatible with other data distribution using chunk or split
        data_dist = self.data_dist(*data_params, t0)
        return data_dist
