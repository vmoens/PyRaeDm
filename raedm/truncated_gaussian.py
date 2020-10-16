import torch
from torch.distributions.utils import clamp_probs
from torch import distributions
from math import sqrt, log, pi
from numbers import Number
import numpy as np
from scipy.stats import truncnorm

def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.empty(sample_shape, device=lower_bound.device)
    done = torch.zeros(sample_shape, device=lower_bound.device, dtype=torch.bool)
    lower_bound = lower_bound.expand(sample_shape)
    upper_bound = upper_bound.expand(sample_shape)
    count = 0
    while not done.all() and count<50:
        count += 1
        idx = torch.logical_not(done)
        r = torch.rand(idx.sum(), device=lower_bound.device)
        _upper = upper_bound[idx]
        _lower = lower_bound[idx]
        proposed_x = _lower + r * (_upper - _lower)
        log_prob_accept = torch.empty_like(proposed_x)
        id1 = (_upper * _lower).lt(0.0)
        id2 = (torch.logical_not(id1) + (_upper < 0.0)).to(torch.bool)
        id3 = (torch.logical_not(id1) + torch.logical_not(id2)).to(torch.bool)
        log_prob_accept[id1] = -0.5 * proposed_x[id1]**2
        log_prob_accept[id2] = 0.5 * (_upper[id2]**2 - proposed_x[id2]**2)
        log_prob_accept[id3] = 0.5 * (_lower[id3]**2 - proposed_x[id3]**2)
        prob_accept = clamp_probs(log_prob_accept.exp())
        accept = torch.bernoulli(prob_accept).bool() & ~done[idx]
        if accept.any():
            x[idx][accept] = proposed_x[accept]
            done[idx] |= accept
    if count==50 and not done.all():
        raise Exception("Failed to converge")
    return x



def _normal_cdf(value, loc=0.0, scale=1.0):
    # return _normal_log_cdf(value, loc, scale).exp()
    if isinstance(scale, Number):
        scale = torch.full_like(value, scale)
    if isinstance(loc, Number):
        loc = torch.full_like(value, loc)
    out = torch.empty_like(value)
    id_finite = torch.isfinite(value)
    if not id_finite.all():
        out[torch.logical_not(id_finite)] = (torch.sign(value[torch.logical_not(id_finite)]) + 1)/2
        if id_finite.any():
            out[id_finite] = 0.5 * (1 + torch.erf((value[id_finite] - loc[id_finite]) * scale[id_finite].reciprocal() / sqrt(2)))
        return out
    out = 0.5 * (1 + torch.erf((value - loc) * scale.reciprocal() / sqrt(2)))
    return out

def _normal_log_cdf(value, loc=0.0, scale=1.0):
    if isinstance(scale, Number):
        scale = torch.full_like(value, scale)
    return torch.log1p(torch.erf(value-loc)*scale.reciprocal()/sqrt(2))-log(2)

def _normal_log_pdf(value, loc=0.0, scale=1.0):
    var = (scale ** 2)
    log_scale = log(scale) if isinstance(scale, Number) else scale.log()
    return -((value - loc) ** 2) / (2 * var) - log_scale - log(sqrt(2 * pi))

class TruncatedGaussian(distributions.Distribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor,
                 lim_inf=None, lim_sup=None, numerically_safe=True):
        super().__init__()
        self.loc = loc
        self.scale = scale

        if numerically_safe:
            self.scale.data.clamp_min_(1e-3)

        no_lim_inf = lim_inf is None
        no_lim_sup = lim_sup is None
        if no_lim_inf:
            lim_inf = -np.inf
        if no_lim_sup:
            lim_sup = np.inf
        if isinstance(lim_inf, Number):
            self.lim_inf = torch.full_like(loc, lim_inf)
        elif not isinstance(lim_inf, torch.Tensor):
            self.lim_inf = torch.tensor(lim_inf, device=loc.device, dtype=loc.dtype)
        else:
            self.lim_inf = lim_inf.expand_as(loc)

        if isinstance(lim_sup, Number):
            self.lim_sup = torch.full_like(loc, lim_sup)
        elif not isinstance(lim_sup, torch.Tensor):
            self.lim_sup = torch.tensor(lim_sup, device=loc.device, dtype=loc.dtype)
        else:
            self.lim_sup = lim_sup.expand_as(loc)

        if no_lim_inf:
            self.alpha = self.lim_inf
        else:
            self.alpha = (self.lim_inf - self.loc) / self.scale
        if no_lim_sup:
            self.beta = self.lim_sup
        else:
            self.beta = (self.lim_sup - self.loc) / self.scale
        if torch.isnan(self.alpha).any():
            idnan = torch.isnan(self.alpha)
            print('loc: ', self.loc[idnan])
            print('scale: ', self.scale[idnan])
            raise Exception('nan encountered in alpha')

        self.numerically_safe = numerically_safe
        if self.numerically_safe:
            l1 = 40
            l2 = 15
            id_unstable_ = torch.isfinite(self.beta) & \
                           torch.isfinite(self.alpha) & \
                           (self.alpha < -l1) & \
                           (self.beta < -l2)
            self.alpha.data[id_unstable_] = -l1
            self.beta.data[id_unstable_] = -l2
            self.loc.data[id_unstable_] = self.lim_inf.data[id_unstable_] \
                                          + self.alpha.data[id_unstable_] * self.scale.data[id_unstable_]

            id_unstable_ = torch.isfinite(self.beta) & \
                           torch.isfinite(self.alpha) & \
                           (self.alpha > l2) & \
                           (self.beta > l1)
            self.alpha.data[id_unstable_] = l2
            self.beta.data[id_unstable_] = l1
            self.loc.data[id_unstable_] = self.lim_sup.data[id_unstable_] \
                                          + self.beta.data[id_unstable_] * self.scale.data[id_unstable_]


        self.cdf_alpha = clamp_probs(_normal_cdf(self.alpha))
        self.cdf_beta = clamp_probs(_normal_cdf(self.beta))
        self.log_Z = clamp_probs(self.cdf_beta-self.cdf_alpha).log()

    def log_prob(self, value):

        loc = self.loc.expand_as(value)
        scale = self.scale.expand_as(value)
        zeta = (value - loc) / scale

        return _normal_log_pdf(zeta) - scale.log() - self.log_Z

    def cdf(self, value):
        loc = self.loc.expand_as(value)
        scale = self.scale.expand_as(value)
        # zeta = (value - loc) / scale
        cdf = (_normal_cdf(value, loc, scale) - self.cdf_alpha)/self.log_Z.exp()
        if torch.isnan(cdf).any():
            cdf.data[torch.isnan(cdf.data)] = 0.0
        return cdf
        # return _normal_cdf(zeta)

    def log_cdf(self, value):
        loc = self.loc.expand_as(value)
        scale = self.scale.expand_as(value)
        # zeta = (value - loc) / scale
        log_cdf = (_normal_cdf(value, loc, scale) - self.cdf_alpha).log() - self.log_Z
        if torch.isnan(log_cdf.data).any():
            log_cdf.data[torch.isnan(log_cdf.data)] = -np.inf
        return log_cdf
        # return _normal_cdf(zeta).log()

    def sample(self, sample_shape=torch.Size()):
        if not len(sample_shape) == 0:
            N = np.prod(sample_shape)
            shape = list(sample_shape)+list(self.loc.shape)
        else:
            shape = list(self.loc.shape)
            N = 1
        with torch.no_grad():
            count = 0
            while count<50:
                try:
                    # r = _standard_truncnorm_sample(self.alpha.detach(), self.beta.detach(), shape)
                    r = truncnorm.rvs(self.alpha.detach().cpu(), self.beta.detach().cpu(), size=shape)
                    r = torch.tensor(r, device=self.alpha.device, dtype=self.alpha.dtype)
                    break
                except:
                    assert N==1, 'only compatible with '
                    print('Failed to converge, retrying item by item')
                    r = []
                    for i, (a, b) in enumerate(zip(self.alpha.view(-1), self.beta.view(-1))):
                        try:
                            sample = torch.tensor(truncnorm.rvs(float(a), float(b)), dtype=a.dtype,
                                                  device=a.device)
                        except:
                            print(f'alpha: {a} and beta: {b} '
                                  f'== m: {self.loc.view(-1)[i]} and s: {self.scale.view(-1)[i]} '
                                  f'ratio: {abs(b+a)/(b-a)} '
                                  f'failed to converge')
                            eps = 1e-4
                            assert a.sign()==b.sign()
                            if a>0:
                                sample = a + eps
                            else:
                                sample = b - eps
                        r.append(sample)
                    r = torch.stack(r, -1).view(self.alpha.shape)
                    # count += 1
                    break
            if count == 50:
                print('Failed 50 times to sample from TG, returning average')
                r = self.loc + (self.log_prob(self.alpha).exp()-
                                self.log_prob(self.beta).exp())/self.log_Z.exp()*self.scale
                r = r.expand(shape)
            #     print('Attempting to sample on CPU')
            if N > 1:
                r = r.view(shape)
            r = self.loc + self.scale * r
            eps = torch.finfo(r.dtype).eps
            # r = r.clamp(self.lim_inf+eps, self.lim_sup-eps)
            r.data = torch.max(torch.min(r.data, self.lim_sup-eps), self.lim_inf+eps)
        return r

    def rsample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            r = self.sample(sample_shape)
            r_logpdf = self.log_prob(r)
        cdf = self.cdf(r)
        var_to_diff = torch.full_like(cdf, 0.0)
        id = r_logpdf > -50.0
        ratio = -cdf * (-r_logpdf).exp()
        var_to_diff.masked_scatter_(id, ratio[id])
        # var_to_diff = -log/r_logpdf.exp()
        var_to_diff.data.copy_(r)
        # return SwapGrads.apply(r, var_to_diff)
        return var_to_diff

    def reparameterize_as(self, sample, grad_mode=False):
        with torch.no_grad():
            r_logpdf = self.log_prob(sample)
        cdf = self.cdf(sample)
        var_to_diff = torch.full_like(cdf, 0.0)
        id = r_logpdf > -50.0
        var_to_diff[id] = -cdf[id]/r_logpdf[id].exp()
        var_to_diff.data.copy_(sample.data)
        return var_to_diff

class SwapGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # ctx.save_for_backward(a, b)
        return a

    @staticmethod
    def backward(ctx, grad_output):
        # a, b = ctx.saved_tensors
        return None, grad_output

if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    loc = torch.randn(300, 40, requires_grad=True)
    scale = (4*torch.rand(300, 40)).requires_grad_(True)
    lim_inf = 0
    lim_sup = 1# + np.random.rand(3,1)
    d = TruncatedGaussian(loc, scale, lim_inf, lim_sup)
    z = d.rsample()
    (z.sum()).backward()
    print(f'z: {z.mean()}')
    print(f'loc: {loc.mean()}, nan in grad: {torch.isnan(loc.grad).any()}')
    print(f'scale: {scale.mean()}, nan in grad: {torch.isnan(scale.grad).any()}')
