from torch.distributions.utils import _sum_rightmost, lazy_property
from torch.distributions import Distribution
from torch import nn

import torch
import numpy as np

class CancelGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *params):
        with torch.set_grad_enabled(True):
            ctx.save_for_backward(x, *params)
        return x.detach().requires_grad_(x.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.set_grad_enabled(True):
            x, *params = ctx.saved_tensors
            grads = torch.autograd.grad(x, params, grad_output, True)
        return (None, *grads)
cancel_grad = CancelGrad.apply
def systematic_resampling(w, N):
    # replace nans
    w.data.masked_fill_(~torch.isfinite(w), 1e-12)
    w /= w.sum(dim=-1, keepdim=True)

    U1 = torch.rand(w.shape[0], 1, device=w.device) / N
    Ui = torch.stack([U1] + [U1 + (i - 1) / N for i in range(2, N + 1)], -1)
    cw = w.cumsum(-1)
    cw_stack = torch.cat([
        torch.zeros_like(cw[:, :1]),
        cw[:, :-1]
    ], -1).unsqueeze(-1)
    cw = cw.unsqueeze(-1)
    try:
        idx = torch.where((cw_stack < Ui) & (Ui <= cw))[1].view(w.shape[0], N)
    except:
        print('systematic resampling failed, trying multinomial')
        idx = torch.multinomial(w, N, replacement=True)
        # idx = torch.where((cw_stack <= Ui) & (Ui < cw))[1].view(w.shape[0], N)

    return idx


class StackedDistributions(Distribution):
    def __init__(self, dists, sort_index=None, univariates=None):
        batch_shape = dists[0].batch_shape
        event_shape = torch.Size([sum([dist.event_shape[i] for dist in dists])
                                  for i in range(len(dists[0].event_shape))])
        super().__init__(event_shape=event_shape, batch_shape=batch_shape)
        self.dists = dists
        if sort_index is None:
            self.sort_index = torch.arange(len(self.dists))
        else:
            self.sort_index = sort_index

        self.sort_index = [id if isinstance(id, list) else [id]
                           for id in self.sort_index]
        sort_index_np = np.concatenate(self.sort_index)
        assert len(np.unique(sort_index_np)) == len(sort_index_np), 'repeated index in sort_index'
        if univariates is None:
            univariates = [len(s)==1 for s in self.sort_index]
        self.univariates = univariates
        self.inv_idx = list(np.concatenate(self.sort_index).argsort())

    def log_prob(self, value):
        values = [value[..., id[0]] if self.univariates[i] else value[..., id]
                  for i, id in enumerate(self.sort_index)]
        event_dim = len(self.event_shape)
        lp = 0.0
        for i, (_value, dist) in enumerate(zip(values, self.dists)):
            _lp = dist.log_prob(_value)
            lp += _sum_rightmost(_lp, event_dim-len(dist.event_shape))
        while lp.ndimension()!=value.ndimension()-event_dim:
            lp = lp.unsqueeze(-1)
        return lp

    def _cast_output(self, output):
        output = [_output.unsqueeze(-1) if self.univariates[i]
                  else _output for (i, _output) in enumerate(output)]
        output = torch.cat(output, -1)
        return output

    def rsample(self, sample_shape=torch.Size()):
        samples = [dist.rsample(sample_shape) for dist in self.dists]
        samples = self._cast_output(samples)
        samples = samples.gather(-1, torch.tensor(self.inv_idx, device=samples.device).expand_as(samples))
        return samples

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def mean(self):
        values = self._cast_output([dist.mean for dist in self.dists])
        return values[..., self.inv_idx]

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.zero_()