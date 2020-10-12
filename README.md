# PyTorch version of the RaeDm model

This repo currently (only) provides a ddm distribution class, TwoBoundariesFPT (FPT = First Passage Time density).
Here is a toy example of how to use it when training a neural net whose output is a DDM distribution:
```
import torch
from torch import nn
from torch.nn import functional as F
from readm import ddm

# create neural net
x = torch.randn(1024, 10) # input: conditions
y = torch.stack([abs(torch.randn(1024)), 
                 torch.empty(1024).bernoulli_().mul_(2.0).add(-1.0)], -1) # target: RT (real) and choices (-1 or 1)
grand_t0 = y[:,0].min() # minimum RT - lower value of t0

neural_net = nn.Sequential(nn.Linear(10, 64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64, 4))
optim = torch.optim.Adam(neural_net.parameters())
def get_ddm_dist(x):
    out = neural_net(x)
    a, v, w, t0 = out.unbind(dim=-1)
    t0 = -F.softplus(t0)+grand_t0 # all parameters are real-valued, but t0 must be greater than the minimum RT
    ddm_dist = ddm.TwoBoundariesFPT(a, v, w, t0)
    return ddm_dist
for _ in range(100):
    dist = get_ddm_dist(x)
    loss = -dist.log_prob(y).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
```

This distribution has also several other features that may come in handy: `ddm.cdf(y)` will return the cdf at the specific boundary for the given action. Note that this cdf should not sum to 1, unless it is conditioned on the boundary: `ddm.cdf(y, conditioned_on_boundary=True)`.
One can also sample efficiently from the given distribution using `ddm.sample()`, and reparameterised samples can be gathered using `ddm.rsample()`. Note that in this case the second element (i.e. the choice) will be a real value between -1 and 1, and must be rounded to one of these for usage in the ddm class.
This sampling method uses Newton's method: first, generate a target CDF value at random, then generate a boundary at random given marginal probability of hitting upper/lower boundary, finally optimise the RT using NM to match the random (conditioned) CDF generated before. This should be several order of magnitudes faster than Euler's method commonly used, and much more secure than MCMC methods.

Later devs: FIVO implementation, tricks for reparameterisation of t0.
