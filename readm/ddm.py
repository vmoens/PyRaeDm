import torch
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from torch import distributions

T0 = 0.2397217965550664
n = 8
m = 4
EPS = 1e-8
class TwoBoundariesFPT(distributions.Distribution):
    def __init__(self, a, v, w, t0):
        super().__init__()
        self.a = a
        self.v = v
        self.w = w
        self.t0 = t0

    def stats(self):
        return {
            'var_rt': ddm_varrt(self.a, self.v, self.w, self.t0),
            'var_rt_cond_lower': ddm_varrt_cond(self.a, self.v, self.w, self.t0, torch.ones_like(self.a)),
            'var_rt_cond_upper': ddm_varrt_cond(self.a, self.v, self.w, self.t0, -torch.ones_like(self.a)),
            'prop_lower': ddm_avg(self.a, self.v, self.w, 1, torch.ones_like(self.a)),
            'prop_upper': ddm_avg(self.a, self.v, self.w, 1, -torch.ones_like(self.a)),
            'avg_rt': ddm_avgrt(self.a, self.v, self.w, self.t0),
            'avg_rt_cond_lower': ddm_avgrt_cond(self.a, self.v, self.w, self.t0, torch.ones_like(self.a)),
            'avg_rt_cond_upper': ddm_avgrt_cond(self.a, self.v, self.w, self.t0, -torch.ones_like(self.a)),
        }
    def _prepare_params_rand(self):
        a, v, w, t0 = self.a, self.v, self.w, self.t0
        a = ddm_mapa(a)
        w = torch.sigmoid(w)
        return a, v, w, t0

    def _prepare_params(self, value):
        rt, c = value.chunk(2, -1)
        assert ((c==1) | (c==-1)).all()
        rt = rt.squeeze(-1)
        c = c.squeeze(-1)
        a, v, w, t0 = self.a.expand_as(rt), self.v.expand_as(rt), self.w.expand_as(rt), self.t0.expand_as(rt)
        v = -c * v
        w = -c * w
        a = ddm_mapa(a)
        w = torch.sigmoid(w)
        rt = rt - t0
        return rt, c, a, v, w, t0

    def log_prob(self, value: torch.Tensor):
        rt, c, a, v, w, t0 = self._prepare_params(value)

        lp = torch.zeros_like(rt)
        non_inf_idx = rt>0
        min_inf = -750.0 if value.dtype is torch.double else -110
        lp.masked_fill_(~non_inf_idx, min_inf)
        p0 = ddm_p0(a[non_inf_idx], v[non_inf_idx], w[non_inf_idx], rt[non_inf_idx])
        p1 = ddm_p1(a[non_inf_idx], w[non_inf_idx], rt[non_inf_idx])
        lp.masked_scatter_(non_inf_idx, p0+p1)
        return lp

    def cdf(self, value, contitioned_on_boundary=False):
        rt, c, a, v, w, t0 = self._prepare_params(value)
        T0=0.2397217965550664
        n=8
        m=4
        aa = a.pow(2)
        idx_small = rt - T0 * aa<0
        idx_large = ~idx_small
        V = torch.zeros_like(rt)
        V.masked_scatter_(idx_small, DDM_cdf_small(a[idx_small], v[idx_small], w[idx_small], rt[idx_small]))
        V.masked_scatter_(idx_large, DDM_cdf_large(a[idx_large], v[idx_large], w[idx_large], rt[idx_large], n, c[idx_large]))
        V =  V.clamp(EPS, 1.0)
        if contitioned_on_boundary:
            norm = ddm_avg(self.a, self.v, self.w, 1.0, self.c)
            V = V / norm
        return V

    def rsample(self, sample_shape=torch.Size()):
        samples_val, choices_relaxed = ddm_rand_inv_cdf(sample_shape, self.a, self.v, self.w, self.t0, reparameterised=True)
        samples = -self.cdf(samples_val)*(-self.log_prob(samples_val)).exp()
        samples = torch.stack([samples, choices_relaxed], -1)
        samples.data.index_copy_(-1, torch.LongTensor([0,]), samples_val[..., :1])
        return samples

    def sample(self, sample_shape=torch.Size()):
        a, v, w, t0 = self._prepare_params_rand()
        return ddm_rand_inv_cdf(sample_shape, a, v, w, t0)

def ddm_mapa(a):
    return F.softplus(a)

def ddm_p0(a, v, w, t):
    return -a * v * w - 2 * a.log() - t * v.pow(2)/2

def logmax(x):
    return x.clamp_min(EPS).log()

def ddm_logpdf_full_common2(a, t, w, aa, n):
    # Navarro and Fuss 2009
    cst = t*4.934802200544679/aa # Numerical underflow
    V = torch.zeros_like(aa)
    for k in range(1, n+1):
            V += torch.sin(3.14159265358979*w*k)*k*torch.exp(-cst*(k*k-1))
    return logmax(V) - cst + 1.1447298858494

def ddm_logpdf_full_common1(a, t, w, aa, m):
    Ot    = t.reciprocal()
    Ot_aa = Ot*aa
    cst   = 0.5*Ot_aa*w.pow(2)

    V=torch.zeros_like(t)
    for k2 in range(-m, m+1):
        TKW = 2 * k2+w
        V += TKW*torch.exp(-0.5*Ot_aa*TKW*TKW + cst)

    return logmax(V) + 3 * a.log() - 1.5* t.log() - 9.189385332046727e-1 - cst

def ddm_p1(a:torch.Tensor, w:torch.Tensor, t:torch.Tensor):
    aa = a.pow(2)
    idx_neg = t-T0*aa < 0
    V = torch.zeros_like(t)
    V.masked_scatter_(idx_neg, ddm_logpdf_full_common1(a[idx_neg], t[idx_neg], w[idx_neg], aa[idx_neg], m))
    V.masked_scatter_(~idx_neg, ddm_logpdf_full_common2(a[~idx_neg], t[~idx_neg], w[~idx_neg], aa[~idx_neg], n))
    return V

def Ks(t: torch.Tensor, v: torch.Tensor,a: torch.Tensor,w: torch.Tensor,eps):
    x = (abs(v)*t - a*w)/2/a
    K1 = x.clone()
    idx = (abs(x)<1000) & torch.isfinite(x)
    K1.data.masked_scatter_(idx, x.data[idx])
    K1.data.masked_scatter_(~idx, x.data[~idx].sign()*1000)

    V = (v*a*w + v*v*t/2 + eps.log()).exp()/2
    arg = V.clamp(0.0, 1.0)
    V = distributions.Normal(0, 1).icdf(arg)
    V = (-t.sqrt()/2/a) * V

    K2 = V.clone()
    mask = torch.isfinite(V) & (abs(V)<1000)
    K2.data.masked_scatter_(mask, V.data[mask])
    K2.data.masked_scatter_(~mask, V.data[~mask].sign()*1000)

    # K2.data.copy_((1-mask) * V.data + mask * V.data.sign() * 1000)
    return torch.stack([K1, K1 + K2], -1).max(-1)[0].ceil().to(torch.int)

def normlogpdf(x):
    return -x.pow(2)/2 - np.log(2*np.pi)/2

def logMill(x):
    # Gondan et al. 2014
    idx = x >= 10000
    out = torch.zeros_like(x)
    out.masked_scatter_(idx, -x[idx].log())
    out.masked_scatter_(~idx, distributions.Normal(0, 1).cdf(-x[~idx]).log() - normlogpdf(x[~idx]))
    return out

def DDM_cdf_small_step(k, a, w, v, t, sqt, F):
    rj = 2 * k * a + a * w
    dj = -v * a * w - v * v * t / 2 + normlogpdf(rj / sqt)
    pos1 = dj + logMill((rj - v * t) / sqt)
    pos2 = dj + logMill((rj + v * t) / sqt)
    rj = (2 * k + 1) * a + a * (1 - w)
    dj = -v * a * w - v * v * t / 2 + normlogpdf(rj / sqt)
    neg1 = dj + logMill((rj - v * t) / sqt)
    neg2 = dj + logMill((rj + v * t) / sqt)
    pos = torch.stack([pos1, pos2], -1).logsumexp(-1).exp()
    neg = torch.stack([neg1, neg2], -1).logsumexp(-1).exp()
    F += pos - neg
    return F

def DDM_cdf_small(a: torch.Tensor, v: torch.Tensor, w: torch.Tensor, t: torch.Tensor,
                  eps=np.sqrt(EPS)):
    if a.numel()==0:
        return torch.zeros_like(a)
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps, device=a.device)
    K = Ks(t, v, a, w, eps)
    F = torch.zeros_like(a)
    sqt = t.sqrt()
    for k in range(K.max(), -1, -1):
        idx = K >= k
        # print(idx.to(torch.float).mean())
        sub_result = DDM_cdf_small_step(k, a[idx], w[idx], v[idx], t[idx], sqt[idx], F[idx])
        F.masked_scatter_(idx, sub_result)
    return F

def DDM_cdf_large(a: torch.Tensor, v: torch.Tensor, w: torch.Tensor, t: torch.Tensor, n: torch.Tensor, c: torch.Tensor):
    P = -(-2*v*a*(1-w)).expm1() / ((2*v*a*w).exp()-(-2*v*a*(1-w)).exp())
    idx = ~torch.isfinite(P)
    idx_one = idx & ((v/c).sign() == (-c).sign())
    idx_zero = idx & ~idx_one
    P.data.masked_fill_(idx_one, 1.0)
    P.data.masked_fill_(idx_zero, 0.0)
    V = torch.zeros_like(a)
    for k in range(1, n+1):
        V += k * (np.pi*k*w).sin() / (v.pow(2) + (k*np.pi/a).pow(2)) * (-0.5 * (k*np.pi/a).pow(2) * t).exp()
    V *= (- v*a*w - 0.5 * v.pow(2) * t - 2 * a.log() + 1.8378770664093453).exp()
    V.data.masked_fill_(~torch.isfinite(V), 0.0)
    return P - V

h=1e-6
delta=0.001

@torch.no_grad()
def DDM_rand(shape, a, v, w, t0):
    A = torch.zeros(*shape, *a.shape, 2, device=a.device)

    a=ddm_mapa(a) / delta
    pos = a*torch.sigmoid(w)
    pos = pos.expand_as(A[...,0]).clone()
    a = a.expand_as(pos)
    tt=torch.ones_like(a, dtype=torch.int8)
    v = v.expand_as(pos)

    pdown = 0.5*(1.0-v*delta)
    pdown = pdown.expand_as(pos)

    idx = (0 <= pos) & (pos <= a)
    while idx.any():
        tt[idx]+=1
        new_pos = torch.empty_like(pos[idx], dtype=torch.int8).bernoulli_(1-pdown[idx]).mul_(2).add_(-1)
        pos[idx] += new_pos
        idx = (0 <= pos) & (pos <= a)
        # print(tt.view(-1)[0], idx.to(torch.float).mean(), pos.std())
        # print(pos.max(), a.max())
        # print(idx.to(torch.float).mean())

    A[..., 0]=tt*h+t0
    A[..., 1].masked_fill_(pos>=a, 1.0)
    A[..., 1].masked_fill_(pos<=a, -1.0)
    return A






def ddm_avgrt(a,v,w,t0,s=1):
    # Grasman 2009
    a=ddm_mapa(a)
    w = torch.sigmoid(w)

    z = w*a
    Z = (-2*v*z/s**2).expm1()
    A = (-2*a*v/s**2).expm1()
    return t0 + -z/v + a/v * Z/A

def _psi(x, y, v):
    return (2 * v * y).exp() - (2* v * x).exp()

def ddm_avgrt_cond(a,v,w,t0,c):
    # Grasman 2009

    a = ddm_mapa(a)
    v = -v*c
    w = torch.sigmoid(-c*w)
    z = w*a

    return t0 + (z * (_psi(z-a, a, v) + _psi(0, z, v)) + 2 * a * _psi(z, 0, v)) / v / _psi(z, a, v) / _psi(-a, 0, v)

def ddm_varrt(a,v,w,t0, s=1):
    # Grasman 2009
    a = ddm_mapa(a)
    w = torch.sigmoid(w)

    z = w*a
    Z = (-2*v*z/s**2).expm1()
    A = (-2*a*v/s**2).expm1()
    return (-v*a**2 * (Z+4)*Z/A**2 + ((-3*v*a**2 + 4*v*z*a + s**2 * a)*Z + 4*v*z*a)/A - s**2 * z)/v**3

def ddm_varrt_cond(a,v,w,t0,c):

    a = ddm_mapa(a)
    v = c*v
    w = torch.sigmoid(c*w)

    av = a*v
    avw = av*w
    Q3=(-2*av).expm1()
    Q4=(-2*avw).expm1()

    idx = abs(v)>1e-3
    V0 = torch.zeros_like(v)
    v1 = -a/(v**3.0)*(w-(Q4*(a*(w*4.0-3.0)*v+1.0)+avw*4.0)/Q3+Q4*av*(Q4+4.0)/Q3**2.0)
    v2 = 0.33333333333333*a**4 * w* (1.0+w*((-3.0)+(-2.0)*((-2.0)+w)*w))
    V0.masked_scatter_(idx, v1[idx])
    V0.masked_scatter_(~idx, v2[~idx])

    return  V0

def ddm_avg(a,v,w,s,c):
    # limit of cdf for t->inf at lower barrier
    w = -w*c
    v = -v*c

    w=torch.sigmoid(w)
    omw = torch.sigmoid(-w)
    a=ddm_mapa(a)

    v = v/s**2

    V = ((-2*v*a*w).exp()-(-2*v*a).exp())/(-(-2*v*a).expm1())
    idx = ~torch.isfinite(V)
    vinf = ((2*v*a*w).exp()-(-2*v*a*omw).expm1()).reciprocal() - ((2*v*a).exp()-1).reciprocal()
    V.masked_scatter_(idx, vinf[idx])
    return V.clamp(EPS, 1-EPS)


def ddm_rand_inv_cdf(shape, a, v, w, t0, alpha=0.1, decay=0.95, maxiter=1000, thr=1e-3, reparameterised = False):
    out = torch.rand(*shape, *a.shape, 2, device=a.device, dtype=a.dtype)
    choices_prob = ddm_avg(a, v, w, 1, torch.ones_like(a))
    if reparameterised:
        rb = distributions.RelaxedBernoulli(0.1, choices_prob.expand(*shape, *a.shape))
        choices_rs = 2 * rb.rsample() - 1
        choices = (2 * (choices_rs>0) - 1).to(choices_rs.dtype)
    else:
        choices_rs = choices = (2*(out[..., -1] < choices_prob)-1).to(a.dtype)


    target = out[..., 0]
    x0 = ddm_avgrt_cond(a, v, w, t0, choices)
    t = TwoBoundariesFPT(a, v, w, t0)
    x = torch.stack([x0, choices], -1)
    cdf = t.cdf(x)
    norm = ddm_avg(a, v, w, 1, choices)
    fx = (cdf - target * norm)
    inv_cdf_grad = (-t.log_prob(x)).exp()

    diff = abs(cdf / norm - out[..., 0])
    i=0
    alpha = torch.full_like(x0, alpha)
    assert decay<1.0
    pbar = tqdm(total=np.prod(out.shape[:-1]))

    dones = torch.zeros_like(out[...,-1], dtype=torch.bool)
    # dones_current_sum = 0
    while (diff>thr).any() and (i<=maxiter):
        step = fx*inv_cdf_grad
        new_x0 = x0 - alpha*step

        lam = 0.5
        idx = (new_x0<=t0)
        while idx.any():
            new_x0[idx] = x0[idx] - alpha*lam*step[idx]
            idx = new_x0<=t0
            lam /= 2

        x = torch.stack([new_x0, choices], -1)
        cdf = t.cdf(x)
        fx = (cdf - target*norm)
        inv_cdf_grad = (-t.log_prob(x)).exp()

        prev_diff = diff
        diff = abs(fx)
        dones_current_sum = (diff<thr).sum() - dones.sum()
        dones[diff<thr] = True
        idx_keep = diff<prev_diff
        x0[idx_keep] = new_x0[idx_keep]
        # x0 = torch.stack([new_x0, t0.expand_as(new_x0)+EPS], -1).max(-1)[0]
        alpha[~idx_keep] *= decay
        diff[~idx_keep] = prev_diff[~idx_keep]

        i += 1
        pbar.update(dones_current_sum)
        # pbar.set_description(f'cdf: {(cdf/norm).item(): 4.2f}, target: {out[..., 0].item(): 4.2f}, '
        #                      f'step: {(cdf*inv_cdf_grad).item(): 4.2f}, alpha: {alpha.item(): 4.2f}, '
        #                      f'diff: {diff.item()}')

    pbar.close()
    out[..., 0] = x0
    out[..., -1] = choices
    if not reparameterised:
        return out
    return out, choices_rs
