"""
Neural-potential bootstrap ensemble on the coverage mocks: the method of Sec. 4.3 with identical
settings (1-16-16-1 tanh MLP, 48 RK4 steps, Adam + cosine schedule, 1000 iterations, weight prior
1e-3, 24 members per sky: member 0 on the mock itself, 23 on bootstrap replicas of it).
All N_MOCK x 24 networks are trained in one batch.

  python src/coverage_nn.py [--mocks 50] [--members 24] [--iters 1000]
Output: results/coverage_nn.npz  (quantities per mock and member, PIT values)
"""
import argparse, math, os, time
import numpy as np
import torch
from data import load_all, ROOT
from tcosmo import NNPotential, Likelihood, solve
from coverage_common import load_mocks, truth_values, RDS, QUANT, N_MOCK, SUFFIX

ap = argparse.ArgumentParser()
ap.add_argument("--mocks", type=int, default=N_MOCK)
ap.add_argument("--members", type=int, default=24)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--nstep", type=int, default=48)
ap.add_argument("--wd", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
nm, ne = a.mocks, a.members
B = nm * ne
data = load_all()
L = Likelihood(data, nstep=a.nstep)
mk = load_mocks()
gen = torch.Generator().manual_seed(a.seed + 7)
obs = []
for name, cov in (("cc", L.cc_cov), ("bao", L.bao_cov), ("sn", L.sn_cov)):
    base = torch.as_tensor(mk[name][:nm])                                   # (nm, n)
    c = torch.linalg.cholesky(cov)
    noise = (torch.randn(nm, ne, base.shape[1], generator=gen) @ c.T)
    noise[:, 0] = 0                                                          # member 0: the mock itself
    obs.append((base[:, None, :] + noise).reshape(B, -1))
rdn = RDS * torch.randn(nm, ne, generator=gen); rdn[:, 0] = 0
rd_mean = (torch.as_tensor(mk["rd_obs"][:nm])[:, None] + rdn).reshape(B)

RD0 = 147.09
pot = NNPotential(B, seed=a.seed)
ln_om = torch.nn.Parameter(torch.full((B,), math.log(0.142)))
ln_rd = torch.nn.Parameter(torch.zeros(B))
params = list(pot.parameters()) + [ln_om, ln_rd]
opt = torch.optim.Adam(params, lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=2e-4)


def forward():
    om = torch.exp(ln_om); rd = RD0 * torch.exp(ln_rd)
    sol = solve(pot, om, torch.zeros(B), nstep=a.nstep)
    c = L.chi2(sol, rd, obs)
    return sol, om, rd, c[0] + c[1] + c[2] + ((rd - rd_mean) / RDS) ** 2


t0 = time.time()
for it in range(a.iters + 1):
    opt.zero_grad()
    sol, om, rd, chi = forward()
    if it < a.iters:
        # gradient-norm clipping per sky (norm over its 24 members), exactly as neural_potential.py
        # does for a single sky; skies are independent problems
        (chi + a.wd * pot.l2()).sum().backward()
        with torch.no_grad():
            g2 = sum((p.grad.reshape(B, -1) ** 2).sum(1) for p in params).reshape(nm, ne).sum(1)
            scale = torch.clamp(50.0 / (g2.sqrt() + 1e-12), max=1.0).repeat_interleave(ne)
            for p in params:
                p.grad.mul_(scale.reshape([B] + [1] * (p.grad.dim() - 1)))
        opt.step(); sched.step()
    if it % 50 == 0 or it == a.iters:
        c0 = chi.detach().reshape(nm, ne)[:, 0]
        print("it %4d  chi2[member0] median=%.2f max=%.2f  (%.0fs)" % (it, c0.median(), c0.max(), time.time() - t0),
              flush=True)

with torch.no_grad():
    sol, om, rd, chi = forward()
    h = sol["H"][:, -1]
    kin = sol["y"][:, -1] ** 2 / 6; u = sol["u"][:, -1]
    q = dict(H0=100 * h, Om=om / h ** 2, w0=(kin - u) / (kin + u), dphi=sol["phi"][:, -1].abs(),
             vratio=u / sol["u"][:, 0])
    q = {k: v.reshape(nm, ne).numpy() for k, v in q.items()}
tv = truth_values()
pit = np.array([[np.mean(q[k][j, 1:] < tv[k]) for k in QUANT] for j in range(nm)])
pct = np.array([[np.percentile(q[k][j, 1:], [2.275, 15.865, 50, 84.135, 97.725]) for k in QUANT] for j in range(nm)])
np.savez(os.path.join(ROOT, "results", f"coverage_nn{SUFFIX}.npz"), pit=pit, pct=pct, quant=QUANT,
         chi2=chi.reshape(nm, ne).numpy(), runtime=time.time() - t0, members=ne, iters=a.iters,
         **{"q_" + k: v for k, v in q.items()})
print("done in %.0fs" % (time.time() - t0))
