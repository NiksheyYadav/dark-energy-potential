"""
Free-form reconstruction of the scalar potential V(phi) by forward modelling.

  V(phi) is a small neural network; the Klein-Gordon + (order-reduced quadratic-gravity)
  Friedmann system is integrated with a differentiable RK4; chi^2(CC + DESI DR2 + Pantheon+)
  is minimised by gradient descent through the solver.  Unlike Gaussian-process
  reconstructions, no derivative of noisy data is taken and phidot^2 >= 0 by construction.

Uncertainty: deep ensemble with parametric bootstrap ("randomised MAP"): member 0 is trained
on the real data, members 1..B-1 on data + noise drawn from the full covariances (and from the
r_d prior), each with its own random initialisation.  This is an APPROXIMATE posterior.

  python src/neural_potential.py --rd planck              # real data, GR (eps = 0)
  python src/neural_potential.py --rd planck --eps        # eps >= 0 free
  python src/neural_potential.py --rd planck --mock 1.0   # closure test: truth V = V0 exp(-phi)
"""
import argparse, json, math, os, time
import numpy as np
import torch
from data import load_all, ROOT
from tcosmo import NNPotential, ExpPotential, Likelihood, Interp, solve

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=24)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--rd", default="planck")
ap.add_argument("--eps", action="store_true")
ap.add_argument("--mock", type=float, default=None, help="truth lambda for a closure test")
ap.add_argument("--nstep", type=int, default=48)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wd", type=float, default=1e-3, help="Gaussian prior strength on the weights")
args = ap.parse_args()
torch.manual_seed(args.seed)
B, RD0, RDS = args.B, 147.09, 0.26

data = load_all()
L = Likelihood(data, nstep=args.nstep)
gen = torch.Generator().manual_seed(args.seed + 1)
chol = [torch.linalg.cholesky(c) for c in (L.cc_cov, L.bao_cov, L.sn_cov)]

# ---- data vector: real, or mock generated from a known potential -----------------------
truth = None
if args.mock is not None:
    one = torch.ones(1)
    tpot = ExpPotential(math.log(0.325) * one, args.mock * one)
    tsol = solve(tpot, 0.1415 * one, 0.0 * one, nstep=args.nstep)
    Hcc, bao, mu = L.model(tsol, RD0 * one)
    base = [Hcc[0], bao[0], mu[0] - 19.25]
    base = [b + c @ torch.randn(len(b), generator=gen) for b, c in zip(base, chol)]   # one noisy sky
    truth = dict(lam=args.mock, phi=tsol["phi"][0].numpy(), u=tsol["u"][0].numpy(),
                 H0=100 * float(tsol["H"][0, -1]))
else:
    base = [L.cc_H, L.bao_v, L.sn_mb]

# ---- bootstrap replicas (member 0 = unperturbed data) -----------------------------------
obs, rd_mean = [], torch.full((B,), RD0)
for b, c in zip(base, chol):
    noise = torch.randn(B, len(b), generator=gen) @ c.T
    noise[0] = 0
    obs.append(b[None, :] + noise)
rdn = RDS * torch.randn(B, generator=gen); rdn[0] = 0
rd_mean = rd_mean + rdn

# ---- parameters ----------------------------------------------------------------------------
pot = NNPotential(B, seed=args.seed)
ln_om = torch.nn.Parameter(torch.full((B,), math.log(0.142)))
ln_rd = torch.nn.Parameter(torch.zeros(B))                       # r_d = 147.09 * exp(ln_rd)
e_raw = torch.nn.Parameter(torch.full((B,), -3.0))               # eps = 1e-4 * softplus(e_raw)
params = list(pot.parameters()) + [ln_om, ln_rd] + ([e_raw] if args.eps else [])
opt = torch.optim.Adam(params, lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=2e-4)


def forward():
    om = torch.exp(ln_om)
    eps = 1e-4 * torch.nn.functional.softplus(e_raw) if args.eps else torch.zeros(B)
    rd = RD0 * torch.exp(ln_rd)
    sol = solve(pot, om, eps, nstep=args.nstep)
    c = L.chi2(sol, rd, obs)
    pri = ((rd - rd_mean) / RDS) ** 2 if args.rd == "planck" else torch.zeros(B)
    return sol, om, eps, rd, c, pri


t0 = time.time()
hist = []
for it in range(args.iters + 1):
    opt.zero_grad()
    sol, om, eps, rd, c, pri = forward()
    chi = c[0] + c[1] + c[2] + pri
    loss = (chi + args.wd * pot.l2()).sum()
    if it < args.iters:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 50.0)
        opt.step(); sched.step()
    if it % 100 == 0 or it == args.iters:
        hist.append((it, float(chi[0]), float(chi.median())))
        print("it %4d  chi2[real]=%.3f  median chi2=%.3f  h[real]=%.4f  (%.0fs)" %
              (it, chi[0], chi.median(), sol["H"][0, -1], time.time() - t0), flush=True)

# ---- post-processing --------------------------------------------------------------------------
with torch.no_grad():
    sol, om, eps, rd, c, pri = forward()
    zg = np.linspace(0, 2.5, 126)
    Hz, _ = Interp(zg, args.nstep)(sol)
    kin = sol["y"] ** 2 / 6
    w = (kin - sol["u"]) / (kin + sol["u"])
    h = sol["H"][:, -1]
    Ngrid = sol["Ni"] + sol["h"] * np.arange(args.nstep + 1)
    out = dict(z_nodes=np.expm1(-Ngrid), phi=sol["phi"].numpy(), u=sol["u"].numpy(), w=w.numpy(),
               kin=kin.numpy(), zg=zg, Hz=100 * Hz.numpy(), H0=100 * h.numpy(),
               Om=(om / h ** 2).numpy(), rd=rd.numpy(), eps=eps.numpy(), eps_H0=(eps * h ** 2).numpy(),
               w0=w[:, -1].numpy(), chi2_cc=c[0].numpy(), chi2_bao=c[1].numpy(), chi2_sn=c[2].numpy(),
               chi2_prior=pri.numpy(), hist=np.array(hist))
    if truth:
        out.update(truth_phi=truth["phi"], truth_u=truth["u"], truth_lam=truth["lam"], truth_H0=truth["H0"])

tag = "nn_%s%s%s" % (args.rd, "_eps" if args.eps else "", "_mock" if args.mock is not None else "")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
np.savez_compressed(os.path.join(ROOT, "results", tag + ".npz"), **out)
q = lambda v: [float(x) for x in np.percentile(v[1:], [15.865, 50, 84.135])]
summary = dict(tag=tag, B=B, iters=args.iters, runtime_s=time.time() - t0,
               real=dict(chi2_cc=float(c[0][0]), chi2_bao=float(c[1][0]), chi2_sn=float(c[2][0]),
                         chi2_prior=float(pri[0]), chi2_total=float(c[0][0] + c[1][0] + c[2][0] + pri[0]),
                         H0=float(out["H0"][0]), Om=float(out["Om"][0]), w0=float(out["w0"][0]),
                         dphi=float(abs(out["phi"][0, -1])), eps_H0=float(out["eps_H0"][0])),
               ensemble_16_50_84=dict(H0=q(out["H0"]), Om=q(out["Om"]), w0=q(out["w0"]),
                                      dphi=q(np.abs(out["phi"][:, -1])), eps_H0=q(out["eps_H0"])),
               n_params_per_member=int(sum(p.numel() for p in params) / B))
if truth:
    summary["truth"] = dict(lam=truth["lam"], H0=truth["H0"], w0=float(
        (0 - 1) if truth["lam"] == 0 else np.nan))
with open(os.path.join(ROOT, "results", tag + ".json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
