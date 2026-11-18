"""
Additional checks quoted in REPORT.md (each prints its result; all are saved to results/extra_checks.json).

  X1  torch solver (48 steps) vs numpy solver (512 steps): chi^2 at one point.
  X2  forward-mode Jacobian vs central finite differences.
  X3  two-sided profile likelihood in eps (Lambda fixed, and with lambda free).
  X4  SN-only flat LCDM: Omega_m, to compare with Pantheon+ (arXiv:2202.04077: 0.334 +- 0.018).
  X5  closure test cross-check: fit the exponential model to the SAME mock sky the network saw.
  X6  order-reduced branch vs the FULL third-order Friedmann equation (matter + Lambda + eps).

  python src/extra_checks.py
"""
import json, math, os
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar
from torch.func import jvp
import cosmo
from cosmo import chi2_parts, OMR
from data import load_all, ROOT
from tcosmo import ExpPotential, LegendrePerturbed, Likelihood, solve

data = load_all()
one = torch.ones(1)
out = {}
RD0, RDS = 147.09, 0.26

# ------------------------------------------------------------------ X1
p = (0.1425, 0.322, 0.6, 5e-5, 147.0)
L = Likelihood(data, nstep=48)
sol = solve(ExpPotential(math.log(p[1]) * one, p[2] * one), p[0] * one, p[3] * one, nstep=48)
ct = [float(x) for x in L.chi2(sol, p[4] * one)]
cn = [float(x[0]) for x in chi2_parts(np.array([p]), data, nstep=512)[:3]]
out["X1"] = dict(torch48=ct, numpy512=cn)
print("X1  torch(48):", np.round(ct, 5), " numpy(512):", np.round(cn, 5))

# ------------------------------------------------------------------ X2
base = ExpPotential(math.log(0.322) * one, 0.6 * one)
phi_ref = float(sol["phi"][0, -1])


def f(q):
    pot = LegendrePerturbed(base, q[3:9][None, :], phi_ref)
    s = solve(pot, torch.exp(q[0]) * one, q[2] * one, nstep=48)
    Hcc, bao, mu = L.model(s, RD0 * torch.exp(q[1]) * one)
    return torch.cat([Hcc[0], bao[0], mu[0]])


q = torch.zeros(9); q[0] = math.log(0.1425)
worst = 0.0
for i in (0, 2, 4, 5):
    v = torch.zeros(9); v[i] = 1.0
    tang = jvp(f, (q,), (v,))[1]
    e = 1e-5 if i != 2 else 1e-7
    fd = (f(q + e * v) - f(q - e * v)) / (2 * e)
    worst = max(worst, float(((fd - tang).abs() / (tang.abs().max())).max()))
out["X2"] = dict(max_rel_diff=worst)
print("X2  forward-mode Jacobian vs FD, max difference relative to column max: %.2e" % worst)

# ------------------------------------------------------------------ X3
def prof(eps, lam_free):
    def g(x):
        lam = abs(x[3]) if lam_free else 0.0
        c = chi2_parts(np.array([[x[0], x[1], lam, eps, x[2]]]), data, nstep=128)
        v = c[0][0] + c[1][0] + c[2][0] + ((x[2] - RD0) / RDS) ** 2
        return v if np.isfinite(v) else 1e30
    x0 = [0.1435, 0.3278, 147.1] + ([0.8] if lam_free else [])
    r = minimize(g, x0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-5, maxiter=4000, adaptive=True))
    return float(r.fun), [float(v) for v in r.x]


out["X3"] = {}
for lam_free in (False, True):
    rows = []
    for eps in (-6e-4, -4e-4, -2e-4, 0.0, 2e-4, 4e-4, 6e-4, 8e-4, 1.2e-3):
        c, x = prof(eps, lam_free)
        rows.append((eps, c))
    out["X3"]["lam_free" if lam_free else "Lambda"] = rows
    print("X3 ", "lambda free" if lam_free else "Lambda     ", " ".join("%+.0e:%.2f" % r for r in rows))

# ------------------------------------------------------------------ X4
def chi2_sn(Om):
    th = np.array([[Om * 0.49, (1 - Om) * 0.49 - OMR, 0.0, 0.0, 147.0]])
    return chi2_parts(th, data, nstep=256)[2][0]


r = minimize_scalar(chi2_sn, bounds=(0.2, 0.5), method="bounded", options=dict(xatol=1e-7))
e = 2e-3
sig = 1 / math.sqrt((chi2_sn(r.x + e) - 2 * r.fun + chi2_sn(r.x - e)) / (2 * e * e))
out["X4"] = dict(Om=float(r.x), sigma=sig, chi2=float(r.fun), n=len(data["sn"]["z"]))
print("X4  SN-only flat LCDM: Omega_m = %.4f +- %.4f, chi2 = %.2f for %d SNe  (Pantheon+ paper: 0.334 +- 0.018)"
      % (r.x, sig, r.fun, len(data["sn"]["z"])))

# ------------------------------------------------------------------ X5
# regenerate exactly the mock sky of  neural_potential.py --mock 1.0 --seed 0
gen = torch.Generator().manual_seed(1)
chol = [torch.linalg.cholesky(c) for c in (L.cc_cov, L.bao_cov, L.sn_cov)]
tsol = solve(ExpPotential(math.log(0.325) * one, 1.0 * one), 0.1415 * one, 0.0 * one, nstep=48)
Hcc, bao, mu = L.model(tsol, RD0 * one)
basev = [Hcc[0], bao[0], mu[0] - 19.25]
basev = [b + c @ torch.randn(len(b), generator=gen) for b, c in zip(basev, chol)]
mock = {k: dict(v) for k, v in data.items()}
mock["cc"]["H"], mock["bao"]["val"], mock["sn"]["mb"] = (b.numpy() for b in basev)


def chi2_mock(x):
    c = chi2_parts(np.array([[x[0], x[1], abs(x[2]), 0.0, x[3]]]), mock, nstep=128)
    v = c[0][0] + c[1][0] + c[2][0] + ((x[3] - RD0) / RDS) ** 2
    return v if np.isfinite(v) else 1e30


truth = [0.1415, 0.325, 1.0, RD0]
r = minimize(chi2_mock, truth, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-6, maxiter=6000, adaptive=True))
sb = cosmo.solve(np.array([[r.x[0], r.x[1], abs(r.x[2]), 0.0]]))
st = cosmo.solve(np.array([[*truth[:3], 0.0]]))
nn = json.load(open(os.path.join(ROOT, "results", "nn_planck_mock.json")))["real"]
out["X5"] = dict(chi2_truth=float(chi2_mock(truth)), chi2_exp_best=float(r.fun), exp_lam=float(abs(r.x[2])),
                 exp_dphi=float(sb["phi0"][0]), exp_H0=float(100 * sb["hub"][0]), exp_w0=float(sb["w0"][0]),
                 truth_dphi=float(st["phi0"][0]), truth_H0=float(100 * st["hub"][0]), truth_w0=float(st["w0"][0]),
                 nn_chi2=nn["chi2_total"], nn_dphi=nn["dphi"], nn_H0=nn["H0"], nn_w0=nn["w0"])
print("X5  mock sky: chi2(truth) = %.2f | exp-V best: chi2 = %.2f, lam = %.3f, dphi = %.3f, H0 = %.2f, w0 = %.3f"
      % (out["X5"]["chi2_truth"], r.fun, abs(r.x[2]), sb["phi0"][0], 100 * sb["hub"][0], sb["w0"][0]))
print("    truth: dphi = %.3f, H0 = %.2f, w0 = %.3f | network: chi2 = %.2f, dphi = %.3f, H0 = %.2f, w0 = %.3f"
      % (st["phi0"][0], 100 * st["hub"][0], st["w0"][0], nn["chi2_total"], nn["dphi"], nn["H0"], nn["w0"]))

# ------------------------------------------------------------------ X6
# Full equation in N = ln a (radiation off):  H^2 + 6 eps H^2 [H'^2 + 2 H H'' + 6 H H'] = m + Lam
# -> H'' = [ (m + Lam - H^2)/(6 eps H^2) - H'^2 - 6 H H' ] / (2 H).   Start on the reduced branch at z = 4.
om, lam0 = 0.1435, 0.3278
out["X6"] = {}
for eps in (1e-4, 4e-4, 8e-4):
    Hred = lambda N: np.sqrt(om * np.exp(-3 * N) + lam0 + 13.5 * eps * om ** 2 * np.exp(-6 * N))
    dHred = lambda N: (-3 * om * np.exp(-3 * N) - 81 * eps * om ** 2 * np.exp(-6 * N)) / (2 * Hred(N))

    def rhs(N, s):
        H, Hp = s
        return [Hp, ((om * np.exp(-3 * N) + lam0 - H * H) / (6 * eps * H * H) - Hp * Hp - 6 * H * Hp) / (2 * H)]

    Ni = -math.log(5.0)
    full = solve_ivp(rhs, [Ni, 0.0], [Hred(Ni), dHred(Ni)], method="Radau", rtol=1e-10, atol=1e-12, dense_output=True)
    Ng = np.linspace(Ni, 0, 400)
    dev = full.sol(Ng)[0] / Hred(Ng) - 1
    zsel = np.expm1(-Ng) <= 2.33
    out["X6"]["%.0e" % eps] = dict(max_dev_all=float(np.abs(dev).max()), max_dev_data_range=float(np.abs(dev[zsel]).max()),
                                   dev_z0=float(dev[-1]), correction_at_z2p33=float(13.5 * eps * om * 3.33 ** 3),
                                   scalaron_mass_over_H0=float(1 / math.sqrt(6 * eps) / math.sqrt(om + lam0)))
    print("X6  eps~ = %.0e: full third-order vs order-reduced H: max |dH/H| = %.2e (z<=4), %.2e (z<=2.33), %.2e at z=0;"
          " size of the O(eps) term at z=2.33: %.3f" % (eps, np.abs(dev).max(), np.abs(dev[zsel]).max(), abs(dev[-1]),
                                                        13.5 * eps * om * 3.33 ** 3))

with open(os.path.join(ROOT, "results", "extra_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)
