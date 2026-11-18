"""
Exact-gradient diagnostics that the differentiable solver makes cheap.

  F1  d(chi^2)/d(params): reverse-mode autodiff vs central finite differences (accuracy, cost).
  F2  Fisher-matrix errors (forward-mode Jacobian of the data vector) vs emcee posterior widths.
  F3  Identifiability of V(phi): Fisher matrix for Legendre perturbations of ln V on the field
      range traversed by a fiducial thawing solution, marginalised over (omega_m, r_d [, eps]).
      Eigenvalues tell how many independent features of V(phi) the data actually measure.
  F4  Degeneracy of the quadratic-gravity parameter eps with the shape of V(phi).

  python src/fisher.py            (needs results/*_planck.json from fit_mcmc.py)
"""
import json, math, os, time
import numpy as np
import torch
from torch.func import jvp
from data import load_all, ROOT
from tcosmo import ExpPotential, LegendrePerturbed, Likelihood, solve

RD0, RDS = 147.09, 0.26
NSTEP = 48
data = load_all()
L = Likelihood(data, nstep=NSTEP)
A_sn = L.sn_icov - torch.outer(L.sn_icov1, L.sn_icov1) / L.sn_s11          # M marginalised
res = lambda name: json.load(open(os.path.join(ROOT, "results", name + ".json")))
out = {}


def vec(sol, rd):
    Hcc, bao, mu = L.model(sol, rd)
    return torch.cat([Hcc[0], bao[0], mu[0]])


def fisher_from_jac(J, rd_index=None):
    n1, n2 = L.n[0], L.n[0] + L.n[1]
    F = (J[:n1].T @ L.cc_icov @ J[:n1] + J[n1:n2].T @ L.bao_icov @ J[n1:n2] + J[n2:].T @ A_sn @ J[n2:])
    if rd_index is not None:
        F[rd_index, rd_index] += 1.0 / RDS ** 2
    return F


def jacobian(f, p):
    cols = []
    for i in range(len(p)):
        v = torch.zeros_like(p); v[i] = 1.0
        cols.append(jvp(f, (p,), (v,))[1])
    return torch.stack(cols, 1)


# ------------------------------------------------------------------ F1
def chi2_exp(p):
    """p = (om, u0, lam, eps, rd)"""
    one = torch.ones(1)
    sol = solve(ExpPotential(torch.log(p[1]) * one, p[2] * one), p[0] * one, p[3] * one, nstep=NSTEP)
    c = L.chi2(sol, p[4] * one)
    return (c[0] + c[1] + c[2])[0]


p0 = torch.tensor([0.1425, 0.322, 0.6, 5e-5, 147.0], requires_grad=True)
t0 = time.perf_counter(); val = chi2_exp(p0); t_fwd = time.perf_counter() - t0
t0 = time.perf_counter(); g_ad, = torch.autograd.grad(chi2_exp(p0), p0); t_ad = time.perf_counter() - t0
steps = torch.tensor([1e-5, 1e-5, 1e-4, 1e-7, 1e-3])
t0 = time.perf_counter()
g_fd = []
with torch.no_grad():
    for i in range(5):
        e = torch.zeros(5); e[i] = steps[i]
        g_fd.append((chi2_exp(p0 + e) - chi2_exp(p0 - e)) / (2 * steps[i]))
t_fd = time.perf_counter() - t0
g_fd = torch.stack(g_fd)
out["F1"] = dict(grad_autodiff=g_ad.tolist(), grad_fd=g_fd.tolist(),
                 max_rel_diff=float(((g_ad - g_fd) / g_ad).abs().max()),
                 t_forward_s=t_fwd, t_autodiff_grad_s=t_ad, t_fd_grad_s=t_fd)
print("F1  max rel. diff autodiff vs FD gradient: %.2e | forward %.2fs, AD grad %.2fs (any #params),"
      " FD grad %.2fs (5 params)" % (out["F1"]["max_rel_diff"], t_fwd, t_ad, t_fd))

# ------------------------------------------------------------------ F2
out["F2"] = {}
for model, names in (("lcdm", ["om", "u0", "rd"]), ("exp", ["om", "u0", "lam", "rd"])):
    r = res(model + "_planck")
    best = r["best"]
    lam_fid = best.get("lam", 0.0)

    def f(p, has_lam="lam" in names):
        one = torch.ones(1)
        lam = p[2] if has_lam else torch.tensor(0.0)
        sol = solve(ExpPotential(torch.log(p[1]) * one, lam * one), p[0] * one, 0 * one, nstep=NSTEP)
        return vec(sol, p[-1] * one)

    p = torch.tensor([best[n] for n in names])
    t0 = time.perf_counter(); J = jacobian(f, p); tj = time.perf_counter() - t0
    cov = torch.linalg.inv(fisher_from_jac(J, rd_index=len(names) - 1))
    sig = torch.sqrt(torch.diag(cov))
    out["F2"][model] = {n: dict(fisher=float(s), mcmc_std=r["params"][n]["std"])
                        for n, s in zip(names, sig)}
    out["F2"][model]["t_jacobian_s"] = tj
    out["F2"][model]["mcmc_runtime_s"] = r["runtime_s"]
    print("F2 ", model, {n: "Fisher %.3g / MCMC %.3g" % (float(s), r["params"][n]["std"])
                         for n, s in zip(names, sig)}, "| Jacobian %.1fs vs MCMC %.0fs" % (tj, r["runtime_s"]))

# ------------------------------------------------------------------ F3 / F4
r = res("exp_planck")
lam_fid = max(r["best"]["lam"], 0.5)        # floor so that the field actually moves in the fiducial
one = torch.ones(1)
om_fid, u0_fid = r["best"]["om"], r["best"]["u0"]
base = ExpPotential(math.log(u0_fid) * one, lam_fid * one)
with torch.no_grad():
    fid = solve(base, om_fid * one, 0 * one, nstep=NSTEP)
phi_ref = float(fid["phi"][0, -1])
K = 6


def f_shape(p):
    """p = (ln om, ln rd/147.09, eps, c_0..c_5)"""
    pot = LegendrePerturbed(base, p[3:3 + K][None, :], phi_ref)
    sol = solve(pot, torch.exp(p[0]) * one, p[2] * one, nstep=NSTEP)
    return vec(sol, RD0 * torch.exp(p[1]) * one)


p = torch.zeros(3 + K); p[0] = math.log(om_fid)
J = jacobian(f_shape, p)
F = fisher_from_jac(J)
F[1, 1] += (RD0 / RDS) ** 2


def marg(F, keep, nuis):
    Fkk = F[np.ix_(keep, keep)]
    if not nuis:
        return Fkk
    Fkn, Fnn = F[np.ix_(keep, nuis)], F[np.ix_(nuis, nuis)]
    return Fkk - Fkn @ torch.linalg.solve(Fnn, Fkn.T)


shape = list(range(3, 3 + K))
out["F3"] = dict(lam_fid=lam_fid, phi_ref=phi_ref, w0_fid=float(
    (fid["y"][0, -1] ** 2 / 6 - fid["u"][0, -1]) / (fid["y"][0, -1] ** 2 / 6 + fid["u"][0, -1])))
for label, nuis in (("GR (eps=0)", [0, 1]), ("eps free", [0, 1, 2])):
    Fs = marg(F, shape, nuis)
    ev, evec = torch.linalg.eigh(Fs)
    order = torch.argsort(ev, descending=True)
    ev, evec = ev[order], evec[:, order]
    sig = 1 / torch.sqrt(ev.clamp(min=1e-300))
    out["F3"][label] = dict(sigma_modes=sig.tolist(), eigvecs_legendre=evec.T.tolist(),
                            n_modes_sigma_lt_0p1=int((sig < 0.1).sum()), n_modes_sigma_lt_1=int((sig < 1).sum()))
    print("F3 ", label, "| sigma of ln V eigenmodes:", ["%.3g" % s for s in sig],
          "| modes with sigma<0.1:", int((sig < 0.1).sum()), " sigma<1:", int((sig < 1).sum()))
    print("     leading eigenvectors (Legendre P0..P5):")
    for i in range(3):
        print("      ", ["%+.2f" % x for x in evec[:, i]])

h_fid = float(fid["H"][0, -1])
out["F4"] = {}
for label, free_shape in (("V fixed up to amplitude (c0)", [3]), ("amplitude+slope (c0,c1)", [3, 4]),
                          ("c0..c2", [3, 4, 5]), ("c0..c5", shape)):
    idx = [0, 1, 2] + free_shape
    cov = torch.linalg.inv(F[np.ix_(idx, idx)])
    s_eps = float(torch.sqrt(cov[2, 2]))
    corr = (cov[2] / torch.sqrt(cov[2, 2] * torch.diag(cov))).tolist()
    out["F4"][label] = dict(sigma_eps_tilde=s_eps, sigma_eps_H0=s_eps * h_fid ** 2, corr_with=dict(
        zip(["ln_om", "ln_rd", "eps"] + ["c%d" % (i - 3) for i in free_shape], corr)))
    print("F4  sigma(eps~) = %.3g  [%s]   corr(eps, .) = %s" % (
        s_eps, label, {k: "%.2f" % v for k, v in out["F4"][label]["corr_with"].items() if k != "eps"}))

# ------------------------------------------------------------------ F5
# Same information in physical units.  With ln V expanded to order K-1 in Legendre polynomials on
# [0, phi_ref]:  d lnV/dphi = 2 c1/phi_ref  (+...),  d^2 lnV/dphi^2 = 12 c2/phi_ref^2 (+...).
# Marginal 1-sigma errors of the slope and curvature coefficients, GR (eps = 0), nuisance marginalised.
out["F5"] = dict(delta_lnV_fiducial=lam_fid * abs(phi_ref))
print("F5  fiducial: lambda = %.3f, field excursion = %.3f M_pl, total change of ln V = %.3f" % (
    lam_fid, abs(phi_ref), lam_fid * abs(phi_ref)))
for K_use in (2, 3, 4):
    idx = [0, 1] + list(range(3, 3 + K_use))
    cov = torch.linalg.inv(F[np.ix_(idx, idx)])
    s = torch.sqrt(torch.diag(cov))[2:]
    row = dict(sigma_lnV_amplitude=float(s[0]), sigma_slope=float(2 * s[1] / abs(phi_ref)))
    if K_use >= 3:
        row["sigma_curvature"] = float(12 * s[2] / phi_ref ** 2)
    out["F5"]["K=%d" % K_use] = row
    print("F5  ln V to order %d: " % (K_use - 1) + ", ".join("%s = %.3g" % kv for kv in row.items()))

out["F_matrix"] = F.tolist()
with open(os.path.join(ROOT, "results", "fisher.json"), "w") as fh:
    json.dump(out, fh, indent=1)
