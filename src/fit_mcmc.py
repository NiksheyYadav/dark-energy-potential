"""
emcee fits of the nested model family  u(phi) = u0 exp(-lam phi)  (+ eps a^-6-type correction).

  python src/fit_mcmc.py --model lcdm      --rd free
  python src/fit_mcmc.py --model exp_eps   --rd planck --steps 4000

models : lcdm (om,u0,rd) | lcdm_eps (+eps) | exp (+lam) | exp_eps (+lam,+eps)
--rd   : free   -> flat prior 120 < r_d < 180 Mpc (BAO uncalibrated; H0 set by CC only)
         planck -> Gaussian prior r_d = 147.09 +- 0.26 Mpc (Planck 2018 LCDM, arXiv:1807.06209);
                   valid here because the model is standard for z > 4 by construction.
Output : results/<model>_<rd>.npz (chain, derived) and .json (summary).
"""
import argparse, json, os, time
import numpy as np
import emcee
from scipy.optimize import minimize
from cosmo import chi2_parts, solve
from data import load_all, ROOT, CC_SET

FREE = {"lcdm": [0, 1, 4], "lcdm_eps": [0, 1, 3, 4], "exp": [0, 1, 2, 4], "exp_eps": [0, 1, 2, 3, 4]}
NAMES = ["om", "u0", "lam", "eps", "rd"]
LO = np.array([0.05, 0.05, 0.0, 0.0, 120.0])
HI = np.array([0.30, 0.90, 2.5, 2.0e-3, 180.0])
START = np.array([0.142, 0.32, 0.3, 5e-5, 147.0])
RD_PLANCK = (147.09, 0.26)
NSTEP = 128

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lcdm")
ap.add_argument("--rd", default="free")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--walkers", type=int, default=32)
args = ap.parse_args()
idx = FREE[args.model]
data = load_all()
ndata = len(data["cc"]["z"]) + len(data["bao"]["z"]) + len(data["sn"]["z"])


def full(x):
    x = np.atleast_2d(x)
    th = np.zeros((x.shape[0], 5))
    th[:, 4] = 147.0
    th[:, idx] = x
    return th


def chi2(x):
    th = full(x)
    c1, c2, c3, sol = chi2_parts(th, data, nstep=NSTEP)
    tot = c1 + c2 + c3
    pri = np.zeros_like(tot)
    if args.rd == "planck":
        pri = ((th[:, 4] - RD_PLANCK[0]) / RD_PLANCK[1]) ** 2
    return tot, pri, (c1, c2, c3), sol


def logp(x):
    x = np.atleast_2d(x)
    ok = np.all((x >= LO[idx]) & (x <= HI[idx]), axis=1)
    out = np.full(x.shape[0], -np.inf)
    if ok.any():
        tot, pri, _, _ = chi2(x[ok])
        v = -0.5 * (tot + pri)
        out[ok] = np.where(np.isfinite(v), v, -np.inf)
    return out


def best_fit(x0):
    f = lambda x: float(-2 * logp(x)[0]) if np.isfinite(logp(x)[0]) else 1e30
    r = minimize(f, x0, method="Nelder-Mead", bounds=list(zip(LO[idx], HI[idx])),
                 options=dict(xatol=1e-8, fatol=1e-6, maxiter=6000, adaptive=True))
    return r.x, r.fun


t0 = time.time()
x0, f0 = best_fit(START[idx])
rng = np.random.default_rng(1)
scale = np.array([2e-3, 5e-3, 0.05, 2e-5, 0.3])[idx]
p0 = np.clip(x0 + scale * rng.standard_normal((args.walkers, len(idx))), LO[idx] + 1e-9, HI[idx] - 1e-9)
sampler = emcee.EnsembleSampler(args.walkers, len(idx), logp, vectorize=True)
sampler.run_mcmc(p0, args.steps, progress=False)
try:
    tau = sampler.get_autocorr_time(tol=0)
except Exception:
    tau = np.full(len(idx), np.nan)
burn = int(max(args.steps // 4, 5 * np.nanmax(tau)))
chain = sampler.get_chain(discard=burn, flat=True)
lp = sampler.get_log_prob(discard=burn, flat=True)
xb, fb = best_fit(chain[np.argmax(lp)])
if f0 < fb:
    xb, fb = x0, f0
tot, pri, parts, solb = chi2(xb)

# derived parameters on a thinned chain
thin = chain[:: max(1, len(chain) // 6000)]
sol = solve(full(thin), nstep=NSTEP)
h = sol["hub"]
th = full(thin)
derived = dict(H0=100 * h, Om=th[:, 0] / h ** 2, w0=sol["w0"], eps_H0=th[:, 3] * h ** 2,
               phi0=sol["phi0"])


def summ(v):
    lo, med, hi = np.percentile(v, [15.865, 50, 84.135])
    return dict(median=float(med), minus=float(med - lo), plus=float(hi - med),
                up95=float(np.percentile(v, 95)), mean=float(np.mean(v)), std=float(np.std(v)))


k = len(idx) + 1                                    # +1 for the SN absolute magnitude M
summary = dict(model=args.model, rd=args.rd, steps=args.steps, walkers=args.walkers, burn=burn,
               tau=[float(t) for t in tau], n_eff_min=float(len(chain) / np.nanmax(tau)),
               acceptance=float(np.mean(sampler.acceptance_fraction)),
               ndata=ndata, k=k,
               best=dict(zip([NAMES[i] for i in idx], map(float, xb))),
               best_H0=float(100 * solb["hub"][0]), best_w0=float(solb["w0"][0]),
               chi2=dict(cc=float(parts[0][0]), bao=float(parts[1][0]), sn=float(parts[2][0]),
                         rd_prior=float(pri[0]), total=float(tot[0] + pri[0])),
               AIC=float(tot[0] + pri[0] + 2 * k), BIC=float(tot[0] + pri[0] + k * np.log(ndata)),
               params={NAMES[i]: summ(chain[:, j]) for j, i in enumerate(idx)},
               derived={n: summ(v) for n, v in derived.items()},
               runtime_s=time.time() - t0)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
tag = os.path.join(ROOT, "results", f"{args.model}_{args.rd}" + ("_cc32" if CC_SET == "32" else "")
                   + ("_o2" if os.environ.get("EPS_ORDER", "1") == "2" else ""))
np.savez_compressed(tag + ".npz", chain=chain, logp=lp, names=[NAMES[i] for i in idx],
                    thin=thin, **{"d_" + n: v for n, v in derived.items()})
with open(tag + ".json", "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
