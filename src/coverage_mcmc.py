"""
Sampled posteriors (emcee) on the coverage mocks.

  python src/coverage_mcmc.py --model poly2 --procs 6     # ln V = ln V0 - lam phi + c2 phi^2/2
  python src/coverage_mcmc.py --model exp   --procs 6     # well-specified reference (c2 = 0)

Priors (flat): om [0.05,0.30], u0 [0.05,0.90], lam [0,2.5], c2 [-30,30], rd [120,180];
r_d measurement: Gaussian around the fluctuated rd_obs of each mock.
Output: results/coverage_mcmc_<model>.npz with, per mock and quantity, the posterior quantile of
the truth (PIT value), posterior median and 16/84 percentiles, and chain diagnostics.
"""
import argparse, os, time
import numpy as np


def run_one(args):
    j, model, steps = args[:3]
    keep = len(args) > 3 and args[3]   # also return the thinned samples and derived quantities
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import emcee
    import cosmo
    from data import load_all
    from coverage_common import load_mocks, truth_values, TRUTH, RDS, QUANT
    data = load_all()
    mk = load_mocks()
    d = {k: dict(v) for k, v in data.items()}
    d["cc"]["H"], d["bao"]["val"], d["sn"]["mb"] = mk["cc"][j], mk["bao"][j], mk["sn"][j]
    rd_obs = mk["rd_obs"][j]
    poly = model == "poly2"
    lo = np.array([0.05, 0.05, 0.0, 120.0] + ([-30.0] if poly else []))
    hi = np.array([0.30, 0.90, 2.5, 180.0] + ([30.0] if poly else []))

    def full(x):
        th = np.zeros((x.shape[0], 6))
        th[:, 0], th[:, 1], th[:, 2], th[:, 4] = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        if poly:
            th[:, 5] = x[:, 4]
        return th

    def logp(x):
        x = np.atleast_2d(x)
        ok = np.all((x >= lo) & (x <= hi), axis=1)
        out = np.full(x.shape[0], -np.inf)
        if ok.any():
            th = full(x[ok])
            c = cosmo.chi2_parts(th, d, nstep=64)
            v = -0.5 * (c[0] + c[1] + c[2] + ((th[:, 4] - rd_obs) / RDS) ** 2)
            out[ok] = np.where(np.isfinite(v), v, -np.inf)
        return out

    rng = np.random.default_rng(1000 + j)
    x0 = np.array([TRUTH["om"], TRUTH["u0"], TRUTH["lam"], TRUTH["rd"]] + ([0.0] if poly else []))
    sc = np.array([2e-3, 1e-2, 0.1, 0.2] + ([3.0] if poly else []))
    nw = 32
    p0 = np.clip(x0 + sc * rng.standard_normal((nw, len(x0))), lo + 1e-9, hi - 1e-9)
    t0 = time.time()
    sam = emcee.EnsembleSampler(nw, len(x0), logp, vectorize=True)
    sam.run_mcmc(p0, steps, progress=False)
    burn = steps // 4
    try:
        tau = float(np.max(sam.get_autocorr_time(discard=burn, tol=0)))
    except Exception:
        tau = np.nan
    chain = sam.get_chain(discard=burn, flat=True)
    thin = chain[:: max(1, len(chain) // 2000)]
    s = cosmo.solve(full(thin), nstep=64)
    h = s["hub"]
    q = dict(H0=100 * h, Om=thin[:, 0] / h ** 2, w0=s["w0"], dphi=np.abs(s["phi0"]), vratio=s["vratio"])
    tv = truth_values()
    pit = [float(np.mean(q[k] < tv[k])) for k in QUANT]
    pct = [np.percentile(q[k], [2.275, 15.865, 50, 84.135, 97.725]) for k in QUANT]
    out = (j, pit, pct, tau, len(chain) / tau if tau == tau else np.nan, time.time() - t0,
           float(np.mean(sam.acceptance_fraction)))
    return out + (thin, q) if keep else out


if __name__ == "__main__":
    from multiprocessing import Pool
    from coverage_common import N_MOCK, QUANT, load_mocks, SUFFIX
    from data import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="poly2")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--mocks", type=int, default=N_MOCK)
    a = ap.parse_args()
    load_mocks()
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = []
        for r in pool.imap_unordered(run_one, [(j, a.model, a.steps) for j in range(a.mocks)]):
            res.append(r)
            print("mock %2d done  tau=%.0f  N_eff=%.0f  %.0fs  acc=%.2f  (%d/%d, %.0fs elapsed)"
                  % (r[0], r[3], r[4], r[5], r[6], len(res), a.mocks, time.time() - t0), flush=True)
    res.sort(key=lambda r: r[0])
    np.savez(os.path.join(ROOT, "results", f"coverage_mcmc_{a.model}{SUFFIX}.npz"),
             pit=np.array([r[1] for r in res]), pct=np.array([r[2] for r in res]),
             tau=np.array([r[3] for r in res]), neff=np.array([r[4] for r in res]),
             runtime=np.array([r[5] for r in res]), acc=np.array([r[6] for r in res]), quant=QUANT,
             steps=a.steps, total_runtime=time.time() - t0)
    print("total %.0fs" % (time.time() - t0))
