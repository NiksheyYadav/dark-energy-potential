"""Why does the slope + curvature posterior pile w0 at -1 for the barely rolling truth? (REPORT 4.9)

Reruns the coverage chain of src/coverage_mcmc.py (12 000 steps, same priors and start) on a few
truth-B mocks whose w0 interval fails, keeps the samples, and compares the pile (w0 < -0.995)
with the rest: sign of c2, lambda, field excursion, and the slope of ln V today relative to its
initial value, 1 - c2*phi0/lambda (0 at the minimum phi = lambda/c2 that exists for c2 > 0;
negative past it).
  python src/w0_pileup_check.py 0 1 2    ->  results/w0_pileup_B.npz and a printed summary
"""
import os, sys
os.environ["COV_TRUTH"] = "B"
import numpy as np
from coverage_mcmc import run_one
from coverage_common import truth_values
from data import ROOT


def rank(a):
    return np.argsort(np.argsort(a))


if __name__ == "__main__":
    from multiprocessing import Pool
    mocks = [int(a) for a in sys.argv[1:]] or [0]
    with Pool(len(mocks)) as pool:
        res = pool.map(run_one, [(j, "poly2", int(os.environ.get("STEPS", 12000)), True) for j in mocks])
    tw, save = truth_values()["w0"], {}
    for r in res:
        j, x, q = r[0], r[7], r[8]
        lam, c2, w0, dphi = x[:, 2], x[:, 4], q["w0"], q["dphi"]
        s = 1 - c2 * dphi / np.maximum(lam, 1e-3)
        print(f"mock {j}: {len(w0)} samples, tau {r[3]:.0f} steps, {r[5]:.0f} s; w0 16/50/84 "
              f"{np.round(np.percentile(w0, [15.865, 50, 84.135]), 4)}; truth {tw:.4f} at quantile {np.mean(w0 < tw):.2f}")
        for name, m in (("pile w0<-0.995", w0 < -0.995), ("rest", w0 >= -0.995)):
            if m.any():
                print(f"  {name:15s} {m.mean():4.0%}: c2>0 {np.mean(c2[m] > 0):4.0%}, median c2 {np.median(c2[m]):+6.1f}, "
                      f"lam {np.median(lam[m]):.2f}, |dphi| {np.median(dphi[m]):.3f}, "
                      f"slope today/initial {np.median(s[m]):+.2f} (<0.5 in {np.mean(s[m] < 0.5):.0%})")
        print(f"  rank correlation with w0: c2 {np.corrcoef(rank(c2), rank(w0))[0, 1]:+.2f}, "
              f"lam {np.corrcoef(rank(lam), rank(w0))[0, 1]:+.2f}", flush=True)
        save.update({f"x_{j}": x, f"w0_{j}": w0, f"dphi_{j}": dphi, f"vratio_{j}": q["vratio"]})
    np.savez(os.path.join(ROOT, "results", "w0_pileup_B.npz"), **save)
