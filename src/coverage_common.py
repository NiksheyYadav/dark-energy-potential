"""
Shared pieces of the coverage study (REPORT.md Sec. 4.9).

Truth: the exponential potential at the real-data best fit (Planck r_d prior),
       om = 0.1392, u0 = 0.3738, lam = 0.822, r_d = 147.09  ->  H0 = 67.76, w0 = -0.896.
Mocks: N_MOCK noisy skies drawn from the real covariances (CC, DESI DR2, Pantheon+).  The r_d
       prior is treated as an external measurement and is fluctuated too (rd_obs).
Quantities scored: H0, Omega_m, w0, field excursion |dphi|, V(phi_0)/V(phi_i).
"""
import os
import numpy as np
import cosmo
from data import load_all, ROOT

TRUTHS = {"A": dict(om=0.1392, u0=0.3738, lam=0.822, rd=147.09),    # exp best fit to real data
          "B": dict(om=0.1430, u0=0.3330, lam=0.30, rd=147.09)}     # barely rolling, near LCDM
TRUTH_ID = os.environ.get("COV_TRUTH", "A")                          # set COV_TRUTH=B for truth B
TRUTH = TRUTHS[TRUTH_ID]
SEED = int(os.environ.get("COV_SEED", 2026))                        # COV_SEED: independent noise draws
N_MOCK = int(os.environ.get("COV_NMOCK", 50))                        # COV_NMOCK: number of mock skies
SUFFIX = ("" if TRUTH_ID == "A" else "_" + TRUTH_ID) + ("" if SEED == 2026 else f"_s{SEED}")
RDS = 0.26
QUANT = ["H0", "Om", "w0", "dphi", "vratio"]
MOCKFILE = os.path.join(ROOT, "results", f"coverage_mocks{SUFFIX}.npz")


def truth_values():
    th = np.array([[TRUTH["om"], TRUTH["u0"], TRUTH["lam"], 0.0, TRUTH["rd"]]])
    s = cosmo.solve(th, nstep=512)
    h = s["hub"][0]
    return dict(H0=100 * h, Om=TRUTH["om"] / h ** 2, w0=s["w0"][0], dphi=abs(s["phi0"][0]), vratio=s["vratio"][0])


def make_mocks(seed=SEED):
    data = load_all()
    th = np.array([[TRUTH["om"], TRUTH["u0"], TRUTH["lam"], 0.0, TRUTH["rd"]]])
    sol = cosmo.solve(th, nstep=512)
    cc, bao, sn = data["cc"], data["bao"], data["sn"]
    Hcc, _ = cosmo.observables(sol, cc["z"])
    Hb, Ib = cosmo.observables(sol, bao["z"])
    DM = cosmo.C100 * Ib / TRUTH["rd"]; DH = cosmo.C100 / Hb / TRUTH["rd"]
    DV = (bao["z"] * DM * DM * DH) ** (1 / 3)
    vb = np.where(bao["isDV"], DV, np.where(bao["isDM"], DM, DH))[0]
    _, Is = cosmo.observables(sol, sn["z"])
    mu = 5 * np.log10((1 + sn["zhel"]) * Is[0]) - 19.25
    rng = np.random.default_rng(seed)
    out = {}
    for name, mean, cov in (("cc", 100 * Hcc[0], cc["cov"]), ("bao", vb, bao["cov"]), ("sn", mu, sn["cov"])):
        Lc = np.linalg.cholesky(cov)
        out[name] = mean[None, :] + rng.standard_normal((N_MOCK, len(mean))) @ Lc.T
    out["rd_obs"] = TRUTH["rd"] + RDS * rng.standard_normal(N_MOCK)
    np.savez_compressed(MOCKFILE, **out)
    return out


def load_mocks():
    if not os.path.exists(MOCKFILE):
        make_mocks()
    d = np.load(MOCKFILE)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    m = make_mocks()
    print("mocks:", {k: v.shape for k, v in m.items()})
    print("truth:", {k: round(float(v), 4) for k, v in truth_values().items()})
