"""
Loaders for the three public late-time data sets (files live in ../data).

  CC   : 15 cosmic-chronometer H(z) points, Moresco et al. 2012/2015/2016 (BC03),
         with the full systematic covariance built as in M. Moresco's
         CCcovariance repository (gitlab.com/mmoresco/CCcovariance):
             Cov = diag(errHz^2) + Cov_IMF + Cov_SPS(odd-one-out)
  SN   : Pantheon+ (Scolnic et al. 2022, Brout et al. 2022), zHD > 0.01, no
         Cepheid calibrators -> 1590 light curves, full STAT+SYS covariance.
         The absolute magnitude M is marginalised analytically (flat prior).
  BAO  : DESI DR2 (arXiv:2503.14738), 13 points D_V/r_d, D_M/r_d, D_H/r_d with
         the official covariance (CobayaSampler/bao_data, desi_bao_dr2).
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
C_KMS = 299792.458
CC_SET = os.environ.get("CC_SET", "15")           # CC_SET=32: 32-point chronometer compilation (robustness)


def load_cc():
    f = os.path.join(DATA, "HzTable_MM_BC03.dat")
    z, Hz, err = np.genfromtxt(f, delimiter=",", usecols=(0, 1, 2), unpack=True, comments="#")
    zmod, imf, slib, sps, ooo = np.genfromtxt(os.path.join(DATA, "data_MM20.dat"), unpack=True)
    cov = np.diag(err ** 2)
    for comp in (imf, ooo):                       # fully correlated systematic modes
        v = Hz * np.interp(z, zmod, comp) / 100.0
        cov += np.outer(v, v)
    if CC_SET == "32":                            # add the other 17 points of the 32-point compilation
        z2, H2, e2 = np.genfromtxt(os.path.join(DATA, "HzTable_CC_extra17.dat"), delimiter=",",
                                   usecols=(0, 1, 2), unpack=True, comments="#")
        n, m = len(z), len(z2)
        cov = np.block([[cov, np.zeros((n, m))], [np.zeros((m, n)), np.diag(e2 ** 2)]])
        z, Hz = np.concatenate([z, z2]), np.concatenate([Hz, H2])
        o = np.argsort(z)
        z, Hz, cov = z[o], Hz[o], cov[np.ix_(o, o)]
    return dict(z=z, H=Hz, cov=cov, icov=np.linalg.inv(cov))


def load_bao():
    z, val, kind = [], [], []
    with open(os.path.join(DATA, "desi_dr2_mean.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            a, b, c = line.split()
            z.append(float(a)); val.append(float(b)); kind.append(c)
    cov = np.loadtxt(os.path.join(DATA, "desi_dr2_cov.txt"))
    kind = np.array(kind)
    return dict(z=np.array(z), val=np.array(val), kind=kind, cov=cov, icov=np.linalg.inv(cov),
                isDV=kind == "DV_over_rs", isDM=kind == "DM_over_rs", isDH=kind == "DH_over_rs")


def load_sn(zmin=0.01):
    cache = os.path.join(DATA, "pantheon_plus_cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    f = os.path.join(DATA, "Pantheon+SH0ES.dat")
    with open(f) as fh:
        head = fh.readline().split()
        rows = [ln.split() for ln in fh if ln.strip()]
    col = {name: np.array([r[i] for r in rows]) for i, name in enumerate(head)}
    zhd = col["zHD"].astype(float)
    zhel = col["zHEL"].astype(float)
    mb = col["m_b_corr"].astype(float)
    calib = col["IS_CALIBRATOR"].astype(int)
    raw = np.loadtxt(os.path.join(DATA, "Pantheon+SH0ES_STAT+SYS.cov"))
    n = int(raw[0])
    cov = raw[1:].reshape(n, n)
    keep = (zhd > zmin) & (calib == 0)
    cov = cov[np.ix_(keep, keep)]
    icov = np.linalg.inv(cov)
    one = np.ones(keep.sum())
    out = dict(z=zhd[keep], zhel=zhel[keep], mb=mb[keep], cov=cov, icov=icov,
               icov1=icov @ one, s11=np.array(one @ icov @ one))
    np.savez(cache, **out)
    return out


def load_all():
    return dict(cc=load_cc(), bao=load_bao(), sn=load_sn())


if __name__ == "__main__":
    d = load_all()
    print("CC :", len(d["cc"]["z"]), "points, z =", d["cc"]["z"].min(), "-", d["cc"]["z"].max())
    print("BAO:", len(d["bao"]["z"]), "points:", dict(zip(*np.unique(d["bao"]["kind"], return_counts=True))))
    print("SN :", len(d["sn"]["z"]), "SNe, z =", d["sn"]["z"].min(), "-", d["sn"]["z"].max())
