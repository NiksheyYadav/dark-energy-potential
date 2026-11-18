"""
Background solver + likelihoods (numpy, batched over parameter sets).

Units
-----
  H~  = H / (100 km/s/Mpc)            time in 1/(100 km/s/Mpc)
  phi in reduced Planck masses (kappa*phi), y = dphi/dt
  u(phi) = kappa^2 V / (3 * 100^2)    so that H~^2 = om a^-3 + orad a^-4 + y^2/6 + u
  eps~   = alpha_eff * (100 km/s/Mpc)^2 / c^2,   alpha_eff = alpha + beta/3
  h = H~(z=0) is a DERIVED quantity (no shooting needed).

Equations (N = ln a, order-reduced in eps~, see docs/REPORT.md Sec. 3)
-----------------------------------------------------------------------
  H~^2   = om e^{-3N} + orad e^{-4N} + y^2/6 + u(phi) - 6 eps~ B
  B      = 2 H Hddot - Hdot^2 + 6 H^2 Hdot   evaluated on the eps~=0 relations
  phi'   = y / H~
  y'     = -3 y - 3 u_phi / H~
  I'     = e^{-N} / H~            ->  D_C(z) = (c/100) [I(0) - I(N)]  Mpc

Model family used for MCMC:  u(phi) = u0 exp(-lam phi)
  theta = (om, u0, lam, eps, rd);  LCDM is lam = 0, eps = 0 exactly.
Initial conditions at z_i: phi = 0, y = matter-era thawing attractor
  y_i = -(2/3) u_phi(0) / H_i.
"""
import os
import numpy as np

EPS_ORDER = int(os.environ.get("EPS_ORDER", 1))   # 2: add the eps^2 term of the GR-continuous series
if EPS_ORDER == 2:                                 # (src/order_reduction.py; matter + radiation + Lambda only)
    from order_reduction import series
    _F2 = [(e[0], e[1], e[2], float(c)) for e, c in series(2)[2].items()]

C100 = 2997.92458            # c / (100 km/s/Mpc) in Mpc
OMR = 4.184e-5               # omega_gamma (T0 = 2.7255 K) * (1 + 0.2271 * 3.046)
ZI = 4.0
NSTEP = 256


def _rhs(N, phi, y, om, u0, lam, eps, c2=0.0):
    m = om * np.exp(-3.0 * N)
    r = OMR * np.exp(-4.0 * N)
    u = u0 * np.exp(-lam * phi + 0.5 * c2 * phi * phi)      # ln u = ln u0 - lam phi + c2 phi^2 / 2
    up = (-lam + c2 * phi) * u
    H0sq = m + r + y * y / 6.0 + u
    H0 = np.sqrt(H0sq)
    Hd = -0.5 * (3.0 * m + 4.0 * r + y * y)
    yd = -3.0 * H0 * y - 3.0 * up
    Hdd = 0.5 * (9.0 * m + 16.0 * r) * H0 - y * yd
    B = 2.0 * H0 * Hdd - Hd * Hd + 6.0 * H0sq * Hd
    Hsq = H0sq - 6.0 * eps * B
    if EPS_ORDER == 2:                             # only where V is constant (lam = c2 = 0, so y = 0)
        f2 = sum(c * m ** i * r ** j * u ** l for i, j, l, c in _F2)
        Hsq = np.where((lam == 0) & (np.asarray(c2) == 0), Hsq + eps * eps * f2, Hsq)
    H = np.sqrt(np.where(Hsq > 0, Hsq, np.nan))
    return y / H, -3.0 * y - 3.0 * up / H, np.exp(-N) / H, H


def solve(theta, nstep=NSTEP, zi=ZI):
    """theta: (B, >=4) array of (om, u0, lam, eps).  Returns dict of grids, each (B, nstep+1)."""
    theta = np.atleast_2d(theta)
    om, u0, lam, eps = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    c2 = theta[:, 5] if theta.shape[1] > 5 else 0.0          # optional curvature of ln V (column 5)
    Ni = -np.log1p(zi)
    h = -Ni / nstep
    Ngrid = Ni + h * np.arange(nstep + 1)
    Bn = theta.shape[0]
    phi = np.zeros(Bn)
    Hi = np.sqrt(om * np.exp(-3 * Ni) + OMR * np.exp(-4 * Ni) + u0)
    y = (2.0 / 3.0) * lam * u0 / Hi
    I = np.zeros(Bn)
    Is = np.empty((Bn, nstep + 1)); Hs = np.empty((Bn, nstep + 1))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for k in range(nstep):
            N = Ngrid[k]
            k1 = _rhs(N, phi, y, om, u0, lam, eps, c2)
            Is[:, k] = I; Hs[:, k] = k1[3]
            k2 = _rhs(N + h / 2, phi + h / 2 * k1[0], y + h / 2 * k1[1], om, u0, lam, eps, c2)
            k3 = _rhs(N + h / 2, phi + h / 2 * k2[0], y + h / 2 * k2[1], om, u0, lam, eps, c2)
            k4 = _rhs(N + h, phi + h * k3[0], y + h * k3[1], om, u0, lam, eps, c2)
            phi = phi + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            I = I + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        kf = _rhs(0.0, phi, y, om, u0, lam, eps, c2)
    Is[:, -1] = I; Hs[:, -1] = kf[3]
    u_now = u0 * np.exp(-lam * phi + 0.5 * c2 * phi * phi)
    w0 = (y * y / 6 - u_now) / (y * y / 6 + u_now)
    return dict(N=Ngrid, h=h, I=Is, H=Hs, phi0=phi, y0=y, w0=w0, hub=kf[3], vratio=u_now / u0)


def observables(sol, z):
    """Return (H~(z), I~(z) = int_0^z dz'/H~) for each batch member; cubic Hermite in N."""
    N = -np.log1p(np.asarray(z))
    x = (N - sol["N"][0]) / sol["h"]
    k = np.clip(np.floor(x).astype(int), 0, len(sol["N"]) - 2)
    t = x - k
    h = sol["h"]
    I0, I1 = sol["I"][:, k], sol["I"][:, k + 1]
    H0, H1 = sol["H"][:, k], sol["H"][:, k + 1]
    d0 = np.exp(-sol["N"][k]) / H0
    d1 = np.exp(-sol["N"][k + 1]) / H1
    t2, t3 = t * t, t * t * t
    Iz = ((2 * t3 - 3 * t2 + 1) * I0 + (t3 - 2 * t2 + t) * h * d0
          + (-2 * t3 + 3 * t2) * I1 + (t3 - t2) * h * d1)
    # H from the derivative of the same Hermite interpolant: H = e^{-N} / (dI/dN)
    dI = ((6 * t2 - 6 * t) * I0 + (3 * t2 - 4 * t + 1) * h * d0
          + (-6 * t2 + 6 * t) * I1 + (3 * t2 - 2 * t) * h * d1) / h
    Hz = np.exp(-N) / dI
    return Hz, sol["I"][:, -1:] - Iz


def chi2_parts(theta, data, nstep=NSTEP):
    """theta (B,5) = (om, u0, lam, eps, rd).  Returns chi2_cc, chi2_bao, chi2_sn, sol."""
    theta = np.atleast_2d(theta)
    rd = theta[:, 4]
    sol = solve(theta, nstep=nstep)
    cc, bao, sn = data["cc"], data["bao"], data["sn"]
    # cosmic chronometers
    Hcc, _ = observables(sol, cc["z"])
    d = 100.0 * Hcc - cc["H"]
    c_cc = np.einsum("bi,ij,bj->b", d, cc["icov"], d)
    # BAO
    Hb, Ib = observables(sol, bao["z"])
    DM = C100 * Ib / rd[:, None]
    DH = C100 / Hb / rd[:, None]
    DV = (bao["z"] * DM * DM * DH) ** (1.0 / 3.0)
    model = np.where(bao["isDV"], DV, np.where(bao["isDM"], DM, DH))
    d = model - bao["val"]
    c_bao = np.einsum("bi,ij,bj->b", d, bao["icov"], d)
    # supernovae, M marginalised analytically
    _, Is = observables(sol, sn["z"])
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = 5.0 * np.log10((1.0 + sn["zhel"]) * Is)
    d = sn["mb"] - mu
    q = d @ sn["icov"]
    c_sn = np.einsum("bi,bi->b", q, d) - (d @ sn["icov1"]) ** 2 / sn["s11"]
    return c_cc, c_bao, c_sn, sol


def chi2_total(theta, data, use=("cc", "bao", "sn")):
    c_cc, c_bao, c_sn, sol = chi2_parts(theta, data)
    tot = np.zeros_like(c_cc)
    if "cc" in use: tot = tot + c_cc
    if "bao" in use: tot = tot + c_bao
    if "sn" in use: tot = tot + c_sn
    return np.where(np.isfinite(tot), tot, np.inf), sol
