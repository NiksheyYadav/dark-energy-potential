"""
Baseline validation.  Run:  python src/validate.py

  V1  RK4 solver vs. direct quadrature for flat LCDM (H and comoving distance).
  V2  RK4 solver vs. scipy DOP853 (rtol 1e-11) for the full model (lam, eps != 0).
  V3  analytic check: LCDM + eps  ==  H^2 = m + r-terms + L + (27/2) eps m^2  (radiation off).
  V4  DESI DR2 BAO-only flat LCDM fit (Omega_m, h*r_d) to compare with arXiv:2503.14738.
  V5  timing of one batched likelihood call.
"""
import time
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize
import cosmo
from cosmo import C100, OMR, solve, observables, chi2_parts
from data import load_all

zs = np.array([0.01, 0.05, 0.3, 0.7, 1.0, 1.5, 2.33, 3.5])

# ---------------- V1 ----------------
om, u0 = 0.1430, 0.3166
th = np.array([[om, u0, 0.0, 0.0, 147.0]])
sol = solve(th)
Hn, In = observables(sol, zs)
Hex = np.sqrt(om * (1 + zs) ** 3 + OMR * (1 + zs) ** 4 + u0)
Iex = np.array([quad(lambda x: 1 / np.sqrt(om * (1 + x) ** 3 + OMR * (1 + x) ** 4 + u0), 0, z,
                     epsabs=1e-13, epsrel=1e-13)[0] for z in zs])
print("V1  LCDM  max |dH/H| = %.2e   max |dD/D| = %.2e" %
      (np.abs(Hn[0] / Hex - 1).max(), np.abs(In[0] / Iex - 1).max()))

# ---------------- V2 ----------------
p = (0.14, 0.33, 1.2, 3e-4)


def f(N, s):
    a, b, c, _ = cosmo._rhs(N, s[0], s[1], *p)
    return [a, b, c]


Ni = -np.log1p(cosmo.ZI)
Hi = np.sqrt(p[0] * np.exp(-3 * Ni) + OMR * np.exp(-4 * Ni) + p[1])
s0 = [0.0, (2 / 3) * p[2] * p[1] / Hi, 0.0]
t0 = time.perf_counter()
ref = solve_ivp(f, [Ni, 0], s0, method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True)
t_ref = time.perf_counter() - t0
Nz = -np.log1p(zs)
Iref = ref.y[2, -1] - ref.sol(Nz)[2]
Href = np.array([cosmo._rhs(n, *ref.sol(n)[:2], *p)[3] for n in Nz])
for n in (64, 128, 256):
    t0 = time.perf_counter()
    sol = solve(np.array([[*p, 147.0]]), nstep=n)
    t_rk = time.perf_counter() - t0
    Hn, In = observables(sol, zs)
    print("V2  nstep=%3d  max |dH/H| = %.2e  max |dD/D| = %.2e   (RK4 %.1f ms, DOP853 %.1f ms)" %
          (n, np.abs(Hn[0] / Href - 1).max(), np.abs(In[0] / Iref - 1).max(), 1e3 * t_rk, 1e3 * t_ref))
print("    phi(0) = %.4f M_pl, w0 = %.4f, h = %.4f" % (sol["phi0"][0], sol["w0"][0], sol["hub"][0]))

# ---------------- V3 ----------------
save = cosmo.OMR
cosmo.OMR = 0.0
eps = 5e-4
sol = solve(np.array([[om, u0, 0.0, eps, 147.0]]))
Hn, _ = observables(sol, zs)
Han = np.sqrt(om * (1 + zs) ** 3 + u0 + 13.5 * eps * om ** 2 * (1 + zs) ** 6)
print("V3  LCDM+eps vs analytic a^-6 term: max |dH/H| = %.2e" % np.abs(Hn[0] / Han - 1).max())
cosmo.OMR = save

# ---------------- V4 ----------------
data = load_all()


def chi2_bao(x):
    Om, hrd = x                                   # only Omega_m and h*r_d matter for BAO
    hh = 0.68
    th = np.array([[Om * hh ** 2, (1 - Om) * hh ** 2 - OMR, 0.0, 0.0, hrd / hh]])
    return chi2_parts(th, data)[1][0]


res = minimize(chi2_bao, [0.3, 101.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8))
# Gaussian errors from the numerical Hessian
x0, e = res.x, np.array([1e-3, 1e-1])
Hm = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        ei, ej = np.eye(2)[i] * e[i], np.eye(2)[j] * e[j]
        Hm[i, j] = (chi2_bao(x0 + ei + ej) - chi2_bao(x0 + ei - ej)
                    - chi2_bao(x0 - ei + ej) + chi2_bao(x0 - ei - ej)) / (4 * e[i] * e[j])
cov = np.linalg.inv(0.5 * Hm)
print("V4  DESI DR2 BAO only, flat LCDM:  Omega_m = %.4f +- %.4f,  h*r_d = %.2f +- %.2f Mpc,"
      "  chi2 = %.2f (13 points)" % (x0[0], np.sqrt(cov[0, 0]), x0[1], np.sqrt(cov[1, 1]), res.fun))

# ---------------- V5 ----------------
B = 16
th = np.tile(np.array([[0.143, 0.3166, 0.5, 1e-4, 147.0]]), (B, 1))
t0 = time.perf_counter()
for _ in range(5):
    chi2_parts(th, data)
print("V5  one batched likelihood call (B=%d): %.0f ms" % (B, 1e3 * (time.perf_counter() - t0) / 5))
c = chi2_parts(th[:1], data)
print("    chi2 at test point: CC %.2f  BAO %.2f  SN %.2f" % (c[0][0], c[1][0], c[2][0]))
