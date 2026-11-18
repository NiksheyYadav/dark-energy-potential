"""Order reduction of the quadratic-gravity Friedmann equation to all orders in eps (REPORT 4.8, paper Sec. 2.3).

For matter + radiation + Lambda, H^2 = u(a) solves  u + 6 eps B[u] = u0,  u0 = m + r + L,
  B[u] = 2 H Hddot - Hdot^2 + 6 H^2 Hdot = 4 a u u' + a^2 u u'' - (a^2/4) u'^2      (' = d/da).
The branch continuous with GR is the power series u = sum_k eps^k f_k with
  f_0 = u0,   f_k = -6 sum_{i<k} B(f_i, f_{k-1-i}),
where B(.,.) is the symmetric bilinear form of B[u].  On monomials M ~ a^{-n} it is algebraic:
  B(M1, M2) = M1 M2 [ -2(n1+n2) + (n1^2 + n1 + n2^2 + n2)/2 - n1 n2/4 ].
Monomials are m^i r^j L^l with m = om a^-3, r = orad a^-4 (n = 3i + 4j).  Exact rational arithmetic.
Order 1 reproduces (27/2) eps m^2 - 12 eps m r + 24 eps L r ... (checked against cosmo._rhs);
order 2 adds -729 eps^2 m^3 - 1458 eps^2 L m^2 + (radiation terms).

  python src/order_reduction.py     ->  coefficients and the convergence of the series at the bound
"""
from fractions import Fraction as F
import numpy as np


def b(n1, n2):
    return F(-2 * (n1 + n2)) + F(n1 * n1 + n1 + n2 * n2 + n2, 2) - F(n1 * n2, 4)


def series(order, radiation=True):
    """List of dicts {(i, j, l): coeff} for f_0..f_order (monomial m^i r^j L^l)."""
    f = [{(1, 0, 0): F(1), (0, 0, 1): F(1)}]
    if radiation:
        f[0][(0, 1, 0)] = F(1)
    for k in range(1, order + 1):
        out = {}
        for i in range(k):
            for e1, c1 in f[i].items():
                for e2, c2 in f[k - 1 - i].items():
                    n1, n2 = 3 * e1[0] + 4 * e1[1], 3 * e2[0] + 4 * e2[1]
                    e = (e1[0] + e2[0], e1[1] + e2[1], e1[2] + e2[2])
                    out[e] = out.get(e, F(0)) - 6 * c1 * c2 * b(n1, n2)
        f.append({e: c for e, c in out.items() if c != 0})
    return f


def hsq(m, r, L, eps, order, f=None):
    """sum_{k<=order} eps^k f_k evaluated on arrays m, r, L (any broadcastable shapes)."""
    f = f or series(order)
    tot = 0.0
    for k in range(order + 1):
        fk = sum(float(c) * m ** e[0] * r ** e[1] * L ** e[2] for e, c in f[k].items())
        tot = tot + eps ** k * fk
    return tot


if __name__ == "__main__":
    f = series(3, radiation=False)
    for k in range(4):
        print(f"f_{k}:", {f"m^{e[0]} L^{e[2]}": str(c) for e, c in sorted(f[k].items())})
    # convergence of the matter + Lambda series at the first-order 95% bound, in H0 units (paper's eps)
    K = 40
    f = series(K, radiation=False)
    for eps in (1e-4, 4.1e-4):
        for z in (1.0, 2.33, 4.0):
            m, L = 0.3 * (1 + z) ** 3, 0.7
            t = np.array([eps ** k * sum(float(c) * m ** e[0] * L ** e[2] for e, c in f[k].items()) for k in range(K + 1)])
            ratio = np.abs(t[1:] / t[:-1])
            print(f"eps={eps:.1e} z={z:4.2f}: H^2 correction order1 {t[1] / t[0]:+.4f}, order2/order1 {t[2] / t[1]:+.3f}, "
                  f"|t_k+1/t_k| at k=10,20,39: {ratio[10]:.3f} {ratio[20]:.3f} {ratio[39]:.3f}, "
                  f"sum(1..{K})/order1 {t[1:].sum() / t[1]:.4f}")
