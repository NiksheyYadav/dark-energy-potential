"""
Symbolic verification (sympy) of the background equations of

    S = ∫ d^4x sqrt(-g) [ (R + alpha R^2 + beta R_mn R^mn)/(2 kappa^2)
                          - (1/2)(dphi)^2 - V(phi) ] + S_m

on a spatially flat FLRW metric, using the minisuperspace trick:
keep the lapse N(t) in  ds^2 = -N^2 dt^2 + a^2 dx^2,  vary the reduced
Lagrangian with respect to N, then set N = 1.  The N-equation is the
(00) field equation, i.e. the modified Friedmann constraint.

Checks performed
  1. beta-term contribution == (1/3) * alpha-term contribution
     (so only alpha_eff = alpha + beta/3 is observable at background level).
  2. The modified Friedmann equation equals
        H^2 + 6 alpha_eff (2 H H'' - H'^2 + 6 H^2 H') = kappa^2 rho / 3 .
  3. Order-reduced correction on a matter + Lambda background:
        bracket = -(9/4) m^2  with  m = kappa^2 rho_m / 3,
     i.e.  H^2 = m + L + (27/2) alpha_eff m^2  (an a^-6 "stiff" term).
  4. Radiation era (R = 0): bracket vanishes identically.

Run:  python src/symbolic_check.py
"""
import sympy as sp

t = sp.symbols("t")
alpha, beta, kappa = sp.symbols("alpha beta kappa", positive=True)
N = sp.Function("N", positive=True)(t)
a = sp.Function("a", positive=True)(t)
phi = sp.Function("phi")(t)
V = sp.Function("V")

x, y, z = sp.symbols("x y z")
X = [t, x, y, z]
g = sp.diag(-N**2, a**2, a**2, a**2)
ginv = g.inv()


def christoffel(g, ginv, X):
    n = len(X)
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for m in range(n):
            for k in range(n):
                G[l][m][k] = sp.simplify(sum(
                    ginv[l, s] * (sp.diff(g[s, m], X[k]) + sp.diff(g[s, k], X[m])
                                  - sp.diff(g[m, k], X[s])) for s in range(n)) / 2)
    return G


def ricci(G, X):
    n = len(X)
    Ric = sp.zeros(n, n)
    for m in range(n):
        for k in range(n):
            Ric[m, k] = sp.simplify(
                sum(sp.diff(G[l][m][k], X[l]) for l in range(n))
                - sum(sp.diff(G[l][m][l], X[k]) for l in range(n))
                + sum(G[l][l][s] * G[s][m][k] for l in range(n) for s in range(n))
                - sum(G[l][k][s] * G[s][m][l] for l in range(n) for s in range(n)))
    return Ric


G = christoffel(g, ginv, X)
Ric = ricci(G, X)
Rs = sp.simplify(sum(ginv[i, i] * Ric[i, i] for i in range(4)))
RicUp = ginv * Ric * ginv                      # R^{mu nu}
RicSq = sp.simplify(sum(Ric[i, j] * RicUp[i, j] for i in range(4) for j in range(4)))
sqrtg = N * a**3


def lapse_equation(L):
    """Euler-Lagrange expression for N(t) with up to second derivatives, then N -> 1."""
    N0, N1, N2 = sp.symbols("N0 N1 N2")
    Ls = L.subs(sp.Derivative(N, (t, 2)), N2).subs(sp.Derivative(N, t), N1).subs(N, N0)
    back = {N2: sp.Derivative(N, (t, 2)), N1: sp.Derivative(N, t), N0: N}
    E = (sp.diff(Ls, N0).subs(back)
         - sp.diff(sp.diff(Ls, N1).subs(back), t)
         + sp.diff(sp.diff(Ls, N2).subs(back), t, 2))
    return sp.simplify(E.doit().subs(N, 1).doit())


E_EH = lapse_equation(sqrtg * Rs)
E_R2 = lapse_equation(sqrtg * Rs**2)
E_Ric2 = lapse_equation(sqrtg * RicSq)
E_phi = lapse_equation(sqrtg * (sp.diff(phi, t)**2 / (2 * N**2) - V(phi)))

# rewrite in terms of H(t)
H = sp.Function("H")(t)
adot = a * H
rules = {sp.Derivative(a, (t, 3)): sp.diff(adot, t, 2), sp.Derivative(a, (t, 2)): sp.diff(adot, t),
         sp.Derivative(a, t): adot}


def toH(expr):
    e = expr
    for _ in range(4):                       # iterate: derivatives of a reappear
        e = e.subs(rules).doit()
    return sp.simplify(e / a**3)


eh, r2, ric2, sc = map(toH, (E_EH, E_R2, E_Ric2, E_phi))
print("R        :", sp.simplify(toH(Rs * a**3)))
print("lapse eq, Einstein-Hilbert  :", eh)
print("lapse eq, R^2               :", sp.factor(r2))
print("lapse eq, R_mn R^mn         :", sp.factor(ric2))
print("lapse eq, scalar            :", sc)

# --- check 1: degeneracy -------------------------------------------------
ratio = sp.simplify(ric2 / r2)
print("\n[1] (R_mn R^mn term)/(R^2 term) =", ratio)
assert ratio == sp.Rational(1, 3)

# --- check 2: closed form ------------------------------------------------
Hd, Hdd = sp.diff(H, t), sp.diff(H, t, 2)
bracket = 2 * H * Hdd - Hd**2 + 6 * H**2 * Hd
# EH lapse eq is -6 H^2 ; so (E_EH + alpha E_R2)/(-6) = H^2 + 6 alpha * bracket
lhs = sp.simplify((eh + alpha * r2) / eh * H**2)
print("[2] H^2 + 6 alpha [2HH'' - H'^2 + 6H^2H'] check:",
      sp.simplify(lhs - (H**2 + 6 * alpha * bracket)) == 0)
assert sp.simplify(lhs - (H**2 + 6 * alpha * bracket)) == 0
rho_phi = sp.simplify(-sc)                    # scalar lapse eq = -(phidot^2/2 + V)
print("    scalar energy density from lapse eq:", rho_phi)

# --- check 3: order reduction on matter + Lambda ---------------------------
m0, Lam = sp.symbols("m0 Lambda", positive=True)
aa = sp.Function("a", positive=True)(t)
Hb = sp.sqrt(m0 / aa**3 + Lam)                # zeroth-order (GR) solution
sub = {sp.Derivative(aa, t): aa * Hb}
Hb_d = sp.diff(Hb, t).subs(sub)
Hb_dd = sp.diff(Hb_d, t).subs(sub)
B = sp.simplify(2 * Hb * Hb_dd - Hb_d**2 + 6 * Hb**2 * Hb_d)
print("[3] bracket on matter+Lambda background:", B, "   (m = m0/a^3)")
assert sp.simplify(B + sp.Rational(9, 4) * m0**2 / aa**6) == 0

# --- check 4: radiation ----------------------------------------------------
r0 = sp.symbols("r0", positive=True)
Hr = sp.sqrt(r0 / aa**4)
sub = {sp.Derivative(aa, t): aa * Hr}
Hr_d = sp.diff(Hr, t).subs(sub)
Hr_dd = sp.diff(Hr_d, t).subs(sub)
Br = sp.simplify(2 * Hr * Hr_dd - Hr_d**2 + 6 * Hr**2 * Hr_d)
print("[4] bracket on pure radiation background:", Br)
assert Br == 0
print("\nAll symbolic checks passed.")
