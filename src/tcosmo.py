"""
Differentiable (PyTorch, float64, CPU) version of the background solver in cosmo.py.

Everything is batched over a leading dimension B (ensemble members / parameter sets) and
every operation is differentiable, so d(chi^2)/d(parameters, network weights) is exact
(reverse mode) and Jacobians of the data vector are exact (forward mode).

The potential is any callable  pot(phi) -> (u, du/dphi)  with phi of shape (B,).
"""
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
C100 = 2997.92458
OMR = 4.184e-5
ZI = 4.0
PHI_SCALE = 0.3


class ExpPotential:
    """u = exp(lnu0 - lam * phi)"""
    def __init__(self, lnu0, lam):
        self.lnu0, self.lam = lnu0, lam

    def __call__(self, phi):
        u = torch.exp(self.lnu0 - self.lam * phi)
        return u, -self.lam * u


class NNPotential(torch.nn.Module):
    """B independent MLPs 1-16-16-1 (tanh):  ln u(phi) = c0 + g(phi) - g(0).
    The derivative du/dphi is propagated analytically (no nested autograd)."""
    def __init__(self, B, hidden=16, seed=0, lnu0=math.log(0.32)):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        P = torch.nn.Parameter
        self.W1 = P(torch.randn(B, hidden, generator=g))
        self.b1 = P(0.5 * torch.randn(B, hidden, generator=g))
        self.W2 = P(torch.randn(B, hidden, hidden, generator=g) / math.sqrt(hidden))
        self.b2 = P(torch.zeros(B, hidden))
        self.W3 = P(0.05 * torch.randn(B, hidden, generator=g))
        self.c0 = P(torch.full((B,), lnu0))

    def _g(self, phi):
        x = phi[:, None] / PHI_SCALE
        a1 = torch.tanh(self.W1 * x + self.b1)
        d1 = (1 - a1 * a1) * self.W1
        a2 = torch.tanh(torch.einsum("bij,bj->bi", self.W2, a1) + self.b2)
        d2 = (1 - a2 * a2) * torch.einsum("bij,bj->bi", self.W2, d1)
        return (self.W3 * a2).sum(1), (self.W3 * d2).sum(1) / PHI_SCALE

    def forward(self, phi):
        g, dg = self._g(phi)
        g0, _ = self._g(torch.zeros_like(phi))
        u = torch.exp(self.c0 + g - g0)
        return u, u * dg

    def l2(self):
        return (self.W1 ** 2).sum(1) + (self.W2 ** 2).sum((1, 2)) + (self.W3 ** 2).sum(1)


# Legendre polynomials P_0..P_5 as monomial coefficient rows (c_j multiplies xi^j)
_LEG = torch.tensor([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [-0.5, 0, 1.5, 0, 0, 0],
                     [0, -1.5, 0, 2.5, 0, 0],
                     [0.375, 0, -3.75, 0, 4.375, 0],
                     [0, 1.875, 0, -8.75, 0, 7.875]])


class LegendrePerturbed:
    """ln u = ln u_base(phi) + sum_k c_k P_k(xi),  xi = 2 phi/phi_ref - 1  (function-space basis
    on the field range [0, phi_ref] traversed by the fiducial solution)."""
    def __init__(self, base, coef, phi_ref):
        self.base, self.coef, self.phi_ref = base, coef, phi_ref

    def __call__(self, phi):
        u, up = self.base(phi)
        xi = 2 * phi / self.phi_ref - 1
        pw = torch.stack([xi ** j for j in range(6)], -1)                 # (B,6)
        dpw = torch.stack([j * xi ** max(j - 1, 0) if j else torch.zeros_like(xi) for j in range(6)], -1)
        mono = self.coef @ _LEG[: self.coef.shape[-1]]                    # (B,6) monomial coeffs
        s = (mono * pw).sum(-1)
        ds = (mono * dpw).sum(-1) * 2 / self.phi_ref
        e = torch.exp(s)
        return u * e, (up + u * ds) * e


def _rhs(pot, N, phi, y, om, eps):
    m = om * math.exp(-3 * N)
    r = OMR * math.exp(-4 * N)
    u, up = pot(phi)
    H0sq = m + r + y * y / 6 + u
    H0 = torch.sqrt(H0sq)
    Hd = -0.5 * (3 * m + 4 * r + y * y)
    yd = -3 * H0 * y - 3 * up
    Hdd = 0.5 * (9 * m + 16 * r) * H0 - y * yd
    Bq = 2 * H0 * Hdd - Hd * Hd + 6 * H0sq * Hd
    H = torch.sqrt(H0sq - 6 * eps * Bq)
    return y / H, -3 * y - 3 * up / H, math.exp(-N) / H, H, u


def solve(pot, om, eps, nstep=96, zi=ZI):
    Ni = -math.log1p(zi)
    h = -Ni / nstep
    phi = torch.zeros_like(om)
    u_i, up_i = pot(phi)
    Hi = torch.sqrt(om * math.exp(-3 * Ni) + OMR * math.exp(-4 * Ni) + u_i)
    y = -(2.0 / 3.0) * up_i / Hi
    I = torch.zeros_like(om)
    Is, Hs, Ps, Ys, Us = [], [], [], [], []
    for k in range(nstep):
        N = Ni + k * h
        k1 = _rhs(pot, N, phi, y, om, eps)
        Is.append(I); Hs.append(k1[3]); Ps.append(phi); Ys.append(y); Us.append(k1[4])
        k2 = _rhs(pot, N + h / 2, phi + h / 2 * k1[0], y + h / 2 * k1[1], om, eps)
        k3 = _rhs(pot, N + h / 2, phi + h / 2 * k2[0], y + h / 2 * k2[1], om, eps)
        k4 = _rhs(pot, N + h, phi + h * k3[0], y + h * k3[1], om, eps)
        phi = phi + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        I = I + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    kf = _rhs(pot, 0.0, phi, y, om, eps)
    Is.append(I); Hs.append(kf[3]); Ps.append(phi); Ys.append(y); Us.append(kf[4])
    st = lambda L: torch.stack(L, 1)
    return dict(Ni=Ni, h=h, nstep=nstep, I=st(Is), H=st(Hs), phi=st(Ps), y=st(Ys), u=st(Us))


class Interp:
    """Pre-computed cubic-Hermite stencil for a fixed set of redshifts."""
    def __init__(self, z, nstep, zi=ZI):
        Ni = -math.log1p(zi); h = -Ni / nstep
        N = -np.log1p(np.asarray(z, dtype=float))
        x = (N - Ni) / h
        k = np.clip(np.floor(x).astype(int), 0, nstep - 1)
        t = x - k
        self.k = torch.as_tensor(k); self.h = h
        t = torch.as_tensor(t); self.t = t
        self.e0 = torch.as_tensor(np.exp(-(Ni + k * h))); self.e1 = torch.as_tensor(np.exp(-(Ni + (k + 1) * h)))
        self.c = ((2 * t ** 3 - 3 * t ** 2 + 1), (t ** 3 - 2 * t ** 2 + t) * h,
                  (-2 * t ** 3 + 3 * t ** 2), (t ** 3 - t ** 2) * h)
        self.d = ((6 * t ** 2 - 6 * t) / h, (3 * t ** 2 - 4 * t + 1),
                  (-6 * t ** 2 + 6 * t) / h, (3 * t ** 2 - 2 * t))
        self.eN = torch.as_tensor(np.exp(-N))

    def __call__(self, sol):
        I0, I1 = sol["I"][:, self.k], sol["I"][:, self.k + 1]
        H0, H1 = sol["H"][:, self.k], sol["H"][:, self.k + 1]
        d0, d1 = self.e0 / H0, self.e1 / H1
        Iz = self.c[0] * I0 + self.c[1] * d0 + self.c[2] * I1 + self.c[3] * d1
        dI = self.d[0] * I0 + self.d[1] * d0 + self.d[2] * I1 + self.d[3] * d1
        return self.eN / dI, sol["I"][:, -1:] - Iz        # H = e^{-N} / (dI/dN)


class Likelihood:
    def __init__(self, data, nstep=96):
        T = torch.as_tensor
        cc, bao, sn = data["cc"], data["bao"], data["sn"]
        self.nstep = nstep
        self.cc_H, self.cc_icov, self.cc_cov = T(cc["H"]), T(cc["icov"]), T(cc["cov"])
        self.bao_v, self.bao_icov, self.bao_cov = T(bao["val"]), T(bao["icov"]), T(bao["cov"])
        self.bao_z = T(bao["z"]); self.isDV, self.isDM = T(bao["isDV"]), T(bao["isDM"])
        self.sn_mb, self.sn_icov, self.sn_cov = T(sn["mb"]), T(sn["icov"]), T(sn["cov"])
        self.sn_icov1, self.sn_s11, self.sn_zhel = T(sn["icov1"]), float(sn["s11"]), T(sn["zhel"])
        self.i_cc, self.i_bao, self.i_sn = (Interp(d["z"], nstep) for d in (cc, bao, sn))
        self.n = (len(cc["z"]), len(bao["z"]), len(sn["z"]))

    def model(self, sol, rd):
        Hcc, _ = self.i_cc(sol)
        Hb, Ib = self.i_bao(sol)
        DM = C100 * Ib / rd[:, None]
        DH = C100 / Hb / rd[:, None]
        DV = (self.bao_z * DM * DM * DH) ** (1.0 / 3.0)
        bao = torch.where(self.isDV, DV, torch.where(self.isDM, DM, DH))
        _, Is = self.i_sn(sol)
        mu = 5.0 * torch.log10((1.0 + self.sn_zhel) * Is)
        return 100.0 * Hcc, bao, mu

    def chi2(self, sol, rd, obs=None):
        """obs: optional (cc_H, bao_v, sn_mb) with a leading batch dim (bootstrap replicas)."""
        Hcc, bao, mu = self.model(sol, rd)
        cH, bv, mb = (self.cc_H, self.bao_v, self.sn_mb) if obs is None else obs
        d = Hcc - cH
        c_cc = torch.einsum("bi,ij,bj->b", d, self.cc_icov, d)
        d = bao - bv
        c_bao = torch.einsum("bi,ij,bj->b", d, self.bao_icov, d)
        d = mb - mu
        c_sn = ((d @ self.sn_icov) * d).sum(1) - (d @ self.sn_icov1) ** 2 / self.sn_s11
        return c_cc, c_bao, c_sn
