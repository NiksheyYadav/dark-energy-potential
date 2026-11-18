"""
Collect results/*.json|npz into results/summary.md and figures/*.png.
  python src/analysis.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import ROOT

R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
SH0ES, PLANCK = (73.04, 1.04), (67.36, 0.54)
MODELS = ["lcdm", "lcdm_eps", "exp", "exp_eps"]
LABEL = {"lcdm": "ΛCDM", "lcdm_eps": "ΛCDM + ε", "exp": "exp-V", "exp_eps": "exp-V + ε"}
lines = []
P = lines.append


def pm(d, f="%.4f"):
    return (f + " (+" + f + " / −" + f + ")") % (d["median"], d["plus"], d["minus"])


def tension(d, ref):
    s = 0.5 * (d["plus"] + d["minus"])
    return abs(d["median"] - ref[0]) / np.hypot(s, ref[1])


for rd in ("free", "planck"):
    P(f"\n### r_d prior: {rd}\n")
    P("| model | k | ω_m | H0 [km/s/Mpc] | Ω_m | λ (95% up) | ε=α_eff H0²/c² (95% up) | w0 | r_d [Mpc] |")
    P("|---|---|---|---|---|---|---|---|---|")
    S = {}
    for m in MODELS:
        f = os.path.join(R, f"{m}_{rd}.json")
        if not os.path.exists(f):
            continue
        s = S[m] = json.load(open(f))
        p, d = s["params"], s["derived"]
        lam = "< %.2f" % p["lam"]["up95"] if "lam" in p else "0 (fixed)"
        eps = "< %.1e" % d["eps_H0"]["up95"] if "eps" in p else "0 (fixed)"
        w0 = pm(d["w0"], "%.3f") if "lam" in p else "−1"
        P(f"| {LABEL[m]} | {s['k']} | {pm(p['om'])} | {pm(d['H0'], '%.2f')} | {pm(d['Om'], '%.4f')} | {lam} | {eps} | {w0} | {pm(p['rd'], '%.2f')} |")
    P("\n| model | χ²_CC | χ²_BAO | χ²_SN | χ²_rd-prior | χ²_tot | Δχ² | ΔAIC | ΔBIC | H0 vs SH0ES | H0 vs Planck | τ_max | N_eff | acc. |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, s in S.items():
        c, b = s["chi2"], S["lcdm"]
        P("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %+.2f | %+.2f | %+.2f | %.1fσ | %.1fσ | %.0f | %.0f | %.2f |" % (
            LABEL[m], c["cc"], c["bao"], c["sn"], c["rd_prior"], c["total"], c["total"] - b["chi2"]["total"],
            s["AIC"] - b["AIC"], s["BIC"] - b["BIC"], tension(s["derived"]["H0"], SH0ES),
            tension(s["derived"]["H0"], PLANCK), np.nanmax(s["tau"]), s["n_eff_min"], s["acceptance"]))
    P("\nBest-fit points: " + "; ".join("%s: %s, H0=%.2f, w0=%.3f" % (
        LABEL[m], {k: float("%.4g" % v) for k, v in s["best"].items()}, s["best_H0"], s["best_w0"]) for m, s in S.items()))

# ---------------- triangle plot ----------------
try:
    from getdist import MCSamples, plots
    samples = []
    for m, col in (("lcdm", "gray"), ("exp_eps", "C0")):
        f = os.path.join(R, f"{m}_planck.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        names = ["H0", "Om"] + (["lam", "eps4"] if m == "exp_eps" else [])
        th = d["thin"]
        cols = [d["d_H0"], d["d_Om"]] + ([th[:, 2], 1e4 * d["d_eps_H0"]] if m == "exp_eps" else [])
        labels = ["H_0", r"\Omega_m"] + ([r"\lambda", r"10^4\epsilon"] if m == "exp_eps" else [])
        rng = {"lam": (0, None), "eps4": (0, None)} if m == "exp_eps" else {}
        samples.append(MCSamples(samples=np.column_stack(cols), names=names, labels=labels, ranges=rng,
                                 label=LABEL[m], settings=dict(smooth_scale_2D=0.5, smooth_scale_1D=0.5)))
    if samples:
        g = plots.get_subplot_plotter(width_inch=7)
        g.triangle_plot(samples[::-1], ["H0", "Om", "lam", "eps4"], filled=True)
        g.export(os.path.join(FIG, "triangle_planck_rd.png"))
        P("\n![posteriors](../figures/triangle_planck_rd.png)")
except Exception as e:                                         # plotting must never kill the summary
    P(f"\n(triangle plot skipped: {e})")

# ---------------- neural potential ----------------
for tag in ("nn_planck", "nn_planck_eps", "nn_free", "nn_planck_mock"):
    f = os.path.join(R, tag + ".npz")
    if not os.path.exists(f):
        continue
    d = np.load(f)
    s = json.load(open(os.path.join(R, tag + ".json")))
    P(f"\n### Neural potential: {tag}\n")
    P("```json\n" + json.dumps({k: s[k] for k in ("real", "ensemble_16_50_84", "runtime_s", "B", "iters",
                                                  "n_params_per_member") if k in s}, indent=1) + "\n```")
    z = d["z_nodes"]; sel = z <= 2.5
    dphi = np.abs(d["phi"] - d["phi"][:, :1])
    lnV = np.log(d["u"] / d["u"][:, :1])                     # ln V(phi)/V(phi_i) along the trajectory
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, y, lab in ((ax[0], d["w"], "w(z)"), (ax[1], np.exp(lnV), r"$V(\phi(z))/V(\phi_i)$"),
                      (ax[2], dphi, r"$|\Delta\phi|(z)$  [$M_{\rm pl}$]")):
        lo, med, hi = np.percentile(y[1:], [15.865, 50, 84.135], axis=0)
        lo2, hi2 = np.percentile(y[1:], [2.275, 97.725], axis=0)
        a.fill_between(z[sel], lo2[sel], hi2[sel], color="C0", alpha=0.15, label="ensemble 95%")
        a.fill_between(z[sel], lo[sel], hi[sel], color="C0", alpha=0.35, label="ensemble 68%")
        a.plot(z[sel], y[0][sel], "k-", lw=1.2, label="fit to unperturbed data")
        a.set_xlabel("z"); a.set_ylabel(lab)
    if "truth_u" in d.files:
        tu, tp = d["truth_u"], d["truth_phi"]
        ax[1].plot(z[sel], (tu / tu[0])[sel], "r--", label="truth")
        ax[2].plot(z[sel], np.abs(tp - tp[0])[sel], "r--", label="truth")
        lam = float(d["truth_lam"])
        # truth w(z) is not stored; reconstruct from u and phi via finite differences is noisy -> skip
    ax[0].axhline(-1, color="gray", lw=0.6)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, tag + ".png"), dpi=130); plt.close(fig)
    P(f"\n![{tag}](../figures/{tag}.png)")
    # slope / curvature of ln V over the traversed range (member-wise quadratic fit in Delta phi)
    lamE, curv = [], []
    for i in range(len(d["phi"])):
        x = d["phi"][i] - d["phi"][i, 0]
        if np.ptp(x) < 1e-4:
            continue
        c = np.polyfit(np.abs(x), lnV[i], 2)
        lamE.append(-c[1]); curv.append(2 * c[0])
    if lamE:
        q = lambda v: "%.2f (+%.2f / −%.2f)" % (np.median(v), np.percentile(v, 84.135) - np.median(v),
                                                 np.median(v) - np.percentile(v, 15.865))
        P(f"\nQuadratic fit ln V = −λ_eff|Δφ| + ½ c2 Δφ² over the traversed range: λ_eff = {q(lamE)}, c2 = {q(curv)}")

f = os.path.join(R, "fisher.json")
if os.path.exists(f):
    P("\n### Fisher / identifiability (fisher.py)\n\n```json\n" + json.dumps(json.load(open(f)), indent=1)[:6000] + "\n```")

with open(os.path.join(R, "summary.md"), "w", encoding="utf-8") as fh:
    fh.write("# Fit results (auto-generated by src/analysis.py)\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
