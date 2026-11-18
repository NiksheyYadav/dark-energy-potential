# dark-energy-potential

Code, data and paper source for *How much of the dark-energy potential can late-time data see? Differentiable forward modelling of V(φ) in quadratic gravity* (draft, not yet published).

A free-form quintessence potential V(φ), represented by a small neural network, sits inside the background field equations and is fitted to cosmic chronometers, DESI DR2 BAO and Pantheon+ by differentiating through the ODE solver. The code also includes:
- an exact Fisher analysis of which features of V(φ) the data measure;
- mock-based coverage tests of the uncertainties;
- the R + αR² + βR_μνR^μν correction, with its order reduction worked out to all orders.

## Layout

| Path | What it is |
|---|---|
| `src/symbolic_check.py` | sympy proof that only α + β/3 enters the FLRW equations, and the a⁻⁶ form of the correction |
| `src/order_reduction.py` | The order-reduced Friedmann equation to all orders in ε (exact rational coefficients; the series diverges) |
| `src/data.py` | Loaders for cosmic chronometers, DESI DR2 BAO and Pantheon+, with full covariances |
| `src/cosmo.py` | Batched numpy RK4 background solver and χ² (used by emcee) |
| `src/tcosmo.py` | PyTorch differentiable solver, neural potential and likelihood |
| `src/validate.py` | Solver against quadrature and DOP853; DESI-only ΛCDM baseline |
| `src/fit_mcmc.py` | emcee fits: `lcdm`, `lcdm_eps`, `exp`, `exp_eps`, with `--rd free` or `--rd planck` |
| `src/neural_potential.py` | Free-form V(φ) with a bootstrap ensemble; `--mock` runs the closure test |
| `src/fisher.py` | Autodiff against finite differences, Fisher against MCMC, identifiability of V(φ) |
| `src/extra_checks.py` | Cross-checks, the ε profile, and the full third-order equation against the reduced branch |
| `src/coverage_*.py` | Mock coverage study: mocks (`coverage_common`), neural ensembles (`coverage_nn`), emcee (`coverage_mcmc`), summary (`coverage_analysis`) |
| `src/w0_pileup_check.py` | Saved-sample rerun showing why slope + curvature chains pile w0 at −1 |
| `src/analysis.py` | Builds `results/summary.md` and the figures |
| `data/` | Small public data files, plus `fetch_data.py` for Pantheon+ |
| `results/` | Run summaries (`.json`, `.md`, `.log`); chains and ensembles (`.npz`) are not tracked |
| `figures/` | Figures used in the paper |
| `paper/` | LaTeX source (`main.tex`, `refs.bib`) and helpers: `make_bib.py` builds `refs.bib` from the arXiv API, `lint_tex.py` runs static checks, `citation_trail.py` searches citing papers, `make_zip.py` packs an Overleaf upload |

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python data/fetch_data.py
```

```bash
python src/symbolic_check.py
```

```bash
python src/validate.py
```

```bash
python src/fit_mcmc.py --model exp_eps --rd planck --steps 4000
```

```bash
python src/neural_potential.py --rd planck
```

```bash
python src/fisher.py
```

```bash
python src/analysis.py
```

The coverage study runs `coverage_nn.py`, then `coverage_mcmc.py --model poly2` and `--model exp`, then `coverage_analysis.py`. On a 16-thread laptop the 1200-network ensemble takes about 45 minutes and each set of 50 chains a few hours.

### Switches (environment variables)

| Variable | Effect |
|---|---|
| `COV_TRUTH=B` | Coverage study at the second, barely rolling truth (λ = 0.3) instead of the real-data best fit |
| `COV_SEED`, `COV_NMOCK` | Noise seed and number of mock skies (default 2026 and 50); a non-default seed gets its own file suffix |
| `CC_SET=32` | Use the 32-point chronometer compilation instead of the 15 points with a full covariance |
| `EPS_ORDER=2` | Add the second-order term of the order reduction (ΛCDM + ε only) |

## Data sources

| File | Source |
|---|---|
| `Pantheon+SH0ES.dat`, `Pantheon+SH0ES_STAT+SYS.cov` | github.com/PantheonPlusSH0ES/DataRelease, `Pantheon+_Data/4_DISTANCES_AND_COVAR` (downloaded and checksum-verified by `data/fetch_data.py`) |
| `desi_dr2_mean.txt`, `desi_dr2_cov.txt` | github.com/CobayaSampler/bao_data, `desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt` |
| `HzTable_MM_BC03.dat`, `data_MM20.dat` | gitlab.com/mmoresco/CCcovariance, `data/` |
| `HzTable_CC_extra17.dat` | The other 17 points of the 32-point compilation, from Moresco et al. 2022 (arXiv:2201.07241), Table 1 |

## Building the paper

```bash
tectonic paper/main.tex --outdir paper/build
```

Or upload `main.tex`, `refs.bib` and the figures to Overleaf (`python paper/make_zip.py` packs them).

## Use of AI tools

Claude (Anthropic) assisted with writing and debugging the code, running and summarising the analyses, checking references and drafting the paper. All results were checked by the author, who takes responsibility for them.

## Licence and citation

Code: MIT licence (see `LICENSE`). The data files keep the terms of their original sources (table above). If you use this code, please cite the paper; a citation entry will be added when it is on arXiv.
