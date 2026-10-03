"""experiment_ablation_noise.py — Formal test of the central hypothesis.

The paper-level claim of Seismic Descent is that *spatially correlated* noise
is what distinguishes it from i.i.d. perturbations (SA/SGLD). This experiment
puts that claim under test:

    mode="rff"   : correlated Gaussian-field gradient (the method)
    mode="white" : i.i.d. direction, POWER-MATCHED to the RFF field at the same
                   (dim, amplitude, schedule). This is the fair control: any
                   gain of "rff" over "white" is attributable to spatial
                   correlation, not to raw perturbation magnitude.
    mode="none"  : no perturbation (pure normalized swarm descent; control 2)

Protocol
--------
- Dynamics: SeismicSwarm v20 loop, instrumented (identical to core.py; only
  the noise engine is swapped). Trials are paired: all three modes share the
  same initialization for a given trial seed.
- Power matching: the white-noise magnitude equals the median norm of the RFF
  gradient field at unit amplitude, measured by Monte-Carlo (2048 points, 16
  time samples) per dimension, then scaled by the same |amp(t)| schedule.
- Suite: 12 functions (6 classic + 6 extended), dims {2,5,10,20} by default.
- Statistics: Wilcoxon signed-rank (paired by trial) per function×dim for
  rff-vs-white and rff-vs-none; Holm-Bonferroni correction across the study.

Outputs
-------
- results/ablation_noise.json (raw scores)
- docs/audit_2026/informe_ablacion_ruido.md (report with adjusted p-values)

Usage: python -m benchmarks.experiment_ablation_noise [--trials 15] [--budget 3000]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent.functions import ALL_FUNCTIONS
from seismic_descent.functions_extended import EXTENDED_FUNCTIONS, get_search_range
from seismic_descent.rff import RandomFourierFeatures
from benchmarks.stats import summarize, wilcoxon_paired, holm_adjust, significance_stars

SUITE = {**ALL_FUNCTIONS, **EXTENDED_FUNCTIONS}


def calibrate_unit_noise_norm(rff: RandomFourierFeatures, dim: int, rng: np.random.Generator,
                              n_points: int = 2048, n_times: int = 16) -> float:
    """Median ||grad RFF|| at unit amplitude over random points and times."""
    X = rng.uniform(-1.0, 1.0, size=(n_points, dim))
    norms: List[float] = []
    for k in range(n_times):
        t = (k + 0.5) / n_times * 10.0 * np.pi
        G = rff.grad(X, t, amplitude=1.0)
        norms.append(np.linalg.norm(G, axis=1))
    return float(np.median(np.concatenate(norms)))


def run_mode(fcfg: Dict, dim: int, budget: int, seed: int, mode: str,
             n_particles: int = 10, dt_base: float = 0.2, dt_floor: float = 0.2,
             noise_amplitude: float = 0.5, n_cycles: int = 10,
             dt_cycles_multiplier: float = 5.0) -> float:
    """Instrumented v20 Seismic Descent with interchangeable noise engine."""
    r = get_search_range(fcfg, dim)
    bounds = np.array([[-r, r]] * dim)
    center = np.zeros(dim)
    half_range = np.full(dim, r)

    n_steps = max(20, budget // n_particles)
    rff = RandomFourierFeatures(dim=dim, r=64, n_octaves=1, base_lengthscale=0.4, seed=seed)
    rng = np.random.default_rng(seed)               # shared init -> paired trials
    rng_cal = np.random.default_rng(seed + 777)     # calibration stream
    rng_white = np.random.default_rng(seed + 10_000)  # white noise stream
    unit_norm = calibrate_unit_noise_norm(rff, dim, rng_cal) if mode == "white" else None

    x_norm = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    t = 0.0
    dt_noise = (n_cycles * np.pi) / n_steps

    x_real = center + x_norm * half_range
    vals = np.atleast_1d(np.asarray(fcfg["fn"](x_real), dtype=np.float64))
    best_val = float(vals.min())

    for step in range(n_steps):
        amp = noise_amplitude * np.sin(t * 2.0)
        x_real = center + x_norm * half_range
        g_real = np.asarray(fcfg["grad"](x_real), dtype=np.float64)
        if g_real.ndim == 1:
            g_real = g_real.reshape(1, -1)
        g_map = g_real * half_range
        nrm = np.linalg.norm(g_map, axis=1, keepdims=True)
        g_dir = np.where(nrm > 1e-8, g_map / nrm, 0.0)

        if mode == "rff":
            g_noise = rff.grad(x_norm, t, amplitude=amp)
        elif mode == "white":
            w = rng_white.normal(0.0, 1.0, size=(n_particles, dim))
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            g_noise = w * (unit_norm * amp)
        else:
            g_noise = 0.0

        grad = g_dir + g_noise
        cyclic = dt_floor + (1.0 - dt_floor) * np.abs(np.sin(t * dt_cycles_multiplier))
        x_norm -= (dt_base * cyclic) * grad
        np.clip(x_norm, -1.0, 1.0, out=x_norm)
        t += dt_noise

        x_real = center + x_norm * half_range
        vals = np.atleast_1d(np.asarray(fcfg["fn"](x_real), dtype=np.float64))
        best_val = min(best_val, float(vals.min()))
    return best_val


def main() -> None:
    p = argparse.ArgumentParser(description="Noise ablation: correlated vs power-matched white vs none")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--functions", type=str, nargs="+", default=sorted(SUITE))
    args = p.parse_args()

    modes = ("rff", "white", "none")
    results: Dict[str, Dict] = {"meta": vars(args) | {"modes": list(modes)}, "cells": {}}
    t_start = time.time()

    for fname in args.functions:
        fcfg = SUITE[fname]
        for dim in args.dims:
            cell_key = f"{fname}_{dim}d"
            scores = {m: [] for m in modes}
            for trial in range(args.trials):
                seed = trial + 1
                for mode in modes:  # same seed => same init => paired design
                    scores[mode].append(run_mode(fcfg, dim, args.budget, seed, mode))
            cell = {m: summarize(scores[m]) for m in modes}
            p_rw, _ = wilcoxon_paired(scores["rff"], scores["white"])
            p_rn, _ = wilcoxon_paired(scores["rff"], scores["none"])
            cell["wilcoxon_rff_vs_white"] = p_rw
            cell["wilcoxon_rff_vs_none"] = p_rn
            results["cells"][cell_key] = cell
            print(f"[{cell_key:>22s}] rff={cell['rff']['median']:.4g}  "
                  f"white={cell['white']['median']:.4g}  none={cell['none']['median']:.4g}  "
                  f"(p rff~white={p_rw:.3g} | rff~none={p_rn:.3g})", flush=True)

    # Holm correction across all cells, per comparison family
    keys = sorted(results["cells"])
    adj_rw = holm_adjust([results["cells"][k]["wilcoxon_rff_vs_white"] for k in keys])
    adj_rn = holm_adjust([results["cells"][k]["wilcoxon_rff_vs_none"] for k in keys])
    for k, a_rw, a_rn in zip(keys, adj_rw, adj_rn):
        results["cells"][k]["holm_rff_vs_white"] = float(a_rw) if a_rw == a_rw else None
        results["cells"][k]["holm_rff_vs_none"] = float(a_rn) if a_rn == a_rn else None

    elapsed = time.time() - t_start
    results["meta"]["elapsed_seconds"] = elapsed

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    with open(outdir / "ablation_noise.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- Markdown report ----
    lines: List[str] = []
    lines.append("# Informe de ablación de ruido: correlacionado vs blanco equiparado vs apagado")
    lines.append("")
    lines.append(f"> Trials: {args.trials} | Presupuesto: {args.budget} evals | "
                 f"Diseño pareado por semilla | Potencia de ruido blanco equiparada por calibración MC.")
    lines.append(f"> Generado en {elapsed:.1f}s. Estadística: Wilcoxon signed-rank + corrección de Holm "
                 f"(sobre todas las celdas). Estrellas: *** p<0.001, ** p<0.01, * p<0.05, ns. ")
    lines.append("")
    lines.append("| Celda | RFF (mediana ± IQR) | Blanco (mediana) | Sin ruido (mediana) | "
                 "rff vs blanco (p Holm) | rff vs nada (p Holm) |")
    lines.append("|---|---|---|---|---|---|")
    for k in keys:
        c = results["cells"][k]
        holm_rw = c["holm_rff_vs_white"]
        holm_rn = c["holm_rff_vs_none"]
        p1 = f"{holm_rw:.3g} {significance_stars(holm_rw)}" if holm_rw is not None else "n/a"
        p2 = f"{holm_rn:.3g} {significance_stars(holm_rn)}" if holm_rn is not None else "n/a"
        lines.append(f"| {k} | **{c['rff']['median']:.4g}** ({c['rff']['q25']:.3g}–{c['rff']['q75']:.3g}) | "
                     f"{c['white']['median']:.4g} | {c['none']['median']:.4g} | {p1} | {p2} |")
    lines.append("")
    rep_path = Path("docs/audit_2026/informe_ablacion_ruido.md")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[+] JSON: results/ablation_noise.json")
    print(f"[+] Informe: {rep_path}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
