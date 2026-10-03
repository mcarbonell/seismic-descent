"""experiment_component_ablation.py — Leave-one-out ablation of Champion v23
components + amplitude sensitivity and the sqrt(D) noise-scaling correction.

Part A: which of the 5 v23 mechanisms actually earns its place?
    full           : ORF + gravity + momentum + anisotropic metric + dt_floor
    no_gravity     : gravity_strength=0
    no_momentum    : momentum_base=0
    no_anisotropic : anisotropic_power=0
    no_noise       : noise_engine="none" (how much is the quake itself worth?)
    dt_const       : dt_floor=1.0 (constant dt; removes cyclic modulation)
    no_floor       : dt_floor=0.0 (removes the anti-freezing floor)

Part B: noise_amplitude sensitivity on the v20 core:
    - absolute:  amp = a
    - dim-normalized: amp = a * sqrt(5/D)  (the sqrt(D) correction conjectured
      in the 2026 audit: E||grad RFF|| ~ amp * sqrt(D)/ell)

Suite: 12 functions, dims {5,10,20} (Part A) / {5,10,20} (Part B),
paired trials, budget 3000 evals. Statistics: Wilcoxon paired vs `full`
(Part A) / vs absolute same-a (Part B), Holm-corrected.

Outputs: results/component_ablation.json +
         docs/audit_2026/informe_ablacion_componentes.md

Usage: python -m benchmarks.experiment_component_ablation [--trials 15]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent.functions import ALL_FUNCTIONS
from seismic_descent.functions_extended import EXTENDED_FUNCTIONS, get_search_range
from seismic_descent.champion_v23 import SeismicChampionV23
from benchmarks.experiment_ablation_noise import run_mode, SUITE
from benchmarks.stats import summarize, wilcoxon_paired, holm_adjust, significance_stars

PART_A_CONFIGS: Dict[str, Dict] = {
    "full": {},
    "no_gravity": {"gravity_strength": 0.0},
    "no_momentum": {"momentum_base": 0.0},
    "no_anisotropic": {"anisotropic_power": 0.0},
    "no_noise": {"noise_engine": "none"},
    "dt_const": {"dt_floor": 1.0},
    "no_floor": {"dt_floor": 0.0},
}

AMP_GRID = (0.25, 0.5, 0.75)


def run_v23_config(fcfg: Dict, dim: int, budget: int, seed: int, overrides: Dict) -> float:
    r = get_search_range(fcfg, dim)
    bounds = np.array([[-r, r]] * dim)
    n_steps = max(20, budget // 10)
    kwargs = dict(bounds=bounds, n_particles=10, n_steps=n_steps, noise_engine="orf", seed=seed)
    kwargs.update(overrides)
    opt = SeismicChampionV23(**kwargs)
    _, best_val, _ = opt.optimize(fn=fcfg["fn"], fn_grad=fcfg["grad"])
    return float(best_val)


def part_a(fnames: List[str], dims: List[int], trials: int, budget: int) -> Dict:
    cells: Dict[str, Dict] = {}
    for fname in fnames:
        fcfg = SUITE[fname]
        for dim in dims:
            key = f"{fname}_{dim}d"
            scores = {c: [] for c in PART_A_CONFIGS}
            rng_x0 = np.random.default_rng(999)
            for trial in range(trials):
                seed = trial + 1
                for cname, overrides in PART_A_CONFIGS.items():
                    scores[cname].append(run_v23_config(fcfg, dim, budget, seed, overrides))
            cell = {c: summarize(scores[c]) for c in PART_A_CONFIGS}
            for cname in PART_A_CONFIGS:
                if cname == "full":
                    continue
                pval, _ = wilcoxon_paired(scores[cname], scores["full"])
                cell[f"wilcoxon_{cname}_vs_full"] = pval
            cells[key] = cell
            dev = " | ".join(
                f"{c}={cell[c]['median']:.3g}" for c in PART_A_CONFIGS
            )
            print(f"[A {key:>22s}] {dev}", flush=True)
    return cells


def part_b(fnames: List[str], dims: List[int], trials: int, budget: int) -> Dict:
    cells: Dict[str, Dict] = {}
    for fname in fnames:
        fcfg = SUITE[fname]
        for dim in dims:
            key = f"{fname}_{dim}d"
            cell: Dict[str, Dict] = {}
            for a in AMP_GRID:
                abs_scores = [
                    run_mode(fcfg, dim, budget, t + 1, "rff", noise_amplitude=a)
                    for t in range(trials)
                ]
                scaled_a = a * float(np.sqrt(5.0 / dim))
                scaled_scores = [
                    run_mode(fcfg, dim, budget, t + 1, "rff", noise_amplitude=scaled_a)
                    for t in range(trials)
                ]
                pval, _ = wilcoxon_paired(scaled_scores, abs_scores)
                cell[f"amp_{a}"] = {
                    "absolute": summarize(abs_scores),
                    "dim_normalized": summarize(scaled_scores),
                    "scaled_amp_used": scaled_a,
                    "wilcoxon_scaled_vs_absolute": pval,
                }
            cells[key] = cell
            summary = " | ".join(
                f"a={a}: abs={cell[f'amp_{a}']['absolute']['median']:.3g} "
                f"dnorm={cell[f'amp_{a}']['dim_normalized']['median']:.3g}"
                for a in AMP_GRID
            )
            print(f"[B {key:>22s}] {summary}", flush=True)
    return cells


def main() -> None:
    p = argparse.ArgumentParser(description="Component ablation (v23) + amplitude sensitivity (v20)")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--functions", type=str, nargs="+", default=sorted(SUITE))
    p.add_argument("--skip-b", action="store_true", help="Skip amplitude sensitivity part")
    args = p.parse_args()

    t0 = time.time()
    out: Dict = {"meta": vars(args), "part_a": {}, "part_b": {}}

    out["part_a"] = part_a(args.functions, args.dims, args.trials, args.budget)
    if not args.skip_b:
        out["part_b"] = part_b(args.functions, args.dims, args.trials, args.budget)

    # Holm correction for Part A (each variant vs full, across cells)
    variant_names = [c for c in PART_A_CONFIGS if c != "full"]
    keys = sorted(out["part_a"])
    for v in variant_names:
        adj = holm_adjust([out["part_a"][k].get(f"wilcoxon_{v}_vs_full", np.nan) for k in keys])
        for k, a in zip(keys, adj):
            out["part_a"][k][f"holm_{v}_vs_full"] = float(a) if a == a else None

    elapsed = time.time() - t0
    out["meta"]["elapsed_seconds"] = elapsed

    resdir = Path("results")
    resdir.mkdir(exist_ok=True)
    with open(resdir / "component_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---- Report ----
    lines: List[str] = []
    lines.append("# Informe de ablación de componentes del Champion v23 + sensibilidad de amplitud")
    lines.append("")
    lines.append(f"> Trials: {args.trials} | Presupuesto: {args.budget} evals | Diseño pareado por semilla.")
    lines.append(f"> Generado en {elapsed:.1f}s. Wilcoxon pareado vs `full` (Parte A) y vs amplitud absoluta (Parte B), Holm. ")
    lines.append("")
    header = "| Celda | " + " | ".join(PART_A_CONFIGS) + " |"
    lines.append("## Parte A — leave-one-out del Champion v23 (mediana del mejor valor)\n")
    lines.append(header)
    lines.append("|---" * (len(PART_A_CONFIGS) + 1) + "|")
    for k in keys:
        row = [f"| {k}"]
        for c in PART_A_CONFIGS:
            med = out["part_a"][k][c]["median"]
            if c == "full":
                row.append(f"**{med:.4g}**")
            else:
                holm_v = out["part_a"][k].get(f"holm_{c}_vs_full")
                mark = significance_stars(holm_v) if holm_v is not None else ""
                row.append(f"{med:.4g} {mark}")
        lines.append(" | ".join(row) + " |")
    lines.append("")
    lines.append("Estrellas: diferencia significativa vs `full` tras corrección de Holm "
                 "(* p<0.05, ** p<0.01, *** p<0.001). Mejor mediana ≠ peor que full si ns.")
    if out["part_b"]:
        lines.append("\n## Parte B — sensibilidad a `noise_amplitude` (núcleo v20, mediana) y corrección √D\n")
        lines.append("| Celda | " + " | ".join(f"a={a} abs | a={a} √D-norm (p)" for a in AMP_GRID) + " |")
        lines.append("|---" * (2 * len(AMP_GRID) + 1) + "|")
        for k in sorted(out["part_b"]):
            row = [f"| {k}"]
            for a in AMP_GRID:
                cell = out["part_b"][k][f"amp_{a}"]
                pv = cell["wilcoxon_scaled_vs_absolute"]
                ptxt = f"{pv:.3g} {significance_stars(pv)}" if pv == pv else "n/a"
                row.append(f"{cell['absolute']['median']:.4g}")
                row.append(f"{cell['dim_normalized']['median']:.4g} ({ptxt})")
            lines.append(" | ".join(row) + " |")
        lines.append("")
        lines.append("La columna √D-norm usa amp' = a·√(5/D), de modo que a=0.5 coincide con abs en D=5 "
                     "y reduce la perturbación en D>5 conforme a la ley medida E‖∇ruido‖∝a·√D/ℓ.")
    rep = Path("docs/audit_2026/informe_ablacion_componentes.md")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[+] JSON: results/component_ablation.json")
    print(f"[+] Informe: {rep} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
