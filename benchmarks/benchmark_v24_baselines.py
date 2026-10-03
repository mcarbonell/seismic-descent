"""benchmark_v24_baselines.py — Publication-grade comparative benchmark.

Algorithms (all seeded, paired initializers):
  seismic_v20     : SeismicSwarm v20 base (RFF)
  seismic_v23_orf : SeismicChampionV23 (ORF, full synthesis)
  seismic_v24_orf : v23 + dimension-normalized amplitude sqrt(5/D) (identical to v23 at D=5)
  seismic_v24r1_orf: v24 with ref_dim=1 (recommended constant, train/test-validated)
  seismic_v25c_orf : experimental adaptive amplitude (coherence-gated; benchmarks only)
  cmaes           : plain CMA-ES (pycma, seeded, sigma0=0.2*span) [historical baseline]
  ipop_cmaes      : IPOP-CMA-ES restarts with doubling population [FAIR multimodal baseline]
  pso             : global-best PSO (Clerc-Kennedy constriction coefficients)
  bfgs_ms         : L-BFGS-B multistart w/ analytic gradients (first-order control)
  random          : uniform random search (sanity floor)

Phase 1: 12 functions x dims {5,10,20} x budget 3000 x 15 trials.
Phase 2: budget-scaling crossover (rastrigin, griewank; 5D; budgets 3k/10k/25k),
         including IPOP-CMA-ES so the "CMA-ES infarction" narrative is tested
         against the *restart* variant, not a strawman.

Statistics: median/IQR + Wilcoxon signed-rank vs the best-performing baseline
per cell, Holm-corrected. Gradient evaluations of seismic runs consume the
same 1-unit-per-eval convention as baselines (declared in the report).

Outputs: results/baselines_v24.json + docs/audit_2026/informe_baselines_v24.md
Usage: python -m benchmarks.benchmark_v24_baselines [--trials 15] [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent.functions import ALL_FUNCTIONS
from seismic_descent.functions_extended import EXTENDED_FUNCTIONS, get_search_range
from seismic_descent.core import SeismicSwarm
from seismic_descent.champion_v23 import SeismicChampionV23
from benchmarks.baselines import (
    run_random_search, run_cmaes, run_ipop_cmaes, run_pso, run_bfgs_multistart,
)
from benchmarks.stats import summarize, wilcoxon_paired, holm_adjust, significance_stars
from benchmarks.experiment_v25_adaptive_amp import SeismicChampionV25

SUITE = {**ALL_FUNCTIONS, **EXTENDED_FUNCTIONS}
N_PARTICLES = 10


def run_seismic(fcfg: Dict, dim: int, budget: int, seed: int, x0: np.ndarray,
                arch: str) -> float:
    r = get_search_range(fcfg, dim)
    bounds = np.array([[-r, r]] * dim)
    n_steps = max(20, budget // N_PARTICLES)
    if arch == "v20":
        opt = SeismicSwarm(bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps, seed=seed)
    elif arch == "v24":
        # v24 candidate = v23 + dimension-normalized amplitude (sqrt(5/D)).
        # Identical to v23 at D=5 by construction; the difference shows at D=10/20.
        opt = SeismicChampionV23(bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps,
                                 noise_engine="orf", noise_amp_dim_normalized=True, seed=seed)
    elif arch == "v24r1":
        # Recommended config (findings_v25): ref_dim = 1 (train/test-validated constant).
        opt = SeismicChampionV23(bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps,
                                 noise_engine="orf", noise_amp_dim_normalized=True,
                                 ref_dim=1.0, seed=seed)
    elif arch == "v25c":
        # Experimental adaptive amplitude (coherence-gated; outside the package).
        opt = SeismicChampionV25(mode="coherence", bounds=bounds,
                                 n_particles=N_PARTICLES, n_steps=n_steps,
                                 noise_engine="orf", seed=seed)
    else:
        opt = SeismicChampionV23(bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps,
                                 noise_engine="orf", seed=seed)
    _, val, _ = opt.optimize(fn=fcfg["fn"], fn_grad=fcfg["grad"], x0=x0)
    return float(val)


def run_algo(algo: str, fcfg: Dict, dim: int, budget: int, seed: int, x0: np.ndarray) -> float:
    r = get_search_range(fcfg, dim)
    bounds = np.array([[-r, r]] * dim)
    if algo == "seismic_v20":
        return run_seismic(fcfg, dim, budget, seed, x0, "v20")
    if algo == "seismic_v23_orf":
        return run_seismic(fcfg, dim, budget, seed, x0, "v23")
    if algo == "seismic_v24_orf":
        return run_seismic(fcfg, dim, budget, seed, x0, "v24")
    if algo == "seismic_v24r1_orf":
        return run_seismic(fcfg, dim, budget, seed, x0, "v24r1")
    if algo == "seismic_v25c_orf":
        return run_seismic(fcfg, dim, budget, seed, x0, "v25c")
    if algo == "cmaes":
        return run_cmaes(fcfg["fn"], x0, bounds, budget, seed=seed).best_val
    if algo == "ipop_cmaes":
        return run_ipop_cmaes(fcfg["fn"], x0, bounds, budget, seed=seed).best_val
    if algo == "pso":
        return run_pso(fcfg["fn"], bounds, budget, seed=seed).best_val
    if algo == "bfgs_ms":
        return run_bfgs_multistart(fcfg["fn"], fcfg["grad"], bounds, budget, seed=seed).best_val
    if algo == "random":
        return run_random_search(fcfg["fn"], bounds, budget, seed=seed).best_val
    raise ValueError(algo)


ALGOS = ("seismic_v20", "seismic_v23_orf", "seismic_v24_orf", "seismic_v24r1_orf", "seismic_v25c_orf",
         "cmaes", "ipop_cmaes", "pso", "bfgs_ms", "random")


def phase1(fnames: List[str], dims: List[int], budget: int, trials: int) -> Dict:
    cells: Dict[str, Dict] = {}
    raw: Dict[str, Dict] = {}
    t0 = time.time()
    for fname in fnames:
        fcfg = SUITE[fname]
        for dim in dims:
            key = f"{fname}_{dim}d"
            scores = {a: [] for a in ALGOS}
            for trial in range(trials):
                seed = trial + 1
                r = get_search_range(fcfg, dim)
                rng = np.random.default_rng(trial * 1000 + dim * 10 + 42)
                x0 = rng.uniform(-r, r, size=dim)
                for algo in ALGOS:
                    try:
                        scores[algo].append(run_algo(algo, fcfg, dim, budget, seed, x0))
                    except Exception:  # missing dep / numerical corner
                        scores[algo].append(float("nan"))
            cells[key] = {a: summarize(scores[a]) for a in ALGOS}
            raw[key] = {a: [float(v) for v in scores[a]] for a in ALGOS}
            line = " | ".join(f"{a}={cells[key][a]['median']:.3g}" for a in ALGOS)
            print(f"[{key:>22s}] {line}  ({time.time()-t0:.0f}s acum.)", flush=True)
    return {"cells": cells, "raw": raw}


def phase2(fnames: List[str], budgets: List[int], trials: int) -> Dict:
    cells: Dict[str, Dict] = {}
    raw: Dict[str, Dict] = {}
    algos_p2 = ("seismic_v20", "seismic_v23_orf", "cmaes", "ipop_cmaes")
    for fname in fnames:
        fcfg = SUITE[fname]
        for budget in budgets:
            key = f"{fname}_5d_{budget}"
            scores = {a: [] for a in algos_p2}
            for trial in range(trials):
                seed = trial + 1
                r = get_search_range(fcfg, 5)
                rng = np.random.default_rng(trial * 1000 + 5 * 10 + 42)
                x0 = rng.uniform(-r, r, size=5)
                for algo in algos_p2:
                    scores[algo].append(run_algo(algo, fcfg, 5, budget, seed, x0))
            cells[key] = {a: summarize(scores[a]) for a in algos_p2}
            raw[key] = {a: [float(v) for v in scores[a]] for a in algos_p2}
            line = " | ".join(f"{a}={cells[key][a]['median']:.3g}" for a in algos_p2)
            print(f"[{key:>22s}] {line}", flush=True)
    return {"cells": cells, "raw": raw}


def main() -> None:
    p = argparse.ArgumentParser(description="Publication-grade benchmark vs seeded baselines")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--functions", type=str, nargs="+", default=sorted(SUITE))
    p.add_argument("--quick", action="store_true", help="2 funcs, 2 dims, 3 trials")
    p.add_argument("--phase1-only", action="store_true", help="skip phase 2 budget crossover")
    args = p.parse_args()
    if args.quick:
        args.functions, args.dims, args.trials = ["rastrigin", "ackley"], [5, 10], 3

    t0 = time.time()
    out: Dict = {"meta": vars(args), "phase1": {}, "phase2": {}}
    p1 = phase1(args.functions, args.dims, args.budget, args.trials)
    out["phase1"] = p1["cells"]
    out["phase1_raw"] = p1["raw"]
    if not args.quick and not args.phase1_only:
        p2 = phase2(["rastrigin", "griewank"], [3000, 10000, 25000], args.trials)
        out["phase2"] = p2["cells"]
        out["phase2_raw"] = p2["raw"]

    # Wilcoxon signed-rank: seismic v23 and v24 vs seeded reference baselines (paired by trial)
    keys = sorted(out["phase1"])
    for proto in ("seismic_v23_orf", "seismic_v24_orf", "seismic_v24r1_orf", "seismic_v25c_orf"):
        tag = proto.replace("seismic_", "").replace("_orf", "")
        for ref in ("ipop_cmaes", "pso", "bfgs_ms"):
            pvals: List[float] = []
            for k in keys:
                a = out["phase1_raw"][k][proto]
                b = out["phase1_raw"][k][ref]
                pv, _ = wilcoxon_paired(a, b)
                pvals.append(pv)
                out["phase1"][k][f"wilcoxon_{tag}_vs_{ref}"] = pv
            adj = holm_adjust(pvals)
            for k, a_adj in zip(keys, adj):
                out["phase1"][k][f"holm_{tag}_vs_{ref}"] = float(a_adj) if a_adj == a_adj else None

    elapsed = time.time() - t0
    out["meta"]["elapsed_seconds"] = elapsed

    resdir = Path("results")
    resdir.mkdir(exist_ok=True)
    with open(resdir / "baselines_v24.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---- Report ----
    lines: List[str] = []
    lines.append("# Informe de benchmark comparativo (publi-canónico, v24)")
    lines.append("")
    lines.append(f"> Trials: {args.trials} | Presupuesto: {args.budget} evals | "
                 f"Baselines sembrados (pycma seed, IPOP con reinicios, PSO Clerc-Kennedy, L-BFGS-B multistart).")
    lines.append(f"> Generado en {elapsed:.0f}s. Nota: los optimizadores sísmicos consumen además 1 gradiente "
                 f"analítico por partícula y paso (declarado; tasa estándar 1:1 como en L-BFGS-B).")
    lines.append("")
    header = "| Celda | " + " | ".join(ALGOS) + " | v24r1 vs IPOP (p Holm) | v25c vs IPOP (p Holm) |"
    lines.append("## Fase 1 — medianas (mejor en negrita)\n")
    lines.append(header)
    lines.append("|---" * (len(ALGOS) + 3) + "|")
    for k in keys:
        meds = {a: out["phase1"][k][a]["median"] for a in ALGOS}
        best_a = min(meds, key=meds.get)
        row = [f"| {k}"]
        for a in ALGOS:
            v = meds[a]
            row.append(f"**{v:.4g}**" if a == best_a else f"{v:.4g}")
        for tag in ("v24r1", "v25c"):
            holm_ipop = out["phase1"][k].get(f"holm_{tag}_vs_ipop_cmaes")
            mark = significance_stars(holm_ipop) if holm_ipop is not None else ""
            row.append(f"{tag}: {holm_ipop:.3g} {mark}" if holm_ipop is not None else "—")
        lines.append(" | ".join(row) + " |")
    lines.append("")
    lines.append("Última columna: p-valor de Wilcoxon pareado (v23 vs IPOP-CMA-ES) con corrección de Holm "
                 "sobre todas las celdas (* p<0.05, ** p<0.01, *** p<0.001). "
                 "Nota: 'v23 vs ipop' no significa ganador; significa diferencia significativa.")
    if out["phase2"]:
        lines.append("\n## Fase 2 — cruce de presupuestos (Rastrigin/Griewank 5D; incluye IPOP-CMA-ES)\n")
        lines.append("| Experimento | " + " | ".join(("seismic_v20", "seismic_v23_orf", "cmaes", "ipop_cmaes")) + " |")
        lines.append("|---|---|---|---|---|")
        for k in sorted(out["phase2"]):
            row = [f"| {k}"]
            for a in ("seismic_v20", "seismic_v23_orf", "cmaes", "ipop_cmaes"):
                row.append(f"{out['phase2'][k][a]['median']:.4g}")
            lines.append(" | ".join(row) + " |")
    rep = Path("docs/audit_2026/informe_baselines_v24.md")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[+] JSON: results/baselines_v24.json")
    print(f"[+] Informe: {rep} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
