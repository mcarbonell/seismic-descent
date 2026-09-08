"""
experiment_swarm_gravity.py — Empirical Evaluation of Phase-Modulated Swarm Gravitational Coupling.

Evaluates:
  1. Seismic Canonical (No Gravity, RFF)
  2. Seismic Gravity (Phase-Modulated, gamma=0.3, RFF)
  3. Seismic Gravity (Phase-Modulated, gamma=0.6, RFF)
  4. Seismic Gravity (Constant Gravity, gamma=0.3, RFF)
  5. Seismic Gravity + Ortho-Lissajous (gamma=0.4)
  6. CMA-ES Baseline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent import ALL_FUNCTIONS
from seismic_descent.swarm_gravity import SeismicSwarmGravity

try:
    import cma
    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


def run_cmaes(fn, x0, bounds, eval_budget: int) -> Tuple[float, np.ndarray, np.ndarray]:
    if not _CMA_AVAILABLE:
        return float("nan"), np.array([1]), np.array([float("nan")])
    opts = cma.CMAOptions()
    opts["bounds"] = [bounds[:, 0].tolist(), bounds[:, 1].tolist()]
    opts["maxfevals"] = eval_budget
    opts["verbose"] = -9
    sigma0 = float(np.mean(bounds[:, 1] - bounds[:, 0]) * 0.2)
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)

    evals_done = 1
    evals_rec = [1]
    best_rec = [float(fn(np.array(x0)))]

    while not es.stop() and evals_done < eval_budget:
        solutions = es.ask()
        vals = [float(fn(np.array(s))) for s in solutions]
        es.tell(solutions, vals)
        evals_done += len(solutions)
        evals_rec.append(evals_done)
        best_rec.append(float(es.result.fbest))

    return float(best_rec[-1]), np.array(evals_rec), np.array(best_rec)


def main():
    parser = argparse.ArgumentParser(description="Swarm Gravitational Coupling Benchmark")
    parser.add_argument("--trials", type=int, default=15, help="Repetitions per test")
    parser.add_argument("--budget", type=int, default=3000, help="Evaluation budget")
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10], help="Dimensions")
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=["rastrigin", "rosenbrock", "griewank", "ackley"],
        help="Functions to test",
    )
    args = parser.parse_args()

    eval_grid = np.linspace(10, args.budget, num=100)

    print(f"\n==========================================================================")
    print(f" PHASE-MODULATED SWARM GRAVITY COUPLING BENCHMARK")
    print(f" Functions: {args.functions} | Dims: {args.dims} | Budget: {args.budget} | Trials: {args.trials}")
    print(f"==========================================================================\n")

    results = {
        "meta": {"trials": args.trials, "budget": args.budget, "dims": args.dims, "eval_grid": eval_grid.tolist()},
        "tests": {},
    }

    n_particles = 10
    n_steps = max(20, args.budget // n_particles)

    for fkey in args.functions:
        fconfig = ALL_FUNCTIONS[fkey]
        fn = fconfig["fn"]
        fn_grad = fconfig["grad"]
        search_range = fconfig["search_range"]

        for dims in args.dims:
            test_key = f"{fkey}_{dims}d"
            results["tests"][test_key] = {"function": fconfig["name"], "dims": dims, "algorithms": {}}

            bounds = np.zeros((dims, 2))
            bounds[:, 0] = -search_range
            bounds[:, 1] = search_range

            algos = {
                "Seismic Canonical (No Gravity)": {
                    "builder": lambda s: SeismicSwarmGravity(
                        bounds=bounds, n_particles=n_particles, n_steps=n_steps,
                        gravity_strength=0.0, gravity_mode="none", noise_type="rff", seed=s
                    ),
                    "color": "#64748b",  # Slate
                    "style": ":",
                },
                "Phase-Gravity (gamma=0.3)": {
                    "builder": lambda s: SeismicSwarmGravity(
                        bounds=bounds, n_particles=n_particles, n_steps=n_steps,
                        gravity_strength=0.3, gravity_mode="phase_modulated", noise_type="rff", seed=s
                    ),
                    "color": "#0ea5e9",  # Ocean Blue
                    "style": "--",
                },
                "Phase-Gravity (gamma=0.6)": {
                    "builder": lambda s: SeismicSwarmGravity(
                        bounds=bounds, n_particles=n_particles, n_steps=n_steps,
                        gravity_strength=0.6, gravity_mode="phase_modulated", noise_type="rff", seed=s
                    ),
                    "color": "#10b981",  # Emerald
                    "style": "-",
                },
                "Constant-Gravity (gamma=0.3)": {
                    "builder": lambda s: SeismicSwarmGravity(
                        bounds=bounds, n_particles=n_particles, n_steps=n_steps,
                        gravity_strength=0.3, gravity_mode="constant", noise_type="rff", seed=s
                    ),
                    "color": "#f59e0b",  # Amber
                    "style": "-.",
                },
                "Phase-Grav + Ortho-Lissajous": {
                    "builder": lambda s: SeismicSwarmGravity(
                        bounds=bounds, n_particles=n_particles, n_steps=n_steps,
                        gravity_strength=0.4, gravity_mode="phase_modulated",
                        noise_type="orthogonal_lissajous", seed=s
                    ),
                    "color": "#8b5cf6",  # Violet
                    "style": "-",
                },
                "CMA-ES": {
                    "runner": lambda x0, s: run_cmaes(fn, x0, bounds, args.budget),
                    "color": "#ef4444",  # Red
                    "style": ":",
                },
            }

            print(f"--- {fconfig['name']} ({dims}D) ---")
            print(f"| {'Algorithm':<32} | {'Median Error':<16} | {'Best Error':<16} | {'Runtime (s)':<12} |")
            print("|" + "-"*34 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*14 + "|")

            for aname, acfg in algos.items():
                scores = []
                grid_trajs = []
                t0 = time.time()

                for trial in range(args.trials):
                    rng = np.random.default_rng(trial * 1000 + dims * 10 + 42)
                    x0 = rng.uniform(-search_range, search_range, size=dims)

                    if "builder" in acfg:
                        optimizer = acfg["builder"](trial + 1)
                        best_x, bval, info = optimizer.optimize(fn=fn, fn_grad=fn_grad, x0=x0)
                        vals_raw = np.array(info["best_per_step"])
                        evals_raw = np.arange(1, len(vals_raw) + 1) * n_particles
                    else:
                        bval, evals_raw, vals_raw = acfg["runner"](x0, trial + 1)

                    scores.append(bval)
                    interp_traj = np.interp(eval_grid, evals_raw, vals_raw)
                    grid_trajs.append(np.minimum.accumulate(interp_traj))

                elapsed = time.time() - t0
                traj_mat = np.array(grid_trajs)
                med_traj = np.median(traj_mat, axis=0)
                q25_traj = np.percentile(traj_mat, 25, axis=0)
                q75_traj = np.percentile(traj_mat, 75, axis=0)

                results["tests"][test_key]["algorithms"][aname] = {
                    "median": float(np.median(scores)),
                    "mean": float(np.mean(scores)),
                    "best": float(np.min(scores)),
                    "runtime": elapsed,
                    "median_traj": med_traj.tolist(),
                    "q25_traj": q25_traj.tolist(),
                    "q75_traj": q75_traj.tolist(),
                    "color": acfg["color"],
                    "style": acfg["style"],
                }

                print(f"| {aname:<32} | {np.median(scores):<16.4e} | {np.min(scores):<16.4e} | {elapsed:<12.2f} |")
            print()

    # Save JSON raw data
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "swarm_gravity.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot results
    n_funcs = len(args.functions)
    n_dims = len(args.dims)
    fig, axes = plt.subplots(n_funcs, n_dims, figsize=(7.0 * n_dims, 4.5 * n_funcs), sharex=True, squeeze=False)

    for r_idx, fkey in enumerate(args.functions):
        fname = ALL_FUNCTIONS[fkey]["name"]
        for c_idx, dims in enumerate(args.dims):
            ax = axes[r_idx, c_idx]
            tkey = f"{fkey}_{dims}d"
            test_data = results["tests"][tkey]["algorithms"]

            for aname, adata in test_data.items():
                med = np.array(adata["median_traj"])
                q25 = np.array(adata["q25_traj"])
                q75 = np.array(adata["q75_traj"])
                c = adata["color"]
                ls = adata["style"]
                lw = 2.4 if ("Phase-Gravity (gamma=0.6)" in aname or "Ortho-Lissajous" in aname) else 1.5
                ax.plot(eval_grid, med, label=aname, color=c, linestyle=ls, linewidth=lw)
                if "Phase-Gravity (gamma=0.6)" in aname or "Ortho-Lissajous" in aname:
                    ax.fill_between(eval_grid, q25, q75, color=c, alpha=0.12)

            ax.set_yscale("log")
            ax.set_title(f"{fname} ({dims}D)", fontsize=12, fontweight="bold")
            ax.grid(True, which="both", alpha=0.25, linestyle="--")

            if c_idx == 0:
                ax.set_ylabel("Best f(x) (Log Scale)", fontsize=10)
            if r_idx == n_funcs - 1:
                ax.set_xlabel("Evaluations", fontsize=10)
            if r_idx == 0 and c_idx == n_dims - 1:
                ax.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=7.0, loc="upper right")

    plt.suptitle("Swarm Gravitational Coupling & Cohesion Benchmark", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plot_path = results_dir / "swarm_gravity.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparative plot saved to: {plot_path}")

    # Generate Markdown report
    md_path = results_dir / "swarm_gravity_report.md"
    md = []
    md.append("# Estudio Comparativo: Acoplamiento Gravitacional del Enjambre (Swarm Gravity)")
    md.append("")
    md.append(f"> **Configuración:** {args.trials} repeticiones independientes por prueba | Presupuesto: {args.budget} evaluaciones.")
    md.append("")
    md.append("![Swarm Gravity](swarm_gravity.png)")
    md.append("")
    md.append("## 1. Tablas Comparativas por Función")
    md.append("")

    for tkey, tdata in results["tests"].items():
        fname = tdata["function"]
        dims = tdata["dims"]
        md.append(f"### {fname} — {dims}D")
        md.append("")
        md.append("| Algoritmo / Variante | Mediana Final | Mejor de 15 | Tiempo (s) |")
        md.append("|:---|:---:|:---:|:---:|")

        for aname, adata in tdata["algorithms"].items():
            is_grav = "Gravity" in aname or "Grav" in aname
            bold = "**" if is_grav else ""
            md.append(f"| {bold}{aname}{bold} | {adata['median']:.4e} | {adata['best']:.4e} | {adata['runtime']:.2f}s |")
        md.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[+] Report saved to: {md_path}")


if __name__ == "__main__":
    main()
