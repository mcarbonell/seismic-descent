"""
experiment_lissajous_vs_rff.py — Head-to-Head Comparison: Lissajous Waves vs RFF.

Compares deterministic incommensurate Lissajous wave perturbation against
stochastic Random Fourier Features (RFF) on benchmark optimization problems.

Evaluates:
  1. Seismic (RFF Canonical)
  2. Seismic (Lissajous Direct, O(D) uncoupled)
  3. Seismic (Lissajous Coupled, O(D) toroidal entanglement)
  4. CMA-ES & Simulated Annealing baselines
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

from seismic_descent import ALL_FUNCTIONS, RandomFourierFeatures, LissajousWaveField

try:
    import cma
    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


def run_seismic_engine(
    fn: Callable,
    fn_grad: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    noise_field: Any,
    eval_budget: int = 3000,
    n_particles: int = 10,
    dt_base: float = 0.2,
    dt_floor: float = 0.2,
    noise_amplitude: float = 0.5,
    dt_cycles_multiplier: float = 5.0,
    n_cycles: int = 10,
    seed: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    n_steps = max(20, eval_budget // n_particles)
    dim = len(x0)
    bounds = np.asarray(bounds, dtype=np.float64)
    center = (bounds[:, 1] + bounds[:, 0]) / 2.0
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0

    rng = np.random.default_rng(seed)
    x_norm = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    x_norm[0] = np.clip((x0 - center) / half_range, -1.0, 1.0)

    dt_noise = (n_cycles * np.pi) / n_steps
    t = 0.0

    x_real = center + x_norm * half_range
    real_vals = np.asarray(fn(x_real), dtype=np.float64)
    best_val = float(np.min(real_vals))
    best_per_step = [best_val]

    for step in range(n_steps):
        amp = noise_amplitude * np.sin(t * 2.0)
        x_real = center + x_norm * half_range
        f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
        if f_grad_real.ndim == 1:
            f_grad_real = f_grad_real.reshape(1, -1)

        # 1. Chain rule
        f_grad_mapped = f_grad_real * half_range

        # 2. L2 Gradient Normalization
        norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
        f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)

        # 3. Noise Gradient (either RFF or Lissajous)
        noise_grad = noise_field.grad(x_norm, t, amplitude=amp)
        grad = f_grad_dir + noise_grad

        # 4. Decoupled cyclic dt with floor
        cyclic_scale = dt_floor + (1.0 - dt_floor) * np.abs(np.sin(t * dt_cycles_multiplier))
        current_dt = dt_base * cyclic_scale

        x_norm -= current_dt * grad
        np.clip(x_norm, -1.0, 1.0, out=x_norm)
        t += dt_noise

        x_real = center + x_norm * half_range
        step_vals = np.asarray(fn(x_real), dtype=np.float64)
        min_step = float(np.min(step_vals))
        if min_step < best_val:
            best_val = min_step
        best_per_step.append(best_val)

    evals = np.arange(1, len(best_per_step) + 1) * n_particles
    return best_val, evals, np.array(best_per_step)


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


def run_sa(fn, x0, bounds, eval_budget: int, step_sample: int = 20) -> Tuple[float, np.ndarray, np.ndarray]:
    x = np.array(x0, dtype=float)
    current_val = float(fn(x))
    best_val = current_val
    t = 10.0
    cooling = 0.999
    step_size = float(np.mean(bounds[:, 1] - bounds[:, 0]) * 0.05)

    evals_rec = [1]
    best_rec = [best_val]

    for step in range(2, eval_budget + 1):
        noise = np.random.normal(0, step_size, size=len(x))
        x_new = np.clip(x + noise, bounds[:, 0], bounds[:, 1])
        new_val = float(fn(x_new))
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            x, current_val = x_new, new_val
        if current_val < best_val:
            best_val = current_val

        if step % step_sample == 0 or step == eval_budget:
            evals_rec.append(step)
            best_rec.append(best_val)
        t *= cooling

    return float(best_val), np.array(evals_rec), np.array(best_rec)


def main():
    parser = argparse.ArgumentParser(description="Lissajous vs RFF Benchmark")
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
    print(f" LISSAJOUS WAVES VS RANDOM FOURIER FEATURES (RFF) BENCHMARK")
    print(f" Functions: {args.functions} | Dims: {args.dims} | Budget: {args.budget} | Trials: {args.trials}")
    print(f"==========================================================================\n")

    results = {
        "meta": {"trials": args.trials, "budget": args.budget, "dims": args.dims, "eval_grid": eval_grid.tolist()},
        "tests": {},
    }

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
                "Seismic (RFF Canonical)": {
                    "field_builder": lambda s: RandomFourierFeatures(dim=dims, r=64, seed=s),
                    "color": "#10b981",  # Emerald
                    "style": "-",
                },
                "Seismic (Lissajous Direct)": {
                    "field_builder": lambda s: LissajousWaveField(dim=dims, coupled=False, base_lengthscale=0.4),
                    "color": "#f97316",  # Orange
                    "style": "--",
                },
                "Seismic (Lissajous Coupled)": {
                    "field_builder": lambda s: LissajousWaveField(dim=dims, coupled=True, base_lengthscale=0.4),
                    "color": "#8b5cf6",  # Purple
                    "style": "-",
                },
                "CMA-ES": {
                    "runner": lambda x0, s: run_cmaes(fn, x0, bounds, args.budget),
                    "color": "#3b82f6",  # Blue
                    "style": ":",
                },
                "Simulated Annealing": {
                    "runner": lambda x0, s: run_sa(fn, x0, bounds, args.budget),
                    "color": "#ef4444",  # Red
                    "style": "-.",
                },
            }

            print(f"--- {fconfig['name']} ({dims}D) ---")
            print(f"| {'Algorithm':<28} | {'Median Error':<16} | {'Best Error':<16} | {'Runtime (s)':<12} |")
            print("|" + "-"*30 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*14 + "|")

            for aname, acfg in algos.items():
                scores = []
                grid_trajs = []
                t0 = time.time()

                for trial in range(args.trials):
                    rng = np.random.default_rng(trial * 1000 + dims * 10 + 42)
                    x0 = rng.uniform(-search_range, search_range, size=dims)

                    if "field_builder" in acfg:
                        noise_field = acfg["field_builder"](trial + 1)
                        bval, evals_raw, vals_raw = run_seismic_engine(
                            fn, fn_grad, x0, bounds, noise_field,
                            eval_budget=args.budget, seed=trial + 1
                        )
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

                print(f"| {aname:<28} | {np.median(scores):<16.4e} | {np.min(scores):<16.4e} | {elapsed:<12.2f} |")
            print()

    # Save JSON raw data
    with open("results/lissajous_vs_rff.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot results
    n_funcs = len(args.functions)
    n_dims = len(args.dims)
    fig, axes = plt.subplots(n_funcs, n_dims, figsize=(6.5 * n_dims, 4.5 * n_funcs), sharex=True, squeeze=False)

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
                lw = 2.4 if "Lissajous" in aname else (2.0 if "RFF" in aname else 1.4)
                ax.plot(eval_grid, med, label=aname, color=c, linestyle=ls, linewidth=lw)
                ax.fill_between(eval_grid, q25, q75, color=c, alpha=0.10)

            ax.set_yscale("log")
            ax.set_title(f"{fname} ({dims}D)", fontsize=12, fontweight="bold")
            ax.grid(True, which="both", alpha=0.25, linestyle="--")

            if c_idx == 0:
                ax.set_ylabel("Best f(x) (Log Scale)", fontsize=10)
            if r_idx == n_funcs - 1:
                ax.set_xlabel("Evaluations", fontsize=10)
            if r_idx == 0 and c_idx == n_dims - 1:
                ax.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8, loc="upper right")

    plt.suptitle("Lissajous Wave Fields vs Random Fourier Features (RFF) vs Baselines", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plot_path = "results/lissajous_vs_rff.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Comparative plot saved to: {plot_path}")

    # Generate Markdown report
    md_path = "results/lissajous_vs_rff_report.md"
    md = []
    md.append("# Estudio Comparativo: Ondas de Lissajous vs Random Fourier Features (RFF)")
    md.append("")
    md.append(f"> **Configuración:** {args.trials} repeticiones independientes por prueba | Presupuesto: {args.budget} evaluaciones.")
    md.append("")
    md.append("![Lissajous vs RFF](lissajous_vs_rff.png)")
    md.append("")
    md.append("## 1. Tablas Comparativas por Función")
    md.append("")

    for tkey, tdata in results["tests"].items():
        fname = tdata["function"]
        dims = tdata["dims"]
        md.append(f"### {fname} — {dims}D")
        md.append("")
        md.append("| Algoritmo / Terreno | Mediana Final | Mejor de 15 | Tiempo (s) |")
        md.append("|:---|:---:|:---:|:---:|")

        for aname, adata in tdata["algorithms"].items():
            is_lissa = "Lissajous" in aname
            bold = "**" if is_lissa else ""
            md.append(f"| {bold}{aname}{bold} | {adata['median']:.4e} | {adata['best']:.4e} | {adata['runtime']:.2f}s |")
        md.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[+] Report saved to: {md_path}")


if __name__ == "__main__":
    main()
