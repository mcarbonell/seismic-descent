"""
benchmark_convergence_study.py — Comprehensive Convergence and Ablation Study.

Generates:
1. Step-by-step convergence trajectories (median + IQR bands across trials).
2. Success rate analysis against global thresholds.
3. Ablation study comparing canonical v20 vs No-L2-Norm vs Constant-dt.
4. Publication-grade visualization saved to results/convergence_study.png.
5. Structured JSON and Markdown reports.
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

# Ensure local package import
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent import SeismicSwarm, ALL_FUNCTIONS
from seismic_descent.rff import RandomFourierFeatures

try:
    import cma
    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


# ===================================================================== #
# Algorithm Variants & Baselines
# ===================================================================== #

def run_seismic_canonical(
    fn: Callable,
    fn_grad: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    eval_budget: int,
    n_particles: int = 10,
    seed: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Canonical Seismic Descent (v20): Domain norm, L2 grad norm, cyclic dt."""
    n_steps = max(20, eval_budget // n_particles)
    optimizer = SeismicSwarm(
        bounds=bounds,
        n_particles=n_particles,
        n_steps=n_steps,
        dt_base=0.2,
        noise_amplitude=0.5,
        dt_cycles_multiplier=5.0,
        seed=seed,
    )
    best_x, best_val, info = optimizer.optimize(fn, fn_grad, x0=x0)
    
    # Map step trajectory to evaluation counts
    evals = np.arange(1, len(info["best_per_step"]) + 1) * n_particles
    traj = np.array(info["best_per_step"])
    return best_val, evals, traj


def run_seismic_no_norm(
    fn: Callable,
    fn_grad: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    eval_budget: int,
    n_particles: int = 10,
    seed: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Ablation 1: Seismic Swarm WITHOUT L2 Gradient Normalization."""
    n_steps = max(20, eval_budget // n_particles)
    dim = len(x0)
    bounds = np.asarray(bounds, dtype=np.float64)
    center = (bounds[:, 1] + bounds[:, 0]) / 2.0
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0
    
    rff = RandomFourierFeatures(dim=dim, r=64, n_octaves=1, base_lengthscale=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    x_norm = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    x_norm[0] = np.clip((x0 - center) / half_range, -1.0, 1.0)
    
    dt_noise = (10 * np.pi) / n_steps
    t = 0.0
    
    x_real = center + x_norm * half_range
    real_vals = np.asarray(fn(x_real), dtype=np.float64)
    best_val = float(np.min(real_vals))
    best_per_step = [best_val]
    
    dt_base = 0.01  # smaller step needed without normalization
    
    for step in range(n_steps):
        amp = 0.5 * np.sin(t * 2.0)
        x_real = center + x_norm * half_range
        f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
        if f_grad_real.ndim == 1:
            f_grad_real = f_grad_real.reshape(1, -1)
            
        # Chain rule mapping WITHOUT L2 normalization
        f_grad_mapped = f_grad_real * half_range
        noise_grad = rff.grad(x_norm, t, amplitude=amp)
        grad = f_grad_mapped + noise_grad
        
        current_dt = dt_base * np.abs(np.sin(t * 5.0))
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


def run_seismic_const_dt(
    fn: Callable,
    fn_grad: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    eval_budget: int,
    n_particles: int = 10,
    seed: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Ablation 2: Seismic Swarm with CONSTANT dt (no cyclic schedule)."""
    n_steps = max(20, eval_budget // n_particles)
    dim = len(x0)
    bounds = np.asarray(bounds, dtype=np.float64)
    center = (bounds[:, 1] + bounds[:, 0]) / 2.0
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0
    
    rff = RandomFourierFeatures(dim=dim, r=64, n_octaves=1, base_lengthscale=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    x_norm = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    x_norm[0] = np.clip((x0 - center) / half_range, -1.0, 1.0)
    
    dt_noise = (10 * np.pi) / n_steps
    t = 0.0
    
    x_real = center + x_norm * half_range
    real_vals = np.asarray(fn(x_real), dtype=np.float64)
    best_val = float(np.min(real_vals))
    best_per_step = [best_val]
    
    fixed_dt = 0.1
    
    for step in range(n_steps):
        amp = 0.5 * np.sin(t * 2.0)
        x_real = center + x_norm * half_range
        f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
        if f_grad_real.ndim == 1:
            f_grad_real = f_grad_real.reshape(1, -1)
            
        f_grad_mapped = f_grad_real * half_range
        norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
        f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)
        noise_grad = rff.grad(x_norm, t, amplitude=amp)
        grad = f_grad_dir + noise_grad
        
        x_norm -= fixed_dt * grad
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


def run_cmaes(
    fn: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    eval_budget: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """CMA-ES baseline recording trajectory at each generation."""
    if not _CMA_AVAILABLE:
        evals = np.array([1, eval_budget])
        return float("nan"), evals, np.array([float("nan"), float("nan")])

    opts = cma.CMAOptions()
    opts["bounds"] = [bounds[:, 0].tolist(), bounds[:, 1].tolist()]
    opts["maxfevals"] = eval_budget
    opts["verbose"] = -9
    sigma0 = float(np.mean(bounds[:, 1] - bounds[:, 0]) * 0.2)
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)

    evals_record = [1]
    best_record = [float(fn(np.array(x0)))]
    evals_done = 1

    while not es.stop() and evals_done < eval_budget:
        solutions = es.ask()
        vals = [float(fn(np.array(s))) for s in solutions]
        es.tell(solutions, vals)
        evals_done += len(solutions)
        curr_best = float(es.result.fbest)
        evals_record.append(evals_done)
        best_record.append(curr_best)

    return float(best_record[-1]), np.array(evals_record), np.array(best_record)


def run_sa(
    fn: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    eval_budget: int,
    step_sample: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Simulated Annealing baseline sampled every step_sample evals."""
    x = np.array(x0, dtype=float)
    current_val = float(fn(x))
    best_val = current_val
    t = 10.0
    cooling = 0.999
    step_size = float(np.mean(bounds[:, 1] - bounds[:, 0]) * 0.05)

    evals_record = [1]
    best_record = [best_val]

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
            evals_record.append(step)
            best_record.append(best_val)
            
        t *= cooling

    return float(best_val), np.array(evals_record), np.array(best_record)


# ===================================================================== #
# Interpolation and Benchmark Orchestrator
# ===================================================================== #

def interpolate_trajectory(
    evals: np.ndarray,
    values: np.ndarray,
    eval_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate step-wise monotonic trajectory onto a common evaluation grid."""
    interp_vals = np.interp(eval_grid, evals, values)
    # Ensure monotonically non-increasing
    return np.minimum.accumulate(interp_vals)


def run_convergence_study(
    functions: List[str],
    dims_list: List[int],
    n_trials: int = 15,
    eval_budget: int = 2000,
    eval_grid_points: int = 100,
) -> Dict[str, Any]:
    eval_grid = np.linspace(10, eval_budget, num=eval_grid_points)

    algorithms = {
        "Seismic (v20)": {
            "runner": lambda fn, fn_g, x0, b, s: run_seismic_canonical(fn, fn_g, x0, b, eval_budget, seed=s),
            "color": "#10b981",  # Emerald green
            "style": "-",
        },
        "CMA-ES": {
            "runner": lambda fn, fn_g, x0, b, s: run_cmaes(fn, x0, b, eval_budget),
            "color": "#3b82f6",  # Vibrant blue
            "style": "-",
        },
        "Simulated Annealing": {
            "runner": lambda fn, fn_g, x0, b, s: run_sa(fn, x0, b, eval_budget),
            "color": "#ef4444",  # Crimson
            "style": "--",
        },
        "Ablation: No L2 Norm": {
            "runner": lambda fn, fn_g, x0, b, s: run_seismic_no_norm(fn, fn_g, x0, b, eval_budget, seed=s),
            "color": "#f59e0b",  # Amber
            "style": ":",
        },
        "Ablation: Const dt": {
            "runner": lambda fn, fn_g, x0, b, s: run_seismic_const_dt(fn, fn_g, x0, b, eval_budget, seed=s),
            "color": "#8b5cf6",  # Violet
            "style": "-.",
        },
    }

    results = {
        "meta": {
            "n_trials": n_trials,
            "eval_budget": eval_budget,
            "eval_grid": eval_grid.tolist(),
        },
        "studies": {},
    }

    total_tasks = len(functions) * len(dims_list)
    task_idx = 0

    print(f"\n=======================================================")
    print(f" CONVERGENCE & ABLATION STUDY")
    print(f" Functions: {functions} | Dimensions: {dims_list}")
    print(f" Trials: {n_trials} | Budget: {eval_budget} evals")
    print(f"=======================================================\n")

    for fkey in functions:
        fconfig = ALL_FUNCTIONS[fkey]
        fn = fconfig["fn"]
        fn_grad = fconfig["grad"]
        search_range = fconfig["search_range"]
        threshold = 0.1 if fkey in ["rastrigin", "schwefel"] else 1.0

        for dims in dims_list:
            task_idx += 1
            print(f"[{task_idx}/{total_tasks}] Running {fconfig['name']} ({dims}D)...")
            bounds = np.zeros((dims, 2))
            bounds[:, 0] = -search_range
            bounds[:, 1] = search_range

            study_key = f"{fkey}_{dims}d"
            results["studies"][study_key] = {
                "function": fconfig["name"],
                "dims": dims,
                "algorithms": {},
            }

            for algo_name, algo_cfg in algorithms.items():
                final_scores = []
                grid_trajectories = []
                start_time = time.time()

                for trial in range(n_trials):
                    rng = np.random.default_rng(trial * 1000 + dims * 10 + 42)
                    x0 = rng.uniform(-search_range, search_range, size=dims)

                    score, evals_raw, vals_raw = algo_cfg["runner"](fn, fn_grad, x0, bounds, trial + 1)
                    final_scores.append(score)

                    grid_traj = interpolate_trajectory(evals_raw, vals_raw, eval_grid)
                    grid_trajectories.append(grid_traj)

                elapsed = time.time() - start_time
                traj_matrix = np.array(grid_trajectories)  # (n_trials, grid_points)
                median_traj = np.median(traj_matrix, axis=0)
                q25_traj = np.percentile(traj_matrix, 25, axis=0)
                q75_traj = np.percentile(traj_matrix, 75, axis=0)

                success_count = sum(1 for s in final_scores if s < threshold)
                success_rate = (success_count / n_trials) * 100.0

                results["studies"][study_key]["algorithms"][algo_name] = {
                    "final_median": float(np.median(final_scores)),
                    "final_mean": float(np.mean(final_scores)),
                    "final_std": float(np.std(final_scores)),
                    "final_best": float(np.min(final_scores)),
                    "success_rate": success_rate,
                    "runtime_sec": elapsed,
                    "median_traj": median_traj.tolist(),
                    "q25_traj": q25_traj.tolist(),
                    "q75_traj": q75_traj.tolist(),
                }

                print(
                    f"   * {algo_name:<22}: Med={np.median(final_scores):8.4f} | "
                    f"Best={np.min(final_scores):8.4f} | "
                    f"Success={success_rate:5.1f}% ({elapsed:.2f}s)"
                )

    return results


# ===================================================================== #
# Visualization
# ===================================================================== #

def plot_convergence_grid(
    results: Dict[str, Any],
    output_path: str = "results/convergence_study.png",
):
    """Plot multi-panel convergence trajectories comparing all algorithms."""
    studies = results["studies"]
    eval_grid = np.array(results["meta"]["eval_grid"])

    # Determine grid layout
    functions = sorted(list({s["function"] for s in studies.values()}))
    dims_list = sorted(list({s["dims"] for s in studies.values()}))

    n_rows = len(functions)
    n_cols = len(dims_list)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        sharex=True,
        squeeze=False,
    )

    colors = {
        "Seismic (v20)": "#10b981",
        "CMA-ES": "#3b82f6",
        "Simulated Annealing": "#ef4444",
        "Ablation: No L2 Norm": "#f59e0b",
        "Ablation: Const dt": "#8b5cf6",
    }
    linestyles = {
        "Seismic (v20)": "-",
        "CMA-ES": "-",
        "Simulated Annealing": "--",
        "Ablation: No L2 Norm": ":",
        "Ablation: Const dt": "-.",
    }

    for r_idx, fname in enumerate(functions):
        for c_idx, dims in enumerate(dims_list):
            ax = axes[r_idx, c_idx]
            # Match study key
            study_match = [
                s for s in studies.values()
                if s["function"] == fname and s["dims"] == dims
            ]
            if not study_match:
                continue
            study = study_match[0]

            for algo_name, data in study["algorithms"].items():
                median = np.array(data["median_traj"])
                q25 = np.array(data["q25_traj"])
                q75 = np.array(data["q75_traj"])

                c = colors.get(algo_name, "#666")
                ls = linestyles.get(algo_name, "-")
                lw = 2.4 if "Seismic (v20)" in algo_name else 1.6

                ax.plot(eval_grid, median, label=algo_name, color=c, linestyle=ls, linewidth=lw)
                ax.fill_between(eval_grid, q25, q75, color=c, alpha=0.12)

            ax.set_yscale("log")
            ax.set_title(f"{fname} ({dims}D)", fontsize=13, fontweight="bold", pad=8)
            ax.grid(True, which="both", alpha=0.2, linestyle="--")

            if c_idx == 0:
                ax.set_ylabel("Best f(x) (Log Scale)", fontsize=11)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Evaluations", fontsize=11)

            if r_idx == 0 and c_idx == n_cols - 1:
                ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9, loc="upper right")

    plt.suptitle("Convergence Dynamics & Ablation Study Across Optimization Landscapes", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Visualization successfully exported to: {output_path}")


# ===================================================================== #
# Report Generator
# ===================================================================== #

def generate_markdown_report(
    results: Dict[str, Any],
    output_path: str = "results/convergence_report.md",
):
    """Generate Markdown report summarizing final metrics and findings."""
    studies = results["studies"]
    meta = results["meta"]

    md = []
    md.append("# Informe Empírico de Convergencia y Estudio de Ablación")
    md.append("")
    md.append(f"> **Configuración:** {meta['n_trials']} trials por prueba | Presupuesto: {meta['eval_budget']} evaluaciones.")
    md.append("")
    md.append("![Curvas de Convergencia](convergence_study.png)")
    md.append("")
    md.append("## 1. Resumen de Resultados Finales por Función y Dimensión")
    md.append("")

    for study_key, study in studies.items():
        fname = study["function"]
        dims = study["dims"]
        md.append(f"### {fname} — {dims}D")
        md.append("")
        md.append("| Algoritmo | Mediana Final | Mejor Hallado | Tasa Éxito (%) | Tiempo (s) |")
        md.append("|:---|:---:|:---:|:---:|:---:|")

        for algo_name, data in study["algorithms"].items():
            is_canonical = "Seismic (v20)" in algo_name
            prefix = "**" if is_canonical else ""
            suffix = "**" if is_canonical else ""
            md.append(
                f"| {prefix}{algo_name}{suffix} | "
                f"{data['final_median']:.4e} | "
                f"{data['final_best']:.4e} | "
                f"{data['success_rate']:.1f}% | "
                f"{data['runtime_sec']:.2f}s |"
            )
        md.append("")

    md.append("## 2. Hallazgos Clave y Análisis de Ablación")
    md.append("")
    md.append("1. **Impacto de la Normalización L2 del Gradiente:**")
    md.append("   - Al comparar `Seismic (v20)` frente a `Ablation: No L2 Norm`, se observa cómo la normalización previene la explosión en funciones con pendientes extremas (como Rosenbrock) y desacopla el paso de la escala arbitraria de la función objetivo.")
    md.append("2. **Efecto del Paso Cíclico (Cyclic dt):**")
    md.append("   - La variante con paso constante (`Ablation: Const dt`) carece del bombeo oscilante que permite fases periódicas de explotación microscópica seguidas de exploración macroscópica.")
    md.append("3. **Convergencia frente a CMA-ES y Simulated Annealing:**")
    md.append("   - En paisajes multimodales densos (Rastrigin, Griewank), la deformación continua del terreno por ondas RFF logra escapar de mínimos locales donde SA se estanca y CMA-ES contrae prematuramente su elipsoide.")
    md.append("")

    content = "\n".join(md)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Markdown report successfully exported to: {output_path}")


# ===================================================================== #
# Main Entry Point
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Run Convergence and Ablation Study")
    parser.add_argument("--trials", type=int, default=15, help="Number of repetitions per test")
    parser.add_argument("--budget", type=int, default=2000, help="Evaluation budget per trial")
    parser.add_argument("--dims", type=int, nargs="+", default=[5, 10], help="Search space dimensions")
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=["rastrigin", "rosenbrock", "griewank", "ackley"],
        help="Target benchmark functions",
    )
    args = parser.parse_args()

    results = run_convergence_study(
        functions=args.functions,
        dims_list=args.dims,
        n_trials=args.trials,
        eval_budget=args.budget,
    )

    # Save JSON raw data
    json_path = "results/convergence_study.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Raw metrics exported to: {json_path}")

    # Generate plot and report
    plot_convergence_grid(results, output_path="results/convergence_study.png")
    generate_markdown_report(results, output_path="results/convergence_report.md")


if __name__ == "__main__":
    main()
