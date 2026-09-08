"""
benchmark_suite.py — Multi-function benchmark runner for Seismic Descent.

Compares Seismic Descent against Simulated Annealing (SA) and CMA-ES across
standard benchmarks (Rastrigin, Schwefel, Ackley, Griewank, Rosenbrock).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# Add src to path if package is not yet installed in environment
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent import seismic_swarm, ALL_FUNCTIONS


def sa_generic(
    fn,
    x0: np.ndarray,
    n_steps: int = 5000,
    t0: float = 10.0,
    cooling: float = 0.999,
    step_size: float = 0.3,
    bounds: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Simulated Annealing baseline."""
    x = np.array(x0, dtype=float)
    current_val = float(fn(x))
    best_x = x.copy()
    best_val = current_val
    t = t0

    for _ in range(n_steps):
        noise = np.random.normal(0, step_size, size=len(x))
        x_new = np.clip(x + noise, bounds[:, 0], bounds[:, 1])
        new_val = float(fn(x_new))
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            x, current_val = x_new, new_val
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        t *= cooling

    return best_x, best_val


def cmaes_run(
    fn,
    x0: np.ndarray,
    eval_budget: int,
    bounds: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """CMA-ES baseline (requires `cma` package)."""
    try:
        import cma
    except ImportError:
        return x0, float("nan")

    opts = cma.CMAOptions()
    opts["bounds"] = [bounds[:, 0].tolist(), bounds[:, 1].tolist()]
    opts["maxfevals"] = eval_budget
    opts["verbose"] = -9
    sigma0 = float(np.mean(bounds[:, 1] - bounds[:, 0]) * 0.2)
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [float(fn(np.array(s))) for s in solutions])
    return es.result.xbest, es.result.fbest


def run_benchmark(
    func_key: str,
    dims: int = 5,
    n_trials: int = 5,
    eval_budget: int = 3000,
) -> Dict[str, Any]:
    func_config = ALL_FUNCTIONS[func_key]
    fn = func_config["fn"]
    fn_grad = func_config["grad"]
    search_range = func_config["search_range"]

    bounds = np.zeros((dims, 2))
    bounds[:, 0] = -search_range
    bounds[:, 1] = search_range

    n_particles = 10
    n_steps = max(100, eval_budget // n_particles)

    seismic_scores = []
    sa_scores = []
    cma_scores = []

    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 100 + 42)
        x0 = rng.uniform(-search_range, search_range, size=dims)

        # 1. Seismic Descent (Canonical)
        _, s_score, _ = seismic_swarm(
            fn=fn,
            fn_grad=fn_grad,
            x0_real=x0,
            bounds=bounds,
            n_steps=n_steps,
            n_particles=n_particles,
            dt_base=0.2,
            noise_amplitude=0.5,
            noise_decay=1.0,
            n_cycles=10,
            dt_cycles_multiplier=5.0,
            seed=trial + 1,
        )
        seismic_scores.append(s_score)

        # 2. Simulated Annealing
        _, sa_score = sa_generic(
            fn=fn,
            x0=x0,
            n_steps=eval_budget,
            bounds=bounds,
        )
        sa_scores.append(sa_score)

        # 3. CMA-ES
        _, cma_score = cmaes_run(
            fn=fn,
            x0=x0,
            eval_budget=eval_budget,
            bounds=bounds,
        )
        cma_scores.append(cma_score)

    return {
        "function": func_config["name"],
        "dims": dims,
        "trials": n_trials,
        "seismic_median": float(np.median(seismic_scores)),
        "seismic_std": float(np.std(seismic_scores)),
        "sa_median": float(np.median(sa_scores)),
        "sa_std": float(np.std(sa_scores)),
        "cma_median": float(np.median(cma_scores)),
        "cma_std": float(np.std(cma_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Seismic Descent Benchmarks")
    parser.add_argument("--dims", type=int, default=5, help="Search space dimensionality")
    parser.add_argument("--trials", type=int, default=5, help="Number of repetitions per test")
    parser.add_argument("--budget", type=int, default=3000, help="Evaluation budget per trial")
    parser.add_argument(
        "--function",
        type=str,
        default="all",
        choices=["all"] + list(ALL_FUNCTIONS.keys()),
        help="Target benchmark function",
    )
    args = parser.parse_args()

    funcs = list(ALL_FUNCTIONS.keys()) if args.function == "all" else [args.function]

    print(f"\n=======================================================")
    print(f" SEISMIC DESCENT BENCHMARK SUITE")
    print(f" Dimension: {args.dims}D | Trials: {args.trials} | Budget: {args.budget}")
    print(f"=======================================================\n")

    print(f"| {'Function':<12} | {'Seismic (Med)':<15} | {'SA (Med)':<15} | {'CMA-ES (Med)':<15} |")
    print(f"|{'-'*14}|{'-'*17}|{'-'*17}|{'-'*17}|")

    for fkey in funcs:
        res = run_benchmark(fkey, dims=args.dims, n_trials=args.trials, eval_budget=args.budget)
        print(
            f"| {res['function']:<12} | "
            f"{res['seismic_median']:<15.4f} | "
            f"{res['sa_median']:<15.4f} | "
            f"{res['cma_median']:<15.4f} |"
        )


if __name__ == "__main__":
    main()
