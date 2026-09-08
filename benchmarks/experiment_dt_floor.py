"""
experiment_dt_floor.py — Empirical test of dt floor (minimum baseline step size).

Investigates whether preventing dt from collapsing to zero during cyclic phases
improves convergence speed and solution quality across benchmark functions.

Evaluates:
  floor in [0.0 (v20 canonical), 0.1, 0.25, 0.5, 0.75, 1.0 (pure constant dt)]
  versus CMA-ES and Simulated Annealing.
"""

import argparse
import time
from pathlib import Path
import numpy as np

# Ensure package import
src_path = Path(__file__).resolve().parent.parent / "src"
import sys
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from seismic_descent import ALL_FUNCTIONS
from seismic_descent.rff import RandomFourierFeatures

try:
    import cma
    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


def seismic_with_dt_floor(
    fn,
    fn_grad,
    x0,
    bounds,
    eval_budget: int = 3000,
    n_particles: int = 10,
    dt_base: float = 0.2,
    dt_floor: float = 0.0,
    noise_amplitude: float = 0.5,
    dt_cycles_multiplier: float = 5.0,
    seed: int = 1,
):
    """
    Seismic Swarm with variable dt_floor:
    dt(t) = dt_base * (dt_floor + (1.0 - dt_floor) * |sin(t * multiplier)|)
    """
    n_steps = max(20, eval_budget // n_particles)
    dim = len(x0)
    bounds = np.asarray(bounds, dtype=np.float64)
    center = (bounds[:, 1] + bounds[:, 0]) / 2.0
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0

    rff = RandomFourierFeatures(dim=dim, r=64, n_octaves=1, base_lengthscale=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    x_norm = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    x_norm[0] = np.clip((x0 - center) / half_range, -1.0, 1.0)

    dt_noise = (10.0 * np.pi) / n_steps
    t = 0.0

    x_real = center + x_norm * half_range
    real_vals = np.asarray(fn(x_real), dtype=np.float64)
    best_val = float(np.min(real_vals))

    for step in range(n_steps):
        amp = noise_amplitude * np.sin(t * 2.0)
        x_real = center + x_norm * half_range
        f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
        if f_grad_real.ndim == 1:
            f_grad_real = f_grad_real.reshape(1, -1)

        # 1. Chain rule to normalized hypercube
        f_grad_mapped = f_grad_real * half_range

        # 2. L2 Gradient Normalization
        norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
        f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)

        # 3. RFF Spatially Correlated Noise Gradient
        noise_grad = rff.grad(x_norm, t, amplitude=amp)
        grad = f_grad_dir + noise_grad

        # 4. Decoupled cyclic dt with adjustable floor
        cyclic_oscillation = np.abs(np.sin(t * dt_cycles_multiplier))
        effective_scale = dt_floor + (1.0 - dt_floor) * cyclic_oscillation
        current_dt = dt_base * effective_scale

        x_norm -= current_dt * grad
        np.clip(x_norm, -1.0, 1.0, out=x_norm)
        t += dt_noise

        x_real = center + x_norm * half_range
        step_vals = np.asarray(fn(x_real), dtype=np.float64)
        min_step = float(np.min(step_vals))
        if min_step < best_val:
            best_val = min_step

    return best_val


def run_cmaes(fn, x0, bounds, eval_budget):
    if not _CMA_AVAILABLE:
        return float("nan")
    opts = cma.CMAOptions()
    opts["bounds"] = [bounds[:, 0].tolist(), bounds[:, 1].tolist()]
    opts["maxfevals"] = eval_budget
    opts["verbose"] = -9
    sigma0 = float(np.mean(bounds[:, 1] - bounds[:, 0]) * 0.2)
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [float(fn(np.array(s))) for s in solutions])
    return float(es.result.fbest)


def main():
    parser = argparse.ArgumentParser(description="Test dt floor effect on Seismic Descent")
    parser.add_argument("--trials", type=int, default=15, help="Number of repetitions per config")
    parser.add_argument("--budget", type=int, default=3000, help="Evaluation budget per trial")
    parser.add_argument("--dims", type=int, default=5, help="Dimensionality")
    args = parser.parse_args()

    floors = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    functions = ["rastrigin", "rosenbrock", "griewank", "ackley"]

    print(f"\n==========================================================================")
    print(f" DT FLOOR EMPIRICAL EXPERIMENT ({args.dims}D | Budget={args.budget} | Trials={args.trials})")
    print(f" Formula: dt = dt_base * (floor + (1 - floor) * |sin(t * 5)|)")
    print(f"==========================================================================\n")

    for fkey in functions:
        fconfig = ALL_FUNCTIONS[fkey]
        fn = fconfig["fn"]
        fn_grad = fconfig["grad"]
        search_range = fconfig["search_range"]

        bounds = np.zeros((args.dims, 2))
        bounds[:, 0] = -search_range
        bounds[:, 1] = search_range

        print(f"--- Function: {fconfig['name']} ({args.dims}D) ---")
        header = f"| {'Configuration':<22} | {'Median Error':<16} | {'Mean Error':<16} | {'Best Error':<16} |"
        print(header)
        print("|" + "-"*24 + "|" + "-"*18 + "|" + "-"*18 + "|" + "-"*18 + "|")

        # 1. Baseline CMA-ES
        cma_vals = []
        for trial in range(args.trials):
            rng = np.random.default_rng(trial * 1000 + 42)
            x0 = rng.uniform(-search_range, search_range, size=args.dims)
            cma_vals.append(run_cmaes(fn, x0, bounds, args.budget))

        print(f"| {'[Baseline] CMA-ES':<22} | {np.median(cma_vals):<16.4e} | {np.mean(cma_vals):<16.4e} | {np.min(cma_vals):<16.4e} |")

        # 2. Sweep floors
        best_floor = None
        best_med = float("inf")

        for floor in floors:
            seismic_vals = []
            tag = f"floor = {floor:<4} " + ("(v20 Canonical)" if floor == 0.0 else ("(Pure Const)" if floor == 1.0 else ""))

            for trial in range(args.trials):
                rng = np.random.default_rng(trial * 1000 + 42)
                x0 = rng.uniform(-search_range, search_range, size=args.dims)
                bval = seismic_with_dt_floor(
                    fn, fn_grad, x0, bounds,
                    eval_budget=args.budget,
                    dt_floor=floor,
                    seed=trial + 1,
                )
                seismic_vals.append(bval)

            med = float(np.median(seismic_vals))
            if med < best_med:
                best_med = med
                best_floor = floor

            print(f"| {tag:<22} | {med:<16.4e} | {np.mean(seismic_vals):<16.4e} | {np.min(seismic_vals):<16.4e} |")

        print(f"  >>> Best floor for {fconfig['name']}: {best_floor} (Median: {best_med:.4e})\n")


if __name__ == "__main__":
    main()
