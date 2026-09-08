# Seismic Descent

An optimization algorithm based on gradient descent over a dynamic landscape perturbed by spatially correlated noise. 

![Seismic Descent](assets/seismic-descent.png)

*Read this documentation in [Spanish](README.es.md)*

## The Concept

Instead of taking random jumps to escape local minima (as seen in Simulated Annealing), we trigger an **earthquake on the ground**. The particle simply does the only thing it knows how to do: roll downhill. But because the ground tremors in a coherent, multi-scale fashion, local minima temporarily morph into slopes, allowing the particle to escape naturally and effortlessly.

```
f_total(x, t) = f_original(x) + A(t) * noise(x, t)
```

- `f_original` — the actual objective function
- `noise(x, t)` — spatially correlated noise (Perlin for 2D, RFF for N-Dimensions)
- `A(t) = A0 * sin(t * freq(t))` — cyclical amplitude with decreasing frequency
- The overall best point is continuously tracked strictly against `f_original`, naturally filtering out the artificial earthquake topology.

**The key advantage over Simulated Annealing (SA):** The noise is **spatially correlated** — two nearby points share similar perturbations. The particle smoothly slides boundaries to another valley; it does not blindly teleport. Superimposed noise octaves provide both broad inter-valley exploration and ultra-fine refinement locally, all packed into a single mathematical mechanism.

## N-Dimensional Noise: Random Fourier Features

In 2D, strict Perlin noise is highly effective. To scale efficiently across N-Dimensions without grid-bound computational explosion, we approximate a *Gaussian Random Field (GRF)* via Random Fourier Features (Rahimi & Recht, 2007):

```
noise(x) ≈ sqrt(2/R) * A * Σ_r cos(ω_r · x + t*drift_r + φ_r)
```

Where `ω_r ~ N(0, 1/l²·I)` are vectors in R^D. Being N-Dimensional vectors, they force geometric spatial correlation and feature overlap in any high-dimensional search space instantly.

## Milestones & Recent Optimizations (v7 - v20)

The repository condenses intensive empirical research where the algorithm transcended severe bottlenecks:

- **Analytic Gradients ($\mathcal{O}(1)$)**: We replaced costly finite-difference geometric mapping with strict analytic gradients over the RFF field. Calculating the next earthquake slide went from minutes to near-zero CPU cost.
- **Negative Polarity (`abs()` removal)**: We proved mathematically that letting the sine bounce back into negative amplitudes acts as an active topological inverter, transforming barrier hills into gravitational escape funnels.
- **Seismic Swarm**: Porting logic to fully parallelized `numpy` matrices, $N$ particles share a single dynamic GRF landscape plane ($\mathcal{O}(ND)$ complexity).
- **Universal Bidirectional Normalization (v20)**: Mapping coordinates internally to $[-1, 1]^D$ and extracting directional gradient ($\nabla / \|\nabla\|$) decoupled the step size from target loss magnitudes (solving divergence in steep valleys like Rosenbrock).
- **Decoupled Step Schedule with Baseline Floor (`dt_floor = 0.2`)**: Eliminating zero-step freezing during cyclic troughs yielded an immediate +24% to +30% convergence speedup across all tested landscapes.

## Empirical Benchmark & Multi-Budget Scaling Analysis

![Budget Scaling Analysis](assets/budget_scaling_curves.png)

### Rastrigin 5D — Multi-Budget Scaling vs CMA-ES & Simulated Annealing

A fundamental characteristic of Seismic Descent is its **continuous ergodic escape capability**. While Covariance Matrix Adaptation (CMA-ES) rapidly contracts around an initial basin, its variance shrinks ($\sigma \to 0$), causing it to suffer from premature convergence (the *"CMA-ES Infarction"*). 

In contrast, Seismic Descent's periodic multi-scale landscape oscillations keep shaking particles out of local minima traps, crossing and beating CMA-ES at higher budgets while running up to **17x faster on CPU**:

| Evaluation Budget | Median **Seismic** | Median **CMA-ES** | Median **SA** | Winner | CPU Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **500** | 20.31 | **8.16** | 21.42 | CMA-ES | Seismic is **17.0x** faster |
| **1,000** | 16.03 | **8.95** | 13.77 | CMA-ES | Seismic is **16.6x** faster |
| **3,000** | 9.21 | **4.97** | 10.50 | CMA-ES | Seismic is **9.5x** faster |
| **10,000** | 5.86 | **2.98** | 6.66 | CMA-ES | Seismic is **2.9x** faster |
| **25,000** | **4.85** 🏆 | 6.96 ❌ | 5.44 | **SEISMIC** | Seismic is **1.3x** faster |

*Metrics recorded over 15 independent trials per budget point on Rastrigin 5D using `benchmarks/benchmark_budget_scaling.py`.*

### Convergence Dynamics & Ablation Highlights

![Convergence Dynamics and Ablation Study](assets/convergence_study.png)

1. **$L_2$ Gradient Normalization is Essential**: In Rosenbrock 5D, unnormalized gradients explode to median errors $> 10,000$, whereas $L_2$ normalized Seismic v20 stably navigates the curved valley with median error **6.45**.
2. **`dt_floor` Elimination of Particle Freezing**: Adding a 20-25% baseline velocity prevents the swarm from stalling at the zero-crossings of the sine schedule, improving median error by **24% on Rastrigin** (from 13.19 to 10.04) and **30% on Rosenbrock** (from 6.45 to 4.51).


## Key Property: Seismic Ergodicity

A fundamental discovery in the development of the algorithm (consolidated in v19) is that the exploration driven by correlated noise must be **ergodic**.

Instead of blindly shaking the particle with high-frequency "white noise", *Seismic Descent* generates **complete, coherent topological landscapes** of low and high frequencies (via RFF octaves), and smoothly morphs (interpolates) from one random landscape to the next over time.

This continuous mutation mathematically guarantees that a particle, guided purely by the gradient of this "mutating ground", will eventually explore and visit the entirety of the search space without getting trapped in infinite loops or plateaus. **Ergodicity** is what allows the swarm to flow through the terrain like a liquid, guaranteeing an escape from even the deepest local minima.

### Empirical Thermodynamic Properties (Laplacian Ergodicity)

![Laplacian Ergodic Histogram](assets/laplacian_ergodicity.png)
*Notice how the green ergodicity histogram perfectly draws a sharp Laplacian distribution ($e^{-|x|}$) around each local minimum. The peak height directly correlates with the minimum's depth, while the width correlates with the steepness of the basin walls.*

Observations from the 1D visualizer reveal a profound statistical mechanics property: as `t -> ∞`, the particle's spatial probability density function (the ergodic heatmap) converges into sharp **Laplacian** peaks centered at local minima.

1. **Boltzmann-Gibbs Emulation**: The depth of a minimum determines the exact statistical amplitude of the peak. This means Seismic Descent naturally performs robust Monte Carlo sampling equivalent to a thermodynamic system.
2. **Heavy-Tailed Escapes**: Unlike traditional Gaussian (Brownian) noise used in Langevin dynamics or SGD ($e^{-x^2}$), the **Laplacian** signature ($e^{-|x|}$) empirically proves that the seismic spatial field induces **heavy-tailed jumps**. The probability of the particle massively leaping out of a basin's boundaries is orders of magnitude higher than in standard random walks. This mathematically explains the algorithm's exceptional capability to escape sub-optimal valleys where standard optimizers get permanently trapped.

## Interactive Visualizers

To truly understand how Seismic Descent works, you can explore the algorithm interactively in your browser without any installation:

- **[1D Seismic Explorer](https://mcarbonell.github.io/seismic-descent/visualizer/1d_explorer.html)**: Visualize how the original function, the seismic noise phase, and the morphed landscape interact. Watch the particles escape local minima and see the "Ergodic Heatmap" prove the organic search space coverage.
- **[2D Interactive Map](https://mcarbonell.github.io/seismic-descent/visualizer/index.html)**: Observe the 2D spatial correlation of the Perlin-generated earthquakes visually dragging particles towards the global minimum.

![1D Visualizer Ergodicity](assets/seismic-1d.png)
*Snapshot of the 1D Visualizer optimizing the highly non-linear Rastrigin function. The green histogram at the bottom (Ergodic Heatmap) perfectly maps the continuous topological exploration of the particle across all local minima basins, tangibly proving the algorithm avoids infinite entrapment.*

## Installation

Install from the repository using pip:

```bash
# Core package
pip install -e .

# With optional PyTorch and benchmark suites
pip install -e ".[dev]"
```

Or install external dependencies directly:
```bash
pip install numpy matplotlib torch cma noise pytest
```

## Quickstart

### Python API

```python
from seismic_descent import seismic_swarm, ALL_FUNCTIONS

# Load Rastrigin 5D benchmark
rastrigin = ALL_FUNCTIONS["rastrigin"]
bounds = [[-5.12, 5.12]] * 5
x0 = [3.0] * 5

# Optimize using the champion v20 architecture
best_x, best_val, info = seismic_swarm(
    fn=rastrigin["fn"],
    fn_grad=rastrigin["grad"],
    x0_real=x0,
    bounds=bounds,
    n_steps=2000,
    n_particles=10,
    dt_base=0.2,
    noise_amplitude=0.5,
)

print(f"Optimal value found: {best_val:.6f}")
```

### CLI Benchmark Suite

Run the automated benchmark suite comparing Seismic Descent against Simulated Annealing (SA) and CMA-ES:

```bash
# Run 5D benchmark across all functions
python -m benchmarks.benchmark_suite --dims 5 --trials 5

# Run 2D benchmark on Rastrigin with custom budget
python -m benchmarks.benchmark_suite --dims 2 --trials 3 --budget 2000 --function rastrigin
```

Run automated tests:
```bash
pytest -v
```

## PyTorch Integration

The Seismic Descent algorithm is available as a standard PyTorch optimizer. This allows training neural networks with spatially correlated "earthquake" perturbations to escape local minima:

```python
from seismic_descent import SeismicOptimizer

model = MyModel()
optimizer = SeismicOptimizer(
    model.parameters(), 
    lr=0.01, 
    noise_amplitude=0.5, 
    n_cycles=10,
)
```

See [legacy/seismic_versions/benchmark_mnist.py](legacy/seismic_versions/benchmark_mnist.py) for a complete neural network training benchmark and [docs/pytorch_optimizer.md](docs/pytorch_optimizer.md) for technical derivations.

### Latest Benchmark (MNIST - 20 Epochs)

| Optimizer | Accuracy | Margin |
| :--- | :--- | :--- |
| **SGD** | **98.28%** | Base |
| **Adaptive Floored Seismic** | **97.90%** | ✅ Beats Adam |
| **Adam** | 97.79% | - |

## Project Structure

```
seismic-descent/
├── src/seismic_descent/            # Core installable Python package
│   ├── core.py                     # SeismicSwarm (v20 champion architecture)
│   ├── rff.py                      # Random Fourier Features (analytic gradients)
│   ├── torch_optimizer.py          # PyTorch SeismicOptimizer module
│   └── functions.py                # Rastrigin, Schwefel, Ackley, Griewank, Rosenbrock
│
├── benchmarks/                     # Benchmark runners and comparative suites
│   ├── benchmark_suite.py          # Automated CLI benchmark runner (vs SA & CMA-ES)
│   └── timing/                     # Profiling and execution time tests
│
├── legacy/                         # Preserved chronological experimental versions
│   ├── perlin_opt/                 # v1 to v17 (Perlin, value noise, early RFF swarms)
│   ├── seismic_versions/           # v18 to v22, vmorph, and MNIST experiments
│   └── README.md                   # Detailed experimental history guide
│
├── tests/                          # Automated unit tests (pytest)
│   ├── test_rff.py                 # RFF properties and analytic gradient checks
│   ├── test_seismic_swarm.py       # Optimization convergence tests
│   └── test_torch_optimizer.py     # PyTorch optimizer validation
│
├── visualizer/                     # Interactive HTML5/Canvas visualizers (GitHub Pages)
│   ├── 1d_explorer.html            # 1D ergodic landscape deformation & heatmap
│   ├── 2d_explorer.html            # 2D wireframe mesh & swarm trajectories
│   └── index.html                  # 2D interactive canvas map
│
├── assets/                         # Visual assets and README graphics
├── docs/                           # Research findings (findings_v1 to v23), theory notes
└── scratch/                        # Developer experimental scratchpad
```

## Future Scope

- **Hyperparameter Sweeping**: Conducting formal automated Grid-Search bounds to tie dimension variance $D$ across strict optimal ruleses for $K$ cycles and spatial `$A$` amplitude bounds.
- **Machine Learning Integration**: Forking gradient hooks directly into Pytorch ML logic to benchmark `Seismic Optimizers` in deep parameter spaces (e.g., standard MNIST tests), utilizing training epochs to map noise drifts.
- **Non-Euclidean Topology Adapting**: Re-architecting RFF frameworks as discrete cost matrices to battle Traveling Salesman Problems (TSP).
