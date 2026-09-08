import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from noise import pnoise2
import cma

def rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def perlin_landscape(x, y, t, amplitude=15.0, frequency=0.5, octaves=4):
    val = 0.0
    amp = amplitude
    freq = frequency
    for _ in range(octaves):
        val += amp * pnoise2(x * freq + t * 0.3, y * freq + t * 0.7)
        amp *= 0.5
        freq *= 2.0
    return val

def combined_landscape(x, y, t, noise_amplitude=15.0):
    return rastrigin(x, y) + perlin_landscape(x, y, t, amplitude=noise_amplitude)

def perlin_optimization(x0, y0, n_steps=5000, dt=0.01, noise_amplitude=15.0,
                         noise_decay=1.0, search_range=5.12):
    x, y = x0, y0
    t = 0.0
    best_x, best_y = x, y
    best_val = rastrigin(x, y)
    trajectory = [(x, y, best_val)]
    best_history = [best_val]
    eps = 1e-4

    for step in range(n_steps):
        decay = noise_decay ** step
        freq = 2.0 * decay  # frecuencia alta al principio, lenta al final
        amp = noise_amplitude * decay * abs(np.sin(t * freq))
        f_center = combined_landscape(x, y, t, amp)
        f_dx = combined_landscape(x + eps, y, t, amp)
        f_dy = combined_landscape(x, y + eps, t, amp)
        grad_x = (f_dx - f_center) / eps
        grad_y = (f_dy - f_center) / eps
        x -= dt * grad_x
        y -= dt * grad_y
        x = np.clip(x, -search_range, search_range)
        y = np.clip(y, -search_range, search_range)
        t += 0.05
        real_val = rastrigin(x, y)
        if real_val < best_val:
            best_val = real_val
            best_x, best_y = x, y
        trajectory.append((x, y, real_val))
        best_history.append(best_val)

    return best_x, best_y, best_val, trajectory, best_history

def simulated_annealing(x0, y0, n_steps=5000, T0=10.0, cooling=0.999,
                         step_size=0.3, search_range=5.12):
    x, y = x0, y0
    current_val = rastrigin(x, y)
    best_x, best_y = x, y
    best_val = current_val
    trajectory = [(x, y, current_val)]
    best_history = [best_val]
    T = T0

    for step in range(n_steps):
        nx = x + np.random.normal(0, step_size)
        ny = y + np.random.normal(0, step_size)
        nx = np.clip(nx, -search_range, search_range)
        ny = np.clip(ny, -search_range, search_range)
        new_val = rastrigin(nx, ny)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            x, y = nx, ny
            current_val = new_val
        if current_val < best_val:
            best_val = current_val
            best_x, best_y = x, y
        T *= cooling
        trajectory.append((x, y, current_val))
        best_history.append(best_val)

    return best_x, best_y, best_val, trajectory, best_history

def cmaes(x0, y0, n_steps=5000, search_range=5.12):
    def f(v):
        return rastrigin(v[0], v[1])
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range, -search_range], [search_range, search_range]]
    opts['maxfevals'] = n_steps
    opts['verbose'] = -9
    es = cma.CMAEvolutionStrategy([x0, y0], 4.0, opts)
    best_val = f([x0, y0])
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [f(s) for s in solutions])
        if es.result.fbest < best_val:
            best_val = es.result.fbest
    bx, by = es.result.xbest
    return bx, by, es.result.fbest

# --- Benchmark ---
np.random.seed(42)
n_trials = 50
n_steps = 5000
threshold = 0.5

perlin_successes, sa_successes, cma_successes = 0, 0, 0
perlin_vals, sa_vals, cma_vals = [], [], []

for trial in range(n_trials):
    x0 = np.random.uniform(-5.12, 5.12)
    y0 = np.random.uniform(-5.12, 5.12)
    px, py, pval, _, _ = perlin_optimization(x0, y0, n_steps=n_steps)
    perlin_vals.append(pval)
    if pval < threshold:
        perlin_successes += 1
    sx, sy, sval, _, _ = simulated_annealing(x0, y0, n_steps=n_steps)
    sa_vals.append(sval)
    if sval < threshold:
        sa_successes += 1
    cx, cy, cval = cmaes(x0, y0, n_steps=n_steps * 10)
    cma_vals.append(cval)
    if cval < threshold:
        cma_successes += 1

print(f"=== Resultados sobre {n_trials} pruebas ===")
print(f"Perlin Optimization: {perlin_successes}/{n_trials} exitos, "
      f"media={np.mean(perlin_vals):.4f}, mediana={np.median(perlin_vals):.4f}")
print(f"Simulated Annealing: {sa_successes}/{n_trials} exitos, "
      f"media={np.mean(sa_vals):.4f}, mediana={np.median(sa_vals):.4f}")
print(f"CMA-ES:              {cma_successes}/{n_trials} exitos, "
      f"media={np.mean(cma_vals):.4f}, mediana={np.median(cma_vals):.4f}")

# --- Visualización ---
x0, y0 = 4.0, -3.5
pbx, pby, _, ptraj, phist = perlin_optimization(x0, y0, n_steps=n_steps)
sbx, sby, _, straj, shist = simulated_annealing(x0, y0, n_steps=n_steps)

xx = np.linspace(-5.12, 5.12, 300)
yy = np.linspace(-5.12, 5.12, 300)
X, Y = np.meshgrid(xx, yy)
Z = rastrigin(X, Y)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
ptraj_arr = np.array(ptraj)
ax.plot(ptraj_arr[:, 0], ptraj_arr[:, 1], 'r-', alpha=0.5, linewidth=0.5)
ax.plot(x0, y0, 'wo', markersize=10, label='Inicio')
ax.plot(pbx, pby, 'r*', markersize=15, label='Mejor Perlin')
ax.plot(0, 0, 'g*', markersize=15, label='Óptimo global')
ax.set_title('Trayectoria Perlin Optimization')
ax.legend(fontsize=8)

ax = axes[1]
ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
straj_arr = np.array(straj)
ax.plot(straj_arr[:, 0], straj_arr[:, 1], 'r-', alpha=0.5, linewidth=0.5)
ax.plot(x0, y0, 'wo', markersize=10, label='Inicio')
ax.plot(sbx, sby, 'r*', markersize=15, label='Mejor SA')
ax.plot(0, 0, 'g*', markersize=15, label='Óptimo global')
ax.set_title('Trayectoria Simulated Annealing')
ax.legend(fontsize=8)

ax = axes[2]
ax.semilogy(phist, 'b-', alpha=0.7, label='Perlin Opt')
ax.semilogy(shist, 'r-', alpha=0.7, label='SA')
ax.set_xlabel('Iteración')
ax.set_ylabel('Mejor valor f(x) [log]')
ax.set_title('Convergencia comparada')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('perlin_vs_sa_rastrigin.png', dpi=150)
print("Gráfica guardada en perlin_vs_sa_rastrigin.png")
