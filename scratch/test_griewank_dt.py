import numpy as np
from seismic_descent_v18 import seismic_swarm
from benchmark_functions import GRIEWANK

fn = GRIEWANK['fn']
fn_grad = GRIEWANK['grad']
search_range = GRIEWANK['search_range']

np.random.seed(42)
x0 = np.random.uniform(-search_range, search_range, size=5)
# Using original dt
_, sval1, _ = seismic_swarm(fn, fn_grad, x0, n_steps=600, n_particles=5, search_range=search_range, noise_amplitude=15.0, dt=0.01)
# Using scaled dt
_, sval2, _ = seismic_swarm(fn, fn_grad, x0, n_steps=600, n_particles=5, search_range=search_range, noise_amplitude=15.0, dt=5.0)

print(f"Original dt (0.01) minimum: {sval1}")
print(f"Scaled dt (5.0) minimum: {sval2}")
