import numpy as np
from seismic_descent_v18 import seismic_swarm
from benchmark_functions import ACKLEY, ROSENBROCK

def sweep_params(func_config, dt_list, amp_list, dims=5, n_steps=600):
    fn = func_config['fn']
    fn_grad = func_config['grad']
    search_range = func_config['search_range']
    print(f"--- Sweeping {func_config['name']} {dims}D ---")
    
    np.random.seed(42)
    # create consistent starting point
    x0 = np.random.uniform(-search_range, search_range, size=dims)
    
    best_combo = None
    best_val = np.inf
    
    for dt in dt_list:
        for amp in amp_list:
            _, val, _ = seismic_swarm(
                fn, fn_grad, x0,
                n_steps=n_steps,
                n_particles=5,
                search_range=search_range,
                noise_amplitude=amp,
                dt=dt
            )
            if val < best_val:
                best_val = val
                best_combo = (dt, amp)
            # print(f"dt={dt:7.4f}, amp={amp:7.1f} -> {val:7.4f}")
            
    print(f"BEST -> dt={best_combo[0]:.4f}, amp={best_combo[1]:.1f} | min_val={best_val:.6f}")
    if best_val < 5.0:
        print(">> SUCCESS PARAMETERS FOUND!!")
    else:
        print(">> STILL FAILING.")
    print()

if __name__ == '__main__':
    # Ackley sweep
    dts_ackley = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    amps_ackley = [0.1, 0.5, 1.0, 5.0, 15.0, 50.0]
    sweep_params(ACKLEY, dts_ackley, amps_ackley)
    
    # Rosenbrock sweep
    dts_rosen = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    amps_rosen = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    sweep_params(ROSENBROCK, dts_rosen, amps_rosen)
