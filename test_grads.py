import numpy as np
import sys
from benchmark_functions import ALL_FUNCTIONS

def check_grads():
    np.random.seed(42)
    D = 5
    N = 100
    eps = 1e-7
    
    for name, config in ALL_FUNCTIONS.items():
        fn = config['fn']
        grad_fn = config['grad']
        search_range = config['search_range']
        
        X = np.random.uniform(-search_range, search_range, size=(N, D))
        
        analytic_grad = grad_fn(X)
        numeric_grad = np.zeros_like(X)
        
        for i in range(D):
            X_plus = X.copy()
            X_plus[:, i] += eps
            X_minus = X.copy()
            X_minus[:, i] -= eps
            numeric_grad[:, i] = (fn(X_plus) - fn(X_minus)) / (2 * eps)
            
        diff = np.abs(analytic_grad - numeric_grad)
        rel_diff = diff / (np.abs(numeric_grad) + 1e-8)
        
        max_err = np.max(rel_diff)
        mean_err = np.mean(rel_diff)
        
        print(f"Function: {name}")
        print(f"  Max relative error: {max_err:.2e}")
        print(f"  Mean relative error: {mean_err:.2e}")
        if max_err > 1e-4:
            print(f"  [!] High error detected in {name}!")
        print()

if __name__ == '__main__':
    check_grads()
