"""
benchmark_functions.py — Registro de funciones de optimización benchmark.

Cada función expone:
  - fn(X) : evaluación vectorizada, X shape (N, D) -> (N,) o (D,) -> escalar
  - grad(X) : gradiente analítico vectorizado, mismas shapes
  - info : dict con 'name', 'search_range', 'global_min', 'global_min_val'
"""

import numpy as np

# ================================================================== #
# RASTRIGIN
# ================================================================== #

def rastrigin(X):
    if X.ndim == 1:
        return 10.0 * len(X) + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X))
    D = X.shape[1]
    return 10.0 * D + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)

def rastrigin_grad(X):
    return 2.0 * X + 20.0 * np.pi * np.sin(2.0 * np.pi * X)

RASTRIGIN = {
    'fn': rastrigin,
    'grad': rastrigin_grad,
    'name': 'Rastrigin',
    'search_range': 5.12,
    'global_min': 0.0,
    'global_min_val': 0.0,
}

# ================================================================== #
# SCHWEFEL
# ================================================================== #

def schwefel(X):
    if X.ndim == 1:
        D = len(X)
        return 418.9829 * D - np.sum(X * np.sin(np.sqrt(np.abs(X))))
    D = X.shape[1]
    return 418.9829 * D - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

def schwefel_grad(X):
    is_1d = X.ndim == 1
    if is_1d:
        X = X.reshape(1, -1)
        
    abs_X = np.abs(X)
    sqrt_abs = np.sqrt(abs_X + 1e-30)
    sin_term = np.sin(sqrt_abs)
    cos_term = np.cos(sqrt_abs)
    
    grad = -sin_term - (X / (2.0 * sqrt_abs + 1e-30)) * cos_term
    
    if is_1d:
        return grad[0]
    return grad

SCHWEFEL = {
    'fn': schwefel,
    'grad': schwefel_grad,
    'name': 'Schwefel',
    'search_range': 500.0,
    'global_min': 420.9687,
    'global_min_val': 0.0,
}

# ================================================================== #
# ACKLEY
# ================================================================== #

def ackley(X):
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    if X.ndim == 1:
        D = len(X)
        sum1 = np.sum(X**2)
        sum2 = np.sum(np.cos(c * X))
        return -a * np.exp(-b * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + a + np.e
    D = X.shape[1]
    sum1 = np.sum(X**2, axis=1)
    sum2 = np.sum(np.cos(c * X), axis=1)
    return -a * np.exp(-b * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + a + np.e

def ackley_grad(X):
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    is_1d = X.ndim == 1
    if is_1d:
        X = X.reshape(1, -1)
    D = X.shape[1]
    
    sum_sq = np.sum(X**2, axis=1, keepdims=True)
    sqrt_term = np.sqrt(sum_sq / D + 1e-30)
    
    exp1 = np.exp(-b * sqrt_term)
    term1 = a * b * exp1 * X / (D * sqrt_term)
    
    sum_cos = np.sum(np.cos(c * X), axis=1, keepdims=True)
    exp2 = np.exp(sum_cos / D)
    term2 = exp2 * c * np.sin(c * X) / D
    
    grad = term1 + term2
    if is_1d:
        return grad[0]
    return grad

ACKLEY = {
    'fn': ackley,
    'grad': ackley_grad,
    'name': 'Ackley',
    'search_range': 32.768,
    'global_min': 0.0,
    'global_min_val': 0.0,
}

# ================================================================== #
# GRIEWANK
# ================================================================== #

def griewank(X):
    if X.ndim == 1:
        D = len(X)
        sum_term = np.sum(X**2) / 4000.0
        indices = np.arange(1, D + 1)
        prod_term = np.prod(np.cos(X / np.sqrt(indices)))
        return sum_term - prod_term + 1.0
    D = X.shape[1]
    sum_term = np.sum(X**2, axis=1) / 4000.0
    indices = np.arange(1, D + 1)
    prod_term = np.prod(np.cos(X / np.sqrt(indices)), axis=1)
    return sum_term - prod_term + 1.0

def griewank_grad(X):
    is_1d = X.ndim == 1
    if is_1d:
        X = X.reshape(1, -1)
    N, D = X.shape
    indices = np.arange(1, D + 1)
    sqrt_idx = np.sqrt(indices)
    
    cos_terms = np.cos(X / sqrt_idx)
    full_prod = np.prod(cos_terms, axis=1, keepdims=True)
    
    safe_cos = np.where(np.abs(cos_terms) < 1e-15, 1e-15 * np.sign(cos_terms + 1e-20), cos_terms)
    prod_without_i = full_prod / safe_cos
    
    sin_terms = np.sin(X / sqrt_idx)
    
    grad = X / 2000.0 + prod_without_i * sin_terms / sqrt_idx
    if is_1d:
        return grad[0]
    return grad

GRIEWANK = {
    'fn': griewank,
    'grad': griewank_grad,
    'name': 'Griewank',
    'search_range': 600.0,
    'global_min': 0.0,
    'global_min_val': 0.0,
}

# ================================================================== #
# ROSENBROCK
# ================================================================== #

def rosenbrock(X):
    if X.ndim == 1:
        return np.sum(100.0 * (X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)

def rosenbrock_grad(X):
    is_1d = X.ndim == 1
    if is_1d:
        X = X.reshape(1, -1)
    N, D = X.shape
    grad = np.zeros_like(X)
    
    grad[:, :-1] += -400.0 * X[:, :-1] * (X[:, 1:] - X[:, :-1]**2) - 2.0 * (1 - X[:, :-1])
    grad[:, 1:]  += 200.0 * (X[:, 1:] - X[:, :-1]**2)
    
    if is_1d:
        return grad[0]
    return grad

ROSENBROCK = {
    'fn': rosenbrock,
    'grad': rosenbrock_grad,
    'name': 'Rosenbrock',
    'search_range': 5.0,
    'global_min': 1.0,
    'global_min_val': 0.0,
}

# ================================================================== #
# Registro completo
# ================================================================== #

ALL_FUNCTIONS = {
    'rastrigin': RASTRIGIN,
    'schwefel': SCHWEFEL,
    'ackley': ACKLEY,
    'griewank': GRIEWANK,
    'rosenbrock': ROSENBROCK,
}
