"""
Standard mathematical benchmark functions and their analytic gradients.

Functions support both 1D shape (D,) and 2D batch shape (N, D).
"""

from typing import Dict, Any
import numpy as np


# --------------------------------------------------------------------------- #
# Sphere
# --------------------------------------------------------------------------- #

def sphere(x: np.ndarray) -> Union[float, np.ndarray]:
    """Sphere function f(x) = sum(x_i^2). Global min at 0 with f(0) = 0."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return float(np.sum(x**2))
    return np.sum(x**2, axis=1)


def sphere_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Sphere function: 2*x."""
    return 2.0 * np.asarray(x, dtype=np.float64)


SPHERE: Dict[str, Any] = {
    "fn": sphere,
    "grad": sphere_grad,
    "name": "Sphere",
    "search_range": 5.12,
    "global_min": 0.0,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Rastrigin
# --------------------------------------------------------------------------- #

def rastrigin(x: np.ndarray) -> Union[float, np.ndarray]:
    """Rastrigin highly multimodal benchmark function."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))
    d = x.shape[1]
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Rastrigin function."""
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


RASTRIGIN: Dict[str, Any] = {
    "fn": rastrigin,
    "grad": rastrigin_grad,
    "name": "Rastrigin",
    "search_range": 5.12,
    "global_min": 0.0,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Schwefel
# --------------------------------------------------------------------------- #

def schwefel(x: np.ndarray) -> Union[float, np.ndarray]:
    """Schwefel function with distant deceptive local optima."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        d = len(x)
        return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))
    d = x.shape[1]
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def schwefel_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Schwefel function."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)

    abs_x = np.abs(x)
    sqrt_abs = np.sqrt(abs_x + 1e-30)
    sin_term = np.sin(sqrt_abs)
    cos_term = np.cos(sqrt_abs)

    grad = -sin_term - (sqrt_abs / 2.0) * cos_term
    return grad[0] if is_1d else grad


SCHWEFEL: Dict[str, Any] = {
    "fn": schwefel,
    "grad": schwefel_grad,
    "name": "Schwefel",
    "search_range": 500.0,
    "global_min": 420.9687,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Ackley
# --------------------------------------------------------------------------- #

def ackley(x: np.ndarray) -> Union[float, np.ndarray]:
    """Ackley function characterized by a nearly flat outer region."""
    x = np.asarray(x, dtype=np.float64)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    if x.ndim == 1:
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return float(-a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e)
    d = x.shape[1]
    sum1 = np.sum(x**2, axis=1)
    sum2 = np.sum(np.cos(c * x), axis=1)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e


def ackley_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Ackley function."""
    x = np.asarray(x, dtype=np.float64)
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]

    sum_sq = np.sum(x**2, axis=1, keepdims=True)
    sqrt_term = np.sqrt(sum_sq / d + 1e-30)

    exp1 = np.exp(-b * sqrt_term)
    term1 = a * b * exp1 * x / (d * sqrt_term)

    sum_cos = np.sum(np.cos(c * x), axis=1, keepdims=True)
    exp2 = np.exp(sum_cos / d)
    term2 = exp2 * c * np.sin(c * x) / d

    grad = term1 + term2
    return grad[0] if is_1d else grad


ACKLEY: Dict[str, Any] = {
    "fn": ackley,
    "grad": ackley_grad,
    "name": "Ackley",
    "search_range": 32.768,
    "global_min": 0.0,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Griewank
# --------------------------------------------------------------------------- #

def griewank(x: np.ndarray) -> Union[float, np.ndarray]:
    """Griewank multimodal benchmark function."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        d = len(x)
        sum_term = np.sum(x**2) / 4000.0
        indices = np.arange(1, d + 1)
        prod_term = np.prod(np.cos(x / np.sqrt(indices)))
        return float(sum_term - prod_term + 1.0)
    d = x.shape[1]
    sum_term = np.sum(x**2, axis=1) / 4000.0
    indices = np.arange(1, d + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(indices)), axis=1)
    return sum_term - prod_term + 1.0


def griewank_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Griewank function."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    _, d = x.shape
    indices = np.arange(1, d + 1)
    sqrt_idx = np.sqrt(indices)

    cos_terms = np.cos(x / sqrt_idx)
    left = np.cumprod(cos_terms, axis=1)
    right = np.cumprod(cos_terms[:, ::-1], axis=1)[:, ::-1]

    prod_without_i = np.ones_like(cos_terms)
    prod_without_i[:, 1:] *= left[:, :-1]
    prod_without_i[:, :-1] *= right[:, 1:]

    sin_terms = np.sin(x / sqrt_idx)
    grad = x / 2000.0 + prod_without_i * sin_terms / sqrt_idx
    return grad[0] if is_1d else grad


GRIEWANK: Dict[str, Any] = {
    "fn": griewank,
    "grad": griewank_grad,
    "name": "Griewank",
    "search_range": 600.0,
    "global_min": 0.0,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Rosenbrock
# --------------------------------------------------------------------------- #

def rosenbrock(x: np.ndarray) -> Union[float, np.ndarray]:
    """Rosenbrock valley benchmark function."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Rosenbrock function."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    grad = np.zeros_like(x)
    grad[:, :-1] += -400.0 * x[:, :-1] * (x[:, 1:] - x[:, :-1]**2) - 2.0 * (1 - x[:, :-1])
    grad[:, 1:] += 200.0 * (x[:, 1:] - x[:, :-1]**2)
    return grad[0] if is_1d else grad


ROSENBROCK: Dict[str, Any] = {
    "fn": rosenbrock,
    "grad": rosenbrock_grad,
    "name": "Rosenbrock",
    "search_range": 5.0,
    "global_min": 1.0,
    "global_min_val": 0.0,
}


ALL_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    "sphere": SPHERE,
    "rastrigin": RASTRIGIN,
    "schwefel": SCHWEFEL,
    "ackley": ACKLEY,
    "griewank": GRIEWANK,
    "rosenbrock": ROSENBROCK,
}
