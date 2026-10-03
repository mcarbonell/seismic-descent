"""Extended benchmark functions with exact analytic gradients.

Complements ``functions.py`` (Sphere, Rastrigin, Schwefel, Ackley, Griewank,
Rosenbrock) with standard global-optimization benchmarks so that comparative
studies cover a broader, more publication-grade suite (12 functions):

- Levy (multimodal, widespread use in BBO literature)
- Michalewicz (deceptive multimodal; known best-known optima for D in {2, 5, 10})
- Zakharov (unimodal, plate-shaped)
- Styblinski-Tang (multimodal separable)
- Dixon-Price (valley-shaped)
- Trid (bowl-shaped with strong interactions)

All functions accept (D,) or (N, D) input, mirroring ``functions.py`` conventions.
Every gradient here is verified against central finite differences in
``tests/test_functions_gradients.py``.
"""

from typing import Any, Dict, Optional, Union
import numpy as np

__all__ = [
    "LEVY", "MICHALEWICZ", "ZAKHAROV", "STYBLINSKI_TANG", "DIXON_PRICE", "TRID",
    "EXTENDED_FUNCTIONS",
]


# --------------------------------------------------------------------------- #
# Levy
# --------------------------------------------------------------------------- #

def levy(x: np.ndarray) -> Union[float, np.ndarray]:
    """Levy function. Global min f=0 at x=(1, ..., 1)."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term2 = np.sum(
        (w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2),
        axis=1,
    )
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    out = term1 + term2 + term3
    return float(out[0]) if is_1d else out


def levy_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Levy (w_i = 1 + (x_i - 1)/4, dw/dx = 1/4)."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    w = 1.0 + (x - 1.0) / 4.0
    grad_w = np.zeros_like(w)

    # d/dw_1 sin^2(pi w_1)
    grad_w[:, 0] += np.pi * np.sin(2.0 * np.pi * w[:, 0])
    # d/dw_i (w_i-1)^2 [1 + 10 sin^2(pi w_i + 1)], i = 1..d-1
    wi = w[:, :-1]
    s = np.sin(np.pi * wi + 1.0)
    grad_w[:, :-1] += (2.0 * (wi - 1.0) * (1.0 + 10.0 * s**2)
                       + 10.0 * np.pi * (wi - 1.0)**2 * np.sin(2.0 * (np.pi * wi + 1.0)))
    # d/dw_d (w_d-1)^2 [1 + sin^2(2 pi w_d)]
    wd = w[:, -1]
    s2 = np.sin(2.0 * np.pi * wd)
    grad_w[:, -1] += 2.0 * (wd - 1.0) * (1.0 + s2**2) + 2.0 * np.pi * (wd - 1.0)**2 * np.sin(4.0 * np.pi * wd)

    grad = grad_w * 0.25
    return grad[0] if is_1d else grad


LEVY: Dict[str, Any] = {
    "fn": levy,
    "grad": levy_grad,
    "name": "Levy",
    "search_range": 10.0,
    "global_min": 1.0,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Michalewicz (m = 10)
# --------------------------------------------------------------------------- #

_MICH_M = 10.0

# Best-known optima (standard literature values) for reporting purposes.
_MICH_BEST_KNOWN = {2: -1.8013, 5: -4.687658, 10: -9.660151}


def michalewicz(x: np.ndarray) -> Union[float, np.ndarray]:
    """Michalewicz function (m=10). Deceptive multimodal; optimum depends on D."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    i = np.arange(1, d + 1, dtype=np.float64)
    out = -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * _MICH_M), axis=1)
    return float(out[0]) if is_1d else out


def michalewicz_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Michalewicz (m=10)."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    i = np.arange(1, d + 1, dtype=np.float64)
    u = i * x**2 / np.pi
    s = np.sin(u)
    c = np.cos(u)
    dterm = (np.cos(x) * s ** (2 * _MICH_M)
             + np.sin(x) * (2 * _MICH_M) * s ** (2 * _MICH_M - 1) * c * (2.0 * i * x / np.pi))
    grad = -dterm
    return grad[0] if is_1d else grad


MICHALEWICZ: Dict[str, Any] = {
    "fn": michalewicz,
    "grad": michalewicz_grad,
    "name": "Michalewicz",
    "search_range": np.pi,
    "global_min": None,  # position depends on D; see best_known
    "global_min_val": None,
    "best_known": _MICH_BEST_KNOWN,
}


# --------------------------------------------------------------------------- #
# Zakharov (unimodal, plate-shaped)
# --------------------------------------------------------------------------- #

def zakharov(x: np.ndarray) -> Union[float, np.ndarray]:
    """Zakharov function. Global min f=0 at origin."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    i = np.arange(1, d + 1, dtype=np.float64)
    p = np.sum(0.5 * i * x, axis=1)
    out = np.sum(x**2, axis=1) + p**2 + p**4
    return float(out[0]) if is_1d else out


def zakharov_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Zakharov."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    i = np.arange(1, d + 1, dtype=np.float64)
    p = np.sum(0.5 * i * x, axis=1, keepdims=True)
    grad = 2.0 * x + i * p + 2.0 * i * p**3
    return grad[0] if is_1d else grad


ZAKHAROV: Dict[str, Any] = {
    "fn": zakharov,
    "grad": zakharov_grad,
    "name": "Zakharov",
    "search_range": 10.0,
    "global_min": 0.0,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Styblinski-Tang
# --------------------------------------------------------------------------- #

def styblinski_tang(x: np.ndarray) -> Union[float, np.ndarray]:
    """Styblinski-Tang. Global min f = -39.16599 * D at x = -2.903534."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    out = 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1)
    return float(out[0]) if is_1d else out


def styblinski_tang_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Styblinski-Tang."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (4.0 * x**3 - 32.0 * x + 5.0)


STYBLINSKI_TANG: Dict[str, Any] = {
    "fn": styblinski_tang,
    "grad": styblinski_tang_grad,
    "name": "Styblinski-Tang",
    "search_range": 5.0,
    "global_min": -2.903534,
    "global_min_val": -39.16599,  # per dimension
}


# --------------------------------------------------------------------------- #
# Dixon-Price (valley-shaped)
# --------------------------------------------------------------------------- #

def dixon_price(x: np.ndarray) -> Union[float, np.ndarray]:
    """Dixon-Price. Global min f=0 at x_i = 2^{-(2^i - 2)/2^i}."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    out = (x[:, 0] - 1.0) ** 2
    i = np.arange(2, d + 1, dtype=np.float64)
    out = out + np.sum(i * (2.0 * x[:, 1:]**2 - x[:, :-1]) ** 2, axis=1)
    return float(out[0]) if is_1d else out


def dixon_price_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Dixon-Price."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    grad = np.zeros_like(x)
    grad[:, 0] += 2.0 * (x[:, 0] - 1.0)
    if d > 1:
        i = np.arange(2, d + 1, dtype=np.float64)      # index i for terms 2..d
        g = 2.0 * x[:, 1:] ** 2 - x[:, :-1]             # g_i = 2 x_i^2 - x_{i-1}
        grad[:, :-1] += -2.0 * i * g                     # d/dx_{i-1}
        grad[:, 1:] += 8.0 * i * g * x[:, 1:]            # d/dx_i
    return grad[0] if is_1d else grad


DIXON_PRICE: Dict[str, Any] = {
    "fn": dixon_price,
    "grad": dixon_price_grad,
    "name": "Dixon-Price",
    "search_range": 10.0,
    "global_min": None,
    "global_min_val": 0.0,
}


# --------------------------------------------------------------------------- #
# Trid (bowl with strong interactions)
# --------------------------------------------------------------------------- #

def trid(x: np.ndarray) -> Union[float, np.ndarray]:
    """Trid. Global min f = -D(D+4)(D-1)/6 at x_i = i(D+1-i)."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    out = np.sum((x - 1.0) ** 2, axis=1)
    if d > 1:
        out = out - np.sum(x[:, 1:] * x[:, :-1], axis=1)
    return float(out[0]) if is_1d else out


def trid_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Trid."""
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x.reshape(1, -1)
    d = x.shape[1]
    grad = 2.0 * (x - 1.0)
    if d > 1:
        grad[:, :-1] -= x[:, 1:]
        grad[:, 1:] -= x[:, :-1]
    return grad[0] if is_1d else grad


TRID: Dict[str, Any] = {
    "fn": trid,
    "grad": trid_grad,
    "name": "Trid",
    "search_range": None,  # canonical range is [-D^2, D^2]; see helper below
    "global_min": None,
    "global_min_val": None,
}
# Trid optimum value helper (depends on D): -D(D+4)(D-1)/6; min pos x_i = i(D+1-i).
TRID["min_val_fn"] = lambda d: -d * (d + 4) * (d - 1) / 6.0
TRID["range_fn"] = lambda d: float(d * d)


EXTENDED_FUNCTIONS: Dict[str, Dict[str, Any]] = {
    "levy": LEVY,
    "michalewicz": MICHALEWICZ,
    "zakharov": ZAKHAROV,
    "styblinski_tang": STYBLINSKI_TANG,
    "dixon_price": DIXON_PRICE,
    "trid": TRID,
}


def get_search_range(fconfig: Dict[str, Any], dim: int) -> float:
    """Resolve (possibly dimension-dependent) search range for a function config."""
    if fconfig.get("range_fn") is not None:
        return float(fconfig["range_fn"](dim))
    r = fconfig.get("search_range")
    return float(r)
