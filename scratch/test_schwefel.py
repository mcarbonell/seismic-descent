import numpy as np

def f(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

def grad_wrong(x):
    abs_x = np.abs(x)
    sqrt_abs = np.sqrt(abs_x)
    return -np.sin(sqrt_abs) - (x / (2 * sqrt_abs)) * np.cos(sqrt_abs)

def grad_right(x):
    abs_x = np.abs(x)
    sqrt_abs = np.sqrt(abs_x)
    return -np.sin(sqrt_abs) - (sqrt_abs / 2.0) * np.cos(sqrt_abs)

x = -10.0
eps = 1e-7
print("Finite diff:", (f(x+eps) - f(x-eps))/(2*eps))
print("Wrong:", grad_wrong(x))
print("Right:", grad_right(x))
