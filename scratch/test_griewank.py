import numpy as np

def fn(X):
    D = X.shape[1]
    sum_term = np.sum(X**2, axis=1) / 4000.0
    indices = np.arange(1, D + 1)
    prod_term = np.prod(np.cos(X / np.sqrt(indices)), axis=1)
    return sum_term - prod_term + 1.0

def grad_old(X):
    N, D = X.shape
    indices = np.arange(1, D + 1)
    sqrt_idx = np.sqrt(indices)
    cos_terms = np.cos(X / sqrt_idx)
    full_prod = np.prod(cos_terms, axis=1, keepdims=True)
    safe_cos = np.where(np.abs(cos_terms) < 1e-15, 1e-15 * np.sign(cos_terms + 1e-20), cos_terms)
    prod_without_i = full_prod / safe_cos
    sin_terms = np.sin(X / sqrt_idx)
    grad = X / 2000.0 + prod_without_i * sin_terms / sqrt_idx
    return grad

def grad_fixed(X):
    N, D = X.shape
    indices = np.arange(1, D + 1)
    sqrt_idx = np.sqrt(indices)
    
    cos_terms = np.cos(X / sqrt_idx)
    
    left = np.cumprod(cos_terms, axis=1)
    right = np.cumprod(cos_terms[:, ::-1], axis=1)[:, ::-1]
    
    prod_without_i = np.ones_like(cos_terms)
    prod_without_i[:, 1:] *= left[:, :-1]
    prod_without_i[:, :-1] *= right[:, 1:]
    
    sin_terms = np.sin(X / sqrt_idx)
    grad = X / 2000.0 + prod_without_i * sin_terms / sqrt_idx
    return grad

np.random.seed(42)
X = np.random.uniform(-10, 10, (1, 5))
eps = 1e-7

numeric = np.zeros_like(X)
for i in range(5):
    xp = X.copy()
    xp[0, i] += eps
    xm = X.copy()
    xm[0, i] -= eps
    numeric[0, i] = (fn(xp) - fn(xm)) / (2 * eps)

print("Numeric:", numeric)
print("Old:", grad_old(X))
print("Fixed:", grad_fixed(X))

X_zero = np.zeros((1, 5))
X_zero[0, 0] = np.pi / 2.0  # cos(x / 1) will be ~0
print("\nZero test Numeric:")
numeric_z = np.zeros_like(X_zero)
for i in range(5):
    xp = X_zero.copy()
    xp[0, i] += eps
    xm = X_zero.copy()
    xm[0, i] -= eps
    numeric_z[0, i] = (fn(xp) - fn(xm)) / (2 * eps)
print(numeric_z)
print("Zero test Old:", grad_old(X_zero))
print("Zero test Fixed:", grad_fixed(X_zero))
