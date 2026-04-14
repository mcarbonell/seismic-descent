# Seismic Optimizer for PyTorch

This document explains the implementation of the Seismic Descent algorithm as a standard PyTorch optimizer.

## Overview

Traditional optimizers like SGD or Adam use stochasticity on a per-step basis (mini-batch sampling or weight noise). **SeismicOptimizer** introduces a dynamic, spatially correlated noise field over the entire parameter space of the model.

At every step, the optimizer adds the analytic gradient of this noise field to the loss gradient. The noise field vibrates with a specific frequency and amplitude schedule ("tremors"), which helps the model parameters "slide" out of sharp local minima into broader, more stable valleys.

## Mathematical Foundation

The noise field $\eta(w, t)$ is approximated using **Random Fourier Features (RFF)**:

$$ \eta(w, t) = \sqrt{\frac{2}{R}} \cdot A(t) \cdot \sum_{r=1}^R \cos(\omega_r \cdot w + t \cdot \text{drift}_r + \phi_r) $$

Where:
- $w$ is the flattened parameter vector of the neural network.
- $\omega_r$ represents random frequency vectors.
- $A(t)$ is the current tremor amplitude.

The update rule is:
$$ w_{t+1} = w_t - \gamma \cdot (\nabla_w L + \nabla_w \eta) $$

The gradient $\nabla_w \eta$ is calculated analytically:
$$ \nabla_w \eta = -\sqrt{\frac{2}{R}} \cdot A(t) \cdot \sum_{r=1}^R \sin(\omega_r \cdot w + \text{offset}) \cdot \omega_r $$

## Configuration Parameters

- `lr`: Standard learning rate.
- `noise_amplitude`: The initial scale of the earthquake tremors.
- `noise_decay`: How fast the tremors subside over time.
- `n_cycles`: The number of full seismic cycles (sine waves) to perform during training.
- `n_octaves`: Number of spatial noise scales (fractal noise).

## Scalability and Memory

The current implementation uses a **Global Feature Field**. It concatenates all model parameters into a single vector and applies a single RFF projection. 
- **Memory Cost**: $O(R \cdot M)$ where $R$ is the number of features (default 64) and $M$ is the number of model parameters.
- **VRAM Usage**: For small models like MNIST, it is negligible. For Large Language Models (LLMs), this would require significant VRAM optimization (e.g., block-wise RFF or sparse projections).

## Implementation

The optimizer is defined in [seismic_optimizer.py](../seismic_optimizer.py). 

### Example usage:
```python
from seismic_optimizer import SeismicOptimizer

optimizer = SeismicOptimizer(
    model.parameters(), 
    lr=0.01, 
    noise_amplitude=0.5, 
    n_cycles=10
)
```
