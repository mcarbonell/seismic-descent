"""Unit tests for PyTorch SeismicOptimizer."""

import pytest

try:
    import torch
    import torch.nn as nn
    from seismic_descent.torch_optimizer import SeismicOptimizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_seismic_optimizer_convergence():
    torch.manual_seed(42)

    # Simple linear model fitting y = 2x + 1
    model = nn.Linear(1, 1)
    optimizer = SeismicOptimizer(
        model.parameters(),
        lr=0.05,
        noise_amplitude=0.1,
        n_cycles=5,
        seed=1,
    )
    criterion = nn.MSELoss()

    x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_data = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

    initial_loss = criterion(model(x_data), y_data).item()

    for _ in range(200):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step(loss=loss.item())

    final_loss = criterion(model(x_data), y_data).item()
    assert final_loss < initial_loss * 0.1
