"""
Seismic Descent Optimizer for PyTorch.

Adds a spatially correlated Gaussian Random Field (RFF) to the neural loss landscape.
"""

import math
from typing import Callable, Optional

try:
    import torch
    from torch.optim import Optimizer
    _TORCH_AVAILABLE = True
except ImportError:
    Optimizer = object  # type: ignore
    _TORCH_AVAILABLE = False


class SeismicOptimizer(Optimizer):
    """
    Seismic Descent Optimizer for PyTorch.

    Perturbs the optimization landscape with continuous, spatially correlated
    noise generated via Random Fourier Features (RFF) and temporal wave octaves.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default=1e-3
        Learning rate.
    noise_amplitude : float, default=1.0
        Peak noise field amplitude multiplier.
    noise_decay : float, default=0.9999
        Per-step exponential decay factor for noise.
    n_cycles : int, default=10
        Target earthquake oscillation cycles.
    n_octaves : int, default=4
        Number of spatial noise octaves.
    R : int, default=64
        Number of random Fourier frequencies.
    seed : int, default=42
        Seed for reproducibility.
    adaptive_power : float, default=1.0
        Power exponent when scaling amplitude by loss value.
    adaptive_floor : float, default=0.0
        Additive baseline for adaptive amplitude.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        noise_amplitude: float = 1.0,
        noise_decay: float = 0.9999,
        n_cycles: int = 10,
        n_octaves: int = 4,
        R: int = 64,
        seed: int = 42,
        adaptive_power: float = 1.0,
        adaptive_floor: float = 0.0,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use SeismicOptimizer. "
                "Install it via `pip install torch` or `pip install seismic-descent[torch]`."
            )

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            noise_amplitude=noise_amplitude,
            noise_decay=noise_decay,
            n_cycles=n_cycles,
            n_octaves=n_octaves,
            R=R,
            adaptive_power=adaptive_power,
            adaptive_floor=adaptive_floor,
        )
        super().__init__(params, defaults)

        self.state["t"] = 0.0
        self.state["step"] = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        total_params = sum(p.numel() for group in self.param_groups for p in group["params"])
        self.state["total_dim"] = total_params

        self.state["OMEGAS"] = []
        self.state["PHIS"] = torch.rand((n_octaves, R), generator=self.rng) * 2 * math.pi
        self.state["DRIFTS"] = torch.rand((n_octaves, R), generator=self.rng) * 0.4 + 0.1

        for o in range(n_octaves):
            lengthscale = 2.0 * (2.0 ** o)
            omegas = torch.randn((R, total_params), generator=self.rng) / lengthscale
            self.state["OMEGAS"].append(omegas)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None, loss: Optional[float] = None):
        """Perform a single optimization step."""
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()

        t = self.state["t"]
        step = self.state["step"]

        group = self.param_groups[0]
        a0 = group["noise_amplitude"]

        if loss is not None:
            power = group.get("adaptive_power", 1.0)
            floor = group.get("adaptive_floor", 0.0)
            a0 = a0 * (loss ** power + floor)

        decay = group["noise_decay"] ** step
        n_cycles = group["n_cycles"]
        n_octaves = group["n_octaves"]
        r_freq = group["R"]

        # Temporal harmonic octave wave schedule
        f = 2.0
        t_tensor = torch.tensor(t, dtype=torch.float32)
        amp = a0 * decay * (
            torch.sin(t_tensor * f)
            + 0.5 * torch.sin(t_tensor * 2 * f)
            + 0.25 * torch.sin(t_tensor * 4 * f)
        )

        params_list = [p.view(-1) for grp in self.param_groups for p in grp["params"] if p.grad is not None]
        if not params_list:
            return loss_val

        x = torch.cat(params_list)
        total_noise_grad = torch.zeros_like(x)
        sqrt_2_r = math.sqrt(2.0 / r_freq)

        for o in range(n_octaves):
            omegas = self.state["OMEGAS"][o].to(x.device)
            phis = self.state["PHIS"][o].to(x.device)
            drifts = self.state["DRIFTS"][o].to(x.device)

            projections = torch.matmul(omegas, x)
            angles = projections + t * drifts + phis
            sines = torch.sin(angles)

            grad_contrib = torch.matmul(omegas.t(), sines)
            total_noise_grad -= amp * sqrt_2_r * grad_contrib
            amp *= 0.5

        offset = 0
        for grp in self.param_groups:
            lr = grp["lr"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                numel = p.numel()
                p_noise = total_noise_grad[offset : offset + numel].view_as(p)
                p.data.add_(p.grad + p_noise, alpha=-lr)
                offset += numel

        dt_noise = (n_cycles * math.pi) / 2000.0
        self.state["t"] += dt_noise
        self.state["step"] += 1

        return loss_val
