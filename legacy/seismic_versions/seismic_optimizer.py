import torch
from torch.optim import Optimizer
import math

class SeismicOptimizer(Optimizer):
    """
    Seismic Descent Optimizer for PyTorch.
    
    Adds a spatially correlated noise field (approximated via Random Fourier Features)
    to the objective function. The analytic gradient of this noise field is added 
    to the loss gradient during the update.
    """
    def __init__(self, params, lr=1e-3, noise_amplitude=1.0, noise_decay=0.9999, 
                 n_cycles=10, n_octaves=4, R=64, seed=42,
                 adaptive_power=1.0, adaptive_floor=0.0):
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
            adaptive_floor=adaptive_floor
        )
        super(SeismicOptimizer, self).__init__(params, defaults)
        
        self.state['t'] = 0.0
        self.state['step'] = 0
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        
        # Initialize RFF features globally for all parameters
        # We find the total number of parameters first
        total_params = 0
        for group in self.param_groups:
            for p in group['params']:
                total_params += p.numel()
        
        self.state['total_dim'] = total_params
        
        # We pre-generate OMEGAS, PHIS, DRIFTS
        # To avoid massive VRAM usage, we only generate OMEGAS once 
        # But for very large models, this will be the bottleneck.
        R = defaults['R']
        n_octaves = defaults['n_octaves']
        
        self.state['OMEGAS'] = []
        self.state['PHIS'] = torch.rand((n_octaves, R), generator=self.rng) * 2 * math.pi
        self.state['DRIFTS'] = torch.rand((n_octaves, R), generator=self.rng) * 0.4 + 0.1
        
        for o in range(n_octaves):
            lengthscale = 2.0 * (2.0 ** o)
            # Omegas: (R, total_dim)
            # Note: For million-param models, this is the part that will eat VRAM.
            omegas = torch.randn((R, total_params), generator=self.rng) / lengthscale
            self.state['OMEGAS'].append(omegas)

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()

        # Update time/amplitude
        t = self.state['t']
        step = self.state['step']
        
        # Hyperparameters (taking from first group for simplicity)
        group = self.param_groups[0]
        A0 = group['noise_amplitude']
        
        # Adaptive Amplitude scaling based on loss
        # If loss is provided (item), we scale initial A0
        if loss is not None:
            power = group.get('adaptive_power', 1.0)
            floor = group.get('adaptive_floor', 0.0)
            A0 = A0 * (loss ** power + floor)
            
        decay = group['noise_decay'] ** step
        n_cycles = group['n_cycles']
        n_octaves = group['n_octaves']
        R = group['R']
        lr = group['lr']
        
        # Temporal Octave schedule
        f = 2.0
        amp = A0 * decay * (
            torch.sin(torch.tensor(t * f)) + 
            0.5 * torch.sin(torch.tensor(t * 2 * f)) + 
            0.25 * torch.sin(torch.tensor(t * 4 * f))
        )
        
        # 1. Flatten all parameters to a single X vector
        params_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params_list.append(p.view(-1))
        
        if not params_list:
            return loss
            
        X = torch.cat(params_list) # (total_dim,)
        
        # 2. Compute RFF Noise Gradient analytically
        # grad_noise = -amp * sqrt(2/R) * sum(sin(omega@X + phis + t*drifts) * omega)
        total_noise_grad = torch.zeros_like(X)
        sqrt_2_R = math.sqrt(2.0 / R)
        
        for o in range(n_octaves):
            omegas = self.state['OMEGAS'][o].to(X.device)
            phis   = self.state['PHIS'][o].to(X.device)
            drifts = self.state['DRIFTS'][o].to(X.device)
            
            # projections: (R,)
            projections = torch.matmul(omegas, X)
            angles = projections + t * drifts + phis
            
            sines = torch.sin(angles) # (R,)
            
            # grad_contrib: (total_dim,) = (total_dim, R) @ (R,)
            grad_contrib = torch.matmul(omegas.t(), sines)
            total_noise_grad -= amp * sqrt_2_R * grad_contrib
            
            amp *= 0.5
            
        # 3. Apply updates
        # Scatter the total noise grad back to param.grad or update parameters directly
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                numel = p.numel()
                p_noise_grad = total_noise_grad[offset : offset + numel].view_as(p)
                
                # Update rule: w = w - lr * (grad_loss + p_noise_grad)
                p.data.add_(p.grad + p_noise_grad, alpha=-group['lr'])
                
                offset += numel
        
        # 4. Advance state
        # Assume dt based on n_cycles and 1000 steps per cycle approx or fixed dt
        # For simplicity, let's use a fixed dt that completes n_cycles in 10000 steps
        # or we could make it a param. Let's use the one from the original algo.
        dt_noise = (n_cycles * math.pi) / 2000.0 # Standard 2k steps 
        self.state['t'] += dt_noise
        self.state['step'] += 1
        
        return loss
