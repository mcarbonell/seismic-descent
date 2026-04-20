import numpy as np
import math

class SeismicOptimizerV23:
    """
    Seismic Descent V23 - Vector-Wave Edition.
    Highly vectorized noise-based global optimizer.
    """
    def __init__(self, dim, lr=0.1, octaves=4, persistence=0.5, 
                 initial_scale=1.0, total_steps=1000, seed=None):
        self.dim = dim
        self.lr0 = lr
        self.octaves = octaves
        self.persistence = persistence
        self.initial_scale = initial_scale
        self.total_steps = total_steps
        self.t = 0
        self.rng = np.random.default_rng(seed)
        
        # Pre-generate octave frequencies and amplitudes (Vectorized)
        self.freqs = 2 ** np.arange(octaves)
        self.amps = persistence ** np.arange(octaves)
        
        # Seismic state: coordinate offsets for the noise field
        self.offsets = self.rng.standard_normal((octaves, dim)).astype(np.float32)

    def _get_seismic_signal(self, x):
        """
        Calculates the aggregate noise signal across all octaves using vectorization.
        This replaces slow per-octave loops.
        """
        # x: (dim,)
        # freqs: (octaves,)
        # offsets: (octaves, dim)
        
        # Calculate wave phases: (octaves, dim)
        # Using a vectorized sine-based approximation of Perlin-like noise
        phases = (x * self.freqs[:, np.newaxis]) + self.offsets
        noise_vectors = np.sin(phases) * self.amps[:, np.newaxis]
        
        # Aggregate signal: (dim,)
        return np.sum(noise_vectors, axis=0)

    def step(self, f, x):
        self.t += 1
        
        # Cooling schedule (Adaptive Morphing)
        progress = min(self.t / self.total_steps, 1.0)
        lr = self.lr0 * (1.0 - progress)
        scale = self.initial_scale * (1.0 - progress)
        
        # 1. Probe the noise field at current location
        signal = self._get_seismic_signal(x)
        
        # 2. Estimate local slope using the seismic signal as a basis
        delta = 1e-4
        fp = f(x + signal * delta)
        fm = f(x - signal * delta)
        
        # Finite difference along the seismic vector
        slope = (fp - fm) / (2.0 * delta)
        
        # 3. Update position: move AGAINST the slope along the seismic wave
        # This allows jumping over local minima if the noise frequency matches the landscape
        step_vector = -np.sign(slope) * signal * lr
        x_new = x + step_vector
        
        # 4. Jitter the offsets slightly to keep the wave "alive"
        self.offsets += self.rng.standard_normal((self.octaves, self.dim)) * 0.01
        
        return x_new, 2 # used 2 evaluations
