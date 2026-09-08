"""
Seismic Descent: Multi-dimensional global optimization via spatially correlated landscape deformation.
"""

from seismic_descent.core import SeismicSwarm, seismic_swarm
from seismic_descent.rff import RandomFourierFeatures
from seismic_descent.orf import OrthogonalRandomFeatures
from seismic_descent.lissajous import LissajousWaveField, OrthogonalLissajousWaveField
from seismic_descent.swarm_gravity import SeismicSwarmGravity, seismic_swarm_gravity
from seismic_descent.anisotropic_hmc import SeismicAnisotropicHMC, seismic_anisotropic_hmc
from seismic_descent.champion_v23 import SeismicChampionV23, seismic_champion_v23
from seismic_descent.functions import ALL_FUNCTIONS

__version__ = "0.23.0"

__all__ = [
    "SeismicSwarm",
    "seismic_swarm",
    "RandomFourierFeatures",
    "OrthogonalRandomFeatures",
    "LissajousWaveField",
    "OrthogonalLissajousWaveField",
    "SeismicSwarmGravity",
    "seismic_swarm_gravity",
    "SeismicAnisotropicHMC",
    "seismic_anisotropic_hmc",
    "SeismicChampionV23",
    "seismic_champion_v23",
    "ALL_FUNCTIONS",
    "SeismicOptimizer",
    "__version__",
]

try:
    from seismic_descent.torch_optimizer import SeismicOptimizer
except ImportError:
    pass
