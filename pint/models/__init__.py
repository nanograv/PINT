# __init__.py for PINT models/ directory
"""This module contains implementations of pulsar timing models.
"""
# Import the main timing model classes
from .timing_model import TimingModel

# Import all standard model components here
from .astrometry import AstrometryEquatorial, AstrometryEcliptic
from .binary_bt import BinaryBT
from .binary_dd import BinaryDD
from .binary_ell1 import BinaryELL1
from .dispersion_model import Dispersion, DispersionDMX
from .spindown import Spindown
from .frequency_dependent import FD
from .glitch import Glitch
from .jump import JumpDelay
from .solar_system_shapiro import SolarSystemShapiro
from .noise_model import ScaleToaError, EcorrNoise
from .model_builder import get_model

# Define a standard basic model
StandardTimingModel = TimingModel("StandardTimingModel",
          (AstrometryEquatorial(), Spindown(), Dispersion(), SolarSystemShapiro()))
# BTTimingModel = generate_timing_model("BTTimingModel",
#         (Astrometry, Spindown, Dispersion, SolarSystemShapiro, BT))
# DDTimingModel = generate_timing_model("DDTimingModel",
#         (Astrometry, Spindown, Dispersion, SolarSystemShapiro, DD))
