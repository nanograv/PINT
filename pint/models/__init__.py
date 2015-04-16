# __init__.py for PINT models/ directory
"""This module contains implementations of pulsar timing models.
"""
# Import the main timing model classes
from .timing_model import TimingModel, generate_timing_model

# Import all standard model components here
from .astrometry import Astrometry
from .dispersion import Dispersion
from .spindown import Spindown
from .dd import DD
from .bt import BT
from .solar_system_shapiro import SolarSystemShapiro
from .polycos import Polycos
# Define a standard basic model
StandardTimingModel = generate_timing_model("StandardTimingModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro))
BTTimingModel = generate_timing_model("BTTimingModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro, BT))
DDTimingModel = generate_timing_model("DDTimingModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro, DD))
PolycosModel = generate_timing_model("PolycosModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro, Polycos))
