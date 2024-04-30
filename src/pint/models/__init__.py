"""Pulsar timing models and tools for working with them.

The primary object for representing a timing model is
:class:`~pint.models.timing_model.TimingModel`. This contains a collection of
components (subclasses of :class:`~pint.models.timing_model.Component`), each of
which should have a collection of parameters (subclasses of
:class:`~pint.models.parameter.Parameter`). These parameters carry values
uncertainties and units and can be "frozen" or "free" to indicate whether
fitters (subclasses of :class:`~pint.fitter.Fitter`) should be allowed to modify
them. Normally timing models are created using
:func:`~pint.models.model_builder.get_model` but it is possible to construct and
modify them as python objects.

Binary models are implemented as Components, but they have somewhat special
handling; they are implemented by deriving from
:class:`~pint.models.stand_alone_psr_binaries.binary_generic.PSR_BINARY`, which
provides some of the infrastructure needed to implement them conveniently.
"""

from pint.models.absolute_phase import AbsPhase

# Import all standard model components here
from pint.models.astrometry import AstrometryEcliptic, AstrometryEquatorial
from pint.models.binary_bt import BinaryBT, BinaryBTPiecewise
from pint.models.binary_dd import BinaryDD, BinaryDDS, BinaryDDGR, BinaryDDH
from pint.models.binary_ddk import BinaryDDK
from pint.models.binary_ell1 import BinaryELL1, BinaryELL1H, BinaryELL1k
from pint.models.dispersion_model import (
    DispersionDM,
    DispersionDMX,
    DispersionJump,
    FDJumpDM,
)
from pint.models.dmwavex import DMWaveX
from pint.models.frequency_dependent import FD
from pint.models.glitch import Glitch
from pint.models.phase_offset import PhaseOffset
from pint.models.piecewise import PiecewiseSpindown
from pint.models.ifunc import IFunc
from pint.models.jump import DelayJump, PhaseJump
from pint.models.model_builder import get_model, get_model_and_toas
from pint.models.noise_model import EcorrNoise, PLRedNoise, ScaleToaError
from pint.models.solar_system_shapiro import SolarSystemShapiro
from pint.models.solar_wind_dispersion import SolarWindDispersion, SolarWindDispersionX
from pint.models.spindown import Spindown
from pint.models.fdjump import FDJump

# Import the main timing model classes
from pint.models.timing_model import DEFAULT_ORDER, TimingModel
from pint.models.troposphere_delay import TroposphereDelay
from pint.models.wave import Wave
from pint.models.wavex import WaveX

# Define a standard basic model
StandardTimingModel = TimingModel(
    "StandardTimingModel",
    [AstrometryEquatorial(), Spindown(), DispersionDM(), SolarSystemShapiro()],
)
# BTTimingModel = generate_timing_model("BTTimingModel",
#         (Astrometry, Spindown, Dispersion, SolarSystemShapiro, BT))
# DDTimingModel = generate_timing_model("DDTimingModel",
#         (Astrometry, Spindown, Dispersion, SolarSystemShapiro, DD))
