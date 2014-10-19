"""This module implements a simple model of a constant dispersion measure."""
# dispersion.py
# Simple (constant) ISM dispersion measure
from warnings import warn
from .parameter import Parameter
from .timing_model import TimingModel
import astropy.units as u

# The units on this are not completely correct
# as we don't really use the "pc cm^3" units on DM.
# But the time and freq portions are correct
DMconst = 1.0/2.41e-4 * u.MHz * u.MHz * u.s

class Dispersion(TimingModel):
    """This class provides a timing model for a simple constant
    dispersion measure.
    """

    def __init__(self):
        super(Dispersion, self).__init__()

        self.add_param(Parameter(name="DM",
            units="pc cm^-3", value=0.0,
            description="Dispersion measure"))
        self.delay_funcs += [self.dispersion_delay,]

    def setup(self):
        super(Dispersion, self).setup()

    def dispersion_delay(self, toas):
        """Return the dispersion delay at each toa."""
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']
        return self.DM.value * DMconst / bfreq**2
