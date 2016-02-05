"""This module implements pulsar timing glitches.
"""
# glitch.py
# Defines glitch timing model class
import numpy
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
from .parameter import Parameter, MJDParameter
from .timing_model import TimingModel, MissingParameter
from ..phase import *
from ..utils import time_from_mjd_string, time_to_longdouble, str2longdouble, taylor_horner

# The maximum number of glitches we allow
maxglitches = 10

class Glitch(TimingModel):
    """This class provides glitches."""
    def __init__(self):
        super(Glitch, self).__init__()

        # The number of terms in the taylor exapansion of spin freq (F0...FN)
        self.num_glitches = maxglitches

        for ii in range(1, self.num_glitches + 1):
            self.add_param(Parameter(name="GLPH_%d"%ii,
                units="pulse phase", value=0.0,
                description="Phase change for glitch %d"%ii))
            self.add_param(MJDParameter(name="GLEP_%d"%ii,
                description="Epoch of glitch %d"%ii,
                time_scale='tdb'))
            self.add_param(Parameter(name="GLF0_%d"%ii,
                units="Hz", value=0.0,
                description="Permanent frequency change for glitch %d"%ii))
            self.add_param(Parameter(name="GLF1_%d"%ii,
                units="Hz/s", value=0.0,
                description="Permanent frequency-derivative change for glitch %d"%ii))
            self.add_param(Parameter(name="GLF0D_%d"%ii,
                units="Hz", value=0.0,
                description="Decaying frequency change for glitch %d"%ii))
            self.add_param(Parameter(name="GLTD_%d"%ii,
                units="day", value=0.0,
                description="Decay time constant for glitch %d"%ii))

        self.phase_funcs += [self.glitch_phase,]

    def setup(self):
        super(Glitch, self).setup()
        # Check for required params, at least for first glitch
        ps = ["GLPH_%d", "GLEP_%d", "GLF0_%d", "GLF1_%d", "GLF0D_%d", "GLTD_%d"]
        for ii in range(1, 2):
            for p in ps:
                term = p%ii
                if getattr(self, term).value is None:
                    raise MissingParameter("Glitch", term)
        # Remove all unused glitch params
        for ii in range(self.num_glitches, 0, -1):
            for p in ps:
                term = p%ii
                if hasattr(self, term) and \
                        getattr(self, term).value==0.0 and \
                        getattr(self, term).uncertainty is None:
                    delattr(self, term)
                    self.params.remove(term)
                else:
                    break
        # Add a shortcut for the number of spin terms there are
        self.num_glitches = ii

    def glitch_phase(self, toas, delay):
        """Glitch phase function.

        delay is the time delay from the TOA to time of pulse emission
          at the pulsar, in seconds.

        returns an array of phases in long double
        """
        phs = numpy.zeros_like(toas, dtype=numpy.longdouble)
        for ii in range(1, self.num_glitches + 1):
            dphs = getattr(self, "GLPH_%d"%ii).value
            dF0 = getattr(self, "GLF0_%d"%ii).value
            dF1 = getattr(self, "GLF1_%d"%ii).value
            eph = time_to_longdouble(getattr(self, "GLEP_%d"%ii).value)
            dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
            affected = dt > 0.0 # TOAs affected by glitch
            dF0D = getattr(self, "GLF0D_%d"%ii).value
            if dF0D != 0.0:
                tau = getattr(self, "GLTD_%d"%ii).value * SECS_PER_DAY
                decayterm = dF0D * tau * (1.0 - numpy.exp(-dt[affected]/tau))
            else:
                decayterm = 0.0
            # print ii, dphs, dF0, dt[:5], len(dt[affected]), len(phs[affected])
            phs[affected] += dphs + dt[affected] * \
                (dF0 + 0.5 * dt[affected] * dF1) + decayterm
        return phs

    def d_phase_d_GLF0_1(self, toas):
        """Calculate the derivative wrt GLF0_1"""
        eph = time_to_longdouble(getattr(self, "GLEP_%d"%ii).value)
        delay = self.delay(toas)
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dpdGLF0_1 = numpy.where(dt > 0.0, dt, 0.0)
        return dpdGLF0_1

    def d_phase_d_GLPH_1(self, toas):
        """Calculate the derivative wrt GLPH_1"""
        return numpy.zeros_like(toas)

    def d_phase_d_GLEP_1(self, toas):
        """Calculate the derivative wrt GLEP_1"""
        # ToDo
        return numpy.zeros_like(toas)
