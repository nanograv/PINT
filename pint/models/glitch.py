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
from .parameter import Parameter, MJDParameter, prefixParameter
from .timing_model import TimingModel, MissingParameter
from ..phase import *
from ..utils import time_from_mjd_string, time_to_longdouble, str2longdouble, \
    taylor_horner

# The maximum number of glitches we allow
maxglitches = 10  # Have not use this one in the new version.


class Glitch(TimingModel):
    """This class provides glitches."""
    def __init__(self):
        super(Glitch, self).__init__()

        self.add_param(prefixParameter(name="GLPH_1", units="pulse phase",
                       value=0.0,
                       descriptionTplt=lambda x: "Phase change for glitch %d"
                                                 % x,
                       unitTplt=lambda x: 'pulse phase',
                       type_match='float'))
        self.add_param(prefixParameter(name="GLEP_1", units='day',
                       descriptionTplt=lambda x: "Epoch of glitch %d" % x,
                       unitTplt=lambda x: 'day',
                       type_match='MJD', time_scale='tdb'))
        self.add_param(prefixParameter(name="GLF0_1", units="Hz", value=0.0,
                       descriptionTplt=lambda x: "Permanent frequency change"
                                                 " for glitch %d" % x,
                       unitTplt=lambda x: 'Hz',
                       type_match='float'))
        self.add_param(prefixParameter(name="GLF1_1", units="Hz/s", value=0.0,
                       descriptionTplt=lambda x: "Permanent frequency-"
                                                 "derivative change for glitch"
                                                 " %d " % x,
                       unitTplt=lambda x: 'Hz/s'))
        self.add_param(prefixParameter(name="GLF2_1", units="Hz/s^2", value=0.,
                       descriptionTplt=lambda x: "Permanent second frequency-"
                                                 "derivative change for glitch"
                                                 " %d " % x,
                       unitTplt=lambda x: 'Hz/s^2'))
        self.add_param(prefixParameter(name="GLF0D_1", units="Hz", value=0.0,
                       descriptionTplt=lambda x: "Decaying frequency change "
                                                 "for glitch %d " % x,
                       unitTplt=lambda x: 'Hz',
                       type_match='float'))

        self.add_param(prefixParameter(name="GLTD_1",
                       units="day", value=0.0,
                       descriptionTplt=lambda x: "Decay time constant for"
                                                 " glitch %d" % x,
                       unitTplt=lambda x: 'day',
                       type_match='float'))
        self.phase_funcs += [self.glitch_phase]

    def setup(self):
        super(Glitch, self).setup()
        # Check for required glitch epochs, set not specified parameters to 0
        self.glitch_prop = ['GLPH_', 'GLF0_', 'GLF1_', 'GLF2_',
                            'GLF0D_', 'GLTD_']
        self.glitch_indices = [getattr(self, y).index for x in self.glitch_prop
                               for y in self.params if x in y]
        for idx in set(self.glitch_indices):
            if not hasattr(self, 'GLEP_%d' % idx):
                msg = 'Glicth Epoch is needed for Glicth %d.' % idx
                raise MissingParameter("Glitch", 'GLEP_%d' % idx, msg)
            for param in self.glitch_prop:
                if not hasattr(self, param + '%d' % idx):
                    param0 = getattr(self, param + '1')
                    self.add_param(param0.new_index_prefix_param(idx))
                    getattr(self, param + '%d' % idx).value = 0.0

        # Check the Decay Term.
        glf0dparams = [x for x in self.params if x.startswith('GLF0D_')]
        for glf0dnm in glf0dparams:
            glf0d = getattr(self, glf0dnm)
            idx = glf0d.index
            if glf0d.value != 0.0 and \
                    getattr(self, "GLTD_%d" % idx).value == 0.0:
                msg = "None zero GLF0D_%d parameter needs a none" \
                      " zero GLTD_%d parameter" % (idx, idx)
                raise MissingParameter("Glitch", 'GLTD_%d' % idx, msg)

    def glitch_phase(self, toas, delay):
        """Glitch phase function.
        delay is the time delay from the TOA to time of pulse emission
        at the pulsar, in seconds.
        returns an array of phases in long double
        """
        phs = numpy.zeros_like(toas, dtype=numpy.longdouble)
        glepnames = [x for x in self.params if x.startswith('GLEP_')]
        for glepnm in glepnames:
            glep = getattr(self, glepnm)
            eph = time_to_longdouble(glep.value)
            idx = glep.index
            dphs = getattr(self, "GLPH_%d" % idx).value
            dF0 = getattr(self, "GLF0_%d" % idx).value
            dF1 = getattr(self, "GLF1_%d" % idx).value
            dF2 = getattr(self, "GLF2_%d" % idx).value
            dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
            dt = dt * u.s
            affected = dt > 0.0  # TOAs affected by glitch
            # decay term
            dF0D = getattr(self, "GLF0D_%d" % idx).value
            if dF0D != 0.0:
                tau = getattr(self, "GLTD_%d" % idx).value
                decayterm = dF0D * tau * (1.0 - numpy.exp(- dt[affected]
                                          / tau))
            else:
                decayterm = 0.0

            phs[affected] += dphs + dt[affected] * \
                (dF0 + 0.5 * dt[affected] * dF1 +
                 1./6. * dt[affected]*dt[affected] * dF2) + decayterm
        return phs

    def d_phase_d_GLF0_1(self, toas):
        """Calculate the derivative wrt GLF0_1"""
        eph = time_to_longdouble(getattr(self, "GLEP_1").value)
        delay = self.delay(toas)
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dpdGLF0_1 = numpy.where(dt > 0.0, dt, 0.0)
        return dpdGLF0_1

    def d_phase_d_GLPH_1(self, toas):
        """Calculate the derivative wrt GLPH_1"""
        return numpy.zeros_like(toas['tdbld'])

    def d_phase_d_GLEP_1(self, toas):
        """Calculate the derivative wrt GLEP_1"""
        # ToDo
        return numpy.zeros_like(toas['tdbld'])
