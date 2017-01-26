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
    is_register = True
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
        self.add_param(prefixParameter(name="GLF0D_1", units="Hz", value=0.0,
                       descriptionTplt=lambda x: "Decaying frequency change for"
                                                 " glitch %d " % x,
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
        # Check for required params, Check for Glitch numbers
        self.num_glitches = len(self.get_prefix_mapping('GLPH_'))
        glphparams = [x for x in self.params if x.startswith('GLPH_')]
        # check if glitch phase matches GLEP, GLF0, GLF1
        for glphnm in glphparams:
            glphpar = getattr(self, glphnm)
            idx = glphpar.index
            if not hasattr(self, 'GLEP_%d' % idx):  # Check if epoch is given.
                msg = 'Glicth Epoch is needed for Glicth %d.' % idx
                raise MissingParameter("Glitch", 'GLEP_%d' % idx, msg)
            # Check if the other glicth F0 and F1 exists,
            # if not add as zero parameter
            for glf in ["GLF0_", "GLF1_"]:
                if not hasattr(self, glf + '%d' % idx):
                    # The first glf0 and glf1 should be there
                    glf0exp = getattr(self, glf + '1')
                    self.add_param(glf0exp.new_index_prefix_param(idx))
                    getattr(self, glf + "%d" % idx).value = 0.0

        # Check the Decay Term.
        glf0dparams = [x for x in self.params if x.startswith('GLF0D_')]
        for glf0d in glf0dparams:
            df0d = getattr(self, glf0d)
            idx = df0d.index
            # Check if the decay belongs to a glitch
            if not hasattr(self, 'GLPH_%d' % idx):
                msg = 'Glitch frequency decay index has to match one glicth' \
                      ' phase index.'
                raise MissingParameter("Glitch", 'GLPH_%d' % idx, msg)

            if not hasattr(self, 'GLTD_%d' % idx):
                if df0d.value != 0.0:
                    msg = 'None zero GLTD_%d parameter needs a GLTD_%d' \
                          ' parameter' % (idx, idx)
                    raise MissingParameter("Glitch", 'GLTD_%d' % idx, msg)

    def glitch_phase(self, toas, delay):
        """Glitch phase function.
        delay is the time delay from the TOA to time of pulse emission
        at the pulsar, in seconds.
        returns an array of phases in long double
        """
        phs = numpy.zeros_like(toas, dtype=numpy.longdouble)
        glphnames = [x for x in self.params if x.startswith('GLPH_')]
        for glphnm in glphnames:
            glph = getattr(self, glphnm)
            dphs = glph.value
            idx = glph.index
            dF0 = getattr(self, "GLF0_%d" % idx).quantity
            dF1 = getattr(self, "GLF1_%d" % idx).quantity
            eph = time_to_longdouble(getattr(self, "GLEP_%d" % idx).value)
            dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
            dt = dt * u.s
            affected = dt > 0.0  # TOAs affected by glitch
            if hasattr(self, "GLF0D_%d" % idx):
                dF0D = getattr(self, "GLF0D_%d" % idx).value
                if dF0D != 0.0:
                    tau = getattr(self, "GLTD_%d" % idx).value * SECS_PER_DAY
                    decayterm = dF0D * tau * (1.0 - numpy.exp(- dt[affected]
                                              / tau))
                else:
                    decayterm = 0.0
            else:
                decayterm = 0.0
            phs[affected] += dphs + dt[affected] * \
                (dF0 + 0.5 * dt[affected] * dF1) + decayterm
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
