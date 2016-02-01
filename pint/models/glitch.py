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
from .parameter import Parameter, MJDParameter,prefixParameter
from .timing_model import TimingModel, MissingParameter
from ..phase import *
from ..utils import time_from_mjd_string, time_to_longdouble, str2longdouble, taylor_horner

# The maximum number of glitches we allow
maxglitches = 10 # Have not use this one in the new version.

class Glitch(TimingModel):
    """This class provides glitches."""
    def __init__(self):
        super(Glitch, self).__init__()

        # The number of terms in the taylor exapansion of spin freq (F0...FN)

        self.add_param(prefixParameter(name = "GLPH_1",
            units="pulse phase", value=0.0,
            descriptionTplt = lambda x:"Phase change for glitch %d"%x,))
        self.add_param(prefixParameter(name="GLEP_1",
            descriptionTplt=lambda x:"Epoch of glitch %d"%x,
            units= 'day',
            parse_value=lambda x: time_from_mjd_string(x, scale='tdb')))
        self.add_param(prefixParameter(name="GLF0_1",
            units="Hz", value=0.0,
            descriptionTplt=lambda x:"Permanent frequency change for glitch %d"%x))
        self.add_param(prefixParameter(name="GLF1_1",
            units="Hz/s", value=0.0,
            descriptionTplt=lambda x:"Permanent frequency-derivative change for glitch %d"%x))
        self.add_param(prefixParameter(name="GLF0D_1",
            units="Hz", value=0.0,
            descriptionTplt=lambda x:"Decaying frequency change for glitch %d"%x))

        self.add_param(prefixParameter(name="GLTD_1",
            units="day", value=0.0,
            descriptionTplt=lambda x:"Decay time constant for glitch %d"%x))

        self.phase_funcs += [self.glitch_phase,]

    def setup(self):
        super(Glitch, self).setup()
        # Check for required params, Check for Glitch numbers
        self.num_glitches = self.num_prefix_params['GLPH_']
        # check if glitch phase is continuous
        ps = ["GLPH_", "GLEP_", "GLF0_", "GLF1_"]
        for ii in range(1,self.num_glitches+1):
            glcname = 'GLPH_%d'%ii
            if glcname not in self.params:
                msg = 'Index of glitch prefix parameter should be continuous.'
                raise MissingParameter("Glitch", glcname,msg)
            # Check if the other glicth parameter exist
            for p in ps:
                glcparam = p+'%d'%ii
                if glcparam not in self.params:
                    raise MissingParameter("Glitch", glcparam)


        # Check the Decay Term. If not match number of glitch. If not in parfile. add zeros. 
        ps2 = [ "GLF0D_", "GLTD_"]
        for p in ps2:
            pname = [x for x in self.params if x.startswith(p) ]
            for ii in range(1,self.num_glitches+1):
                if p+"%d"%ii not in self.params:
                    pfxpar = getattr(self,pname[0])
                    self.add_param(pfxpar.new_index_prefix_param(ii))
                    getattr(self,p+"%d"%ii).value = 0.0


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
