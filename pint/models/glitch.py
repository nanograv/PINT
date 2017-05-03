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
    taylor_horner, split_prefixed_name

# The maximum number of glitches we allow
maxglitches = 10  # Have not use this one in the new version.


class Glitch(TimingModel):
    """This class provides glitches."""
    register = True
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
        self.print_par_func = 'print_par_glitch'

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
                    self.add_param(param0.new_param(idx))
                    getattr(self, param + '%d' % idx).value = 0.0
                self.register_deriv_funcs(getattr(self, \
                     'd_phase_d_'+param[0:-1]), 'phase', param + '%d' % idx)

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

    def print_par_glitch(self):
        result = ''
        for idx in set(self.glitch_indices):
            for param in ['GLEP_',] + self.glitch_prop:
                par = getattr(self, param + '%d'%idx)
                result += par.as_parfile_line()
        return result

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
            eph = glep.value
            idx = glep.index
            dphs = getattr(self, "GLPH_%d" % idx).quantity
            dF0 = getattr(self, "GLF0_%d" % idx).quantity
            dF1 = getattr(self, "GLF1_%d" % idx).quantity
            dF2 = getattr(self, "GLF2_%d" % idx).quantity
            dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
            dt = dt * u.second
            affected = dt > 0.0  # TOAs affected by glitch
            # decay term
            dF0D = getattr(self, "GLF0D_%d" % idx).quantity
            if dF0D != 0.0:
                tau = getattr(self, "GLTD_%d" % idx).quantity
                decayterm = dF0D * tau * (1.0 - numpy.exp(- (dt[affected]
                                          / tau).to(u.Unit(""))))
            else:
                decayterm = 0.0

            phs[affected] += dphs + dt[affected] * \
                (dF0 + 0.5 * dt[affected] * dF1 + \
                 1./6. * dt[affected]*dt[affected] * dF2) + decayterm
        return phs

    def d_phase_d_GLPH(self, toas, param, delay):
        """Calculate the derivative wrt GLPH_"""
        p, ids, idv = split_prefixed_name(param)
        if p !=  'GLPH_':
            raise ValueError("Can not calculate d_phase_d_GLPH with respect to %s." % param)
        eph = time_to_longdouble(getattr(self, "GLEP_" + ids).value)
        par_GLPH = getattr(self, param)
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dt = dt * u.second
        affected = numpy.where(dt > 0.0)[0]
        dpdGLPH = numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLPH.units
        dpdGLPH[affected] += 1.0 * u.Unit("")/par_GLPH.units
        return dpdGLPH

    def d_phase_d_GLF0(self, toas, param, delay):
        """
        Calculate the derivative wrt GLF0_
        """
        p, ids, idv = split_prefixed_name(param)
        if p !=  'GLF0_':
            raise ValueError("Can not calculate d_phase_d_GLF0 with respect to %s." % param)
        eph = time_to_longdouble(getattr(self, "GLEP_" + ids).value)
        par_GLF0 = getattr(self, param)
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dt = dt * u.second
        affected = numpy.where(dt > 0.0)[0]
        dpdGLF0 = numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLF0.units
        dpdGLF0[affected] = dt[affected]
        return dpdGLF0

    def d_phase_d_GLF1(self, toas,  param, delay):
        """Calculate the derivative wrt GLF1"""
        p, ids, idv = split_prefixed_name(param)
        if p !=  'GLF1_':
            raise ValueError("Can not calculate d_phase_d_GLF1 with respect to %s." % param)
        eph = time_to_longdouble(getattr(self, "GLEP_" + ids).value)
        par_GLF1 = getattr(self, param)
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dt = dt * u.second
        affected = numpy.where(dt > 0.0)[0]
        dpdGLF1 = numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLF1.units
        dpdGLF1[affected] += numpy.longdouble(0.5) * dt[affected] * dt[affected]
        return dpdGLF1

    def d_phase_d_GLF2(self, toas,  param, delay):
        """Calculate the derivative wrt GLF1"""
        p, ids, idv = split_prefixed_name(param)
        if p !=  'GLF2_':
            raise ValueError("Can not calculate d_phase_d_GLF2 with respect to %s." % param)
        eph = time_to_longdouble(getattr(self, "GLEP_" + ids).value)
        par_GLF2 = getattr(self, param)
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dt = dt * u.second
        affected = numpy.where(dt > 0.0)[0]
        dpdGLF2 = numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLF2.units
        dpdGLF2[affected] += numpy.longdouble(1.0)/6.0 * dt[affected] * dt[affected] * dt[affected]
        return dpdGLF2

    def d_phase_d_GLF0D(self, toas, param, delay):
        """Calculate the derivative wrt GLF0D
        """
        p, ids, idv = split_prefixed_name(param)
        if p !=  'GLF0D_':
            raise ValueError("Can not calculate d_phase_d_GLF0D with respect to %s." % param)
        eph = time_to_longdouble(getattr(self, "GLEP_" + ids).value)
        par_GLF0D = getattr(self, param)
        tau = getattr(self, "GLTD_%d" % idv).quantity
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dt = dt * u.second
        affected = numpy.where(dt > 0.0)[0]
        dpdGLF0D = numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLF0D.units
        dpdGLF0D[affected] += tau * (numpy.longdouble(1.0) - numpy.exp(- dt[affected]
                                  / tau))
        return dpdGLF0D

    def d_phase_d_GLTD(self, toas, param, delay):
        """Calculate the derivative wrt GLF0D
        """
        p, ids, idv = split_prefixed_name(param)
        if p !=  'GLTD_':
            raise ValueError("Can not calculate d_phase_d_GLF0D with respect to %s." % param)
        eph = time_to_longdouble(getattr(self, "GLEP_" + ids).value)
        par_GLTD = getattr(self, param)
        if par_GLTD.value == 0.0:
            return numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLTD.units
        glf0d = getattr(self, 'GLF0D_'+ids).quantity
        tau = par_GLTD.quantity
        dt = (toas['tdbld'] - eph) * SECS_PER_DAY - delay
        dt = dt * u.second
        affected = numpy.where(dt > 0.0)[0]
        dpdGLTD = numpy.zeros(len(toas), dtype=numpy.longdouble) * u.Unit("")/par_GLTD.units
        dpdGLTD[affected] += glf0d * (numpy.longdouble(1.0) - \
                             numpy.exp(- dt[affected] / tau)) + \
                             glf0d * tau * (-numpy.exp(- dt[affected] / tau)) * \
                             dt[affected] / (tau * tau)
        return dpdGLTD
