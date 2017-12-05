"""This module implements phase jumps.
"""
# phase_jump.py
# Defines PhaseJump timing model class
import numpy
import astropy.units as u
from .timing_model import DelayComponent, PhaseComponent, MissingParameter
from . import parameter as p
from pint import dimensionless_cycles


class DelayJump(DelayComponent):
    """This is a class to implement phase jumps
       NOTE: this component is disable for now, since we don't have any method
       to identify the phase jumps and delay jumps.
    """
    register = False
    def __init__(self):
        super(DelayJump, self).__init__()
        # TODO: In the future we should have phase jump as well.
        self.add_param(p.maskParameter(name = 'JUMP', units='second'))
        self.delay_funcs_component += [self.jump_delay,]
        self.category = 'delay_jump'

    def setup(self):
        super(DelayJump, self).setup()
        self.jumps = []
        for mask_par in self.get_params_of_type('maskParameter'):
            if mask_par.startswith('JUMP'):
                self.jumps.append(mask_par)
        for j in self.jumps:
            self.register_deriv_funcs(self.d_delay_d_jump, j)

    def jump_delay(self, toas, acc_delay=None):
        """This method returns the jump delays for each toas section collected by
        jump parameters. The delay value is determined by jump parameter value
        in the unit of seconds.
        """
        jdelay = numpy.zeros(len(toas))
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            mask = jump_par.select_toa_mask(toas)
            # NOTE: Currently parfile jump value has opposite sign with our
            # delay calculation.
            jdelay[mask] += -jump_par.value
        return jdelay * u.second

    def d_delay_d_jump(self, toas, jump_param, acc_delay=None):
        d_delay_d_j = numpy.zeros(len(toas))
        jpar = getattr(self, jump_param)
        mask = jpar.select_toa_mask(toas)
        d_delay_d_j[mask] = -1.0
        return d_delay_d_j * u.second/jpar.units

    def print_par(self):
        result = ''
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            result += jump_par.as_parfile_line()
        return result

class PhaseJump(PhaseComponent):
    """This is a class to implement phase jumps
    """
    register = True
    def __init__(self):
        super(PhaseJump, self).__init__()
        self.add_param(p.maskParameter(name = 'JUMP', units='second'))
        self.phase_funcs_component += [self.jump_phase,]
        self.category = 'phase_jump'

    def setup(self):
        super(PhaseJump, self).setup()
        self.jumps = []
        for mask_par in self.get_params_of_type('maskParameter'):
            if mask_par.startswith('JUMP'):
                self.jumps.append(mask_par)
        for j in self.jumps:
            self.register_deriv_funcs(self.d_phase_d_jump, j)

    def jump_phase(self, toas, delay):
        """This method returns the jump phase for each toas section collected by
        jump parameters. The phase value is determined by jump parameter times
        F0.
        """
        jphase = numpy.zeros(len(toas)) * (self.JUMP1.units * self.F0.units)
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            mask = jump_par.select_toa_mask(toas)
            # NOTE: Currently parfile jump value has opposite sign with our
            # phase calculation.
            jphase[mask] += jump_par.quantity * self.F0.quantity
        return jphase

    def d_phase_d_jump(self, toas, jump_param, delay):
        jpar = getattr(self, jump_param)
        d_phase_d_j = numpy.zeros(len(toas))
        mask = jpar.select_toa_mask(toas)
        d_phase_d_j[mask] = self.F0.value
        with u.set_enabled_equivalencies(dimensionless_cycles):
            return (d_phase_d_j * self.F0.units).to(u.cycle/u.second)

    def print_par(self):
        result = ''
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            result += jump_par.as_parfile_line()
        return result
