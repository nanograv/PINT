"""This module implements phase jumps.
"""
# phase_jump.py
# Defines PhaseJump timing model class
import numpy
import astropy.units as u
from .timing_model import TimingModel, MissingParameter, module_info
import parameter as p


class JumpDelay(TimingModel):
    """This is a class to implement phase jumps
    """
    def __init__(self):
        super(JumpDelay, self).__init__()
        # TODO: In the future we should have phase jump as well.
        self.requires = {'TOA': [], 'freq': []}
        self.provides = {'TOA': ('', None), 'freq': ('', None)}
        self.add_param(p.maskParameter(name = 'JUMP', units='second'))
        self.delay_funcs += [self.jump_delay,]
    def setup(self):
        super(JumpDelay, self).setup()
        self.jumps = []
        for mask_par in self.param_register['maskParameter']:
            if mask_par.startswith('JUMP'):
                self.jumps.append(mask_par)

    def jump_delay(self, toas):
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
        return jdelay
