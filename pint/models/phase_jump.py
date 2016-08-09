"""This module implements phase jumps.
"""
# phase_jump.py
# Defines PhaseJump timing model class
import numpy
import astropy.units as u
from .timing_model import TimingModel, MissingParameter
import parameter as p


class PhaseJump(TimingModel):
    def __init__(self):
        super(PhaseJump, self).__init__()
        self.add_param(p.maskParameter(name = 'JUMP'))
        self.phase_funcs += [self.jump_phase,]
    def setup(self):
        super(PhaseJump, self).setup()
        self.jumps = []
        for mask_par in self.param_register['maskParameter']:
            if mask_par.startswith('JUMP'):
                self.jumps.append(mask_par)

    def jump_phase(self, toas, delay):
        phase = numpy.zeros(len(toas))
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            mask = jump_par.select_toa_mask(toas)
            phase[mask] += jump_par.num_value * self.F0.num_value 
        return phase
