"""Explicit phase offset"""

from pint.models.timing_model import PhaseComponent
from pint.models.parameter import floatParameter
from astropy import units as u
import numpy as np


class PhaseOffset(PhaseComponent):
    """Explicit pulse phase offset.

    Parameters supported:

    .. paramtable::
        :class: pint.models.jump.PhaseJump
    """

    register = True
    category = "phase_offset"

    def __init__(self):
        super().__init__()
        self.add_param(
            floatParameter(
                name="OFFSET",
                units="radian",
                description="Overall phase offset.",
            )
        )
        self.phase_funcs_component += [self.offset_phase]
        self.register_deriv_funcs(self.d_phase_offset_d_OFFSET, "OFFSET")

    def offset_phase(self, toas):
        return np.ones(len(toas)) * self.OFFSET.quantity

    def d_phase_offset_d_OFFSET(self, toas):
        return np.ones(len(toas)) * u.Unit("")
