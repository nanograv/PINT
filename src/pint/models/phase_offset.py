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
                units="",
                description="Overall phase offset.",
            )
        )
        self.phase_funcs_component += [self.offset_phase]
        self.register_deriv_funcs(self.d_offset_phase_d_OFFSET, "OFFSET")

    def offset_phase(self, toas, delay):
        return (
            (np.zeros(len(toas)) * self.OFFSET.quantity).to(u.dimensionless_unscaled)
            if toas.tzr
            else (np.ones(len(toas)) * self.OFFSET.quantity).to(
                u.dimensionless_unscaled
            )
        )

    def d_offset_phase_d_OFFSET(self, toas, param, delay):
        return (
            np.zeros(len(toas)) * u.Unit("")
            if toas.tzr
            else np.ones(len(toas)) * u.Unit("")
        )
