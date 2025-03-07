"""Explicit phase offset"""

import numpy as np
from astropy import units as u

from pint.models.parameter import floatParameter
from pint.models.timing_model import PhaseComponent


class PhaseOffset(PhaseComponent):
    """Explicit pulse phase offset between physical TOAs and the TZR TOA.
    See `examples/phase_offset_example.py` for example usage.

    Parameters supported:

    .. paramtable::
        :class: pint.models.phase_offset.PhaseOffset
    """

    register = True
    category = "phase_offset"

    def __init__(self):
        super().__init__()
        self.add_param(
            floatParameter(
                name="PHOFF",
                value=0.0,
                units="",
                description="Overall phase offset between physical TOAs and the TZR TOA.",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.phase_funcs_component += [self.offset_phase]
        self.register_deriv_funcs(self.d_offset_phase_d_PHOFF, "PHOFF")

    def offset_phase(self, toas, delay):
        """An overall phase offset between physical TOAs and the TZR TOA."""

        return (
            (np.zeros(len(toas)) * self.PHOFF.quantity).to(u.dimensionless_unscaled)
            if toas.tzr
            else (-np.ones(len(toas)) * self.PHOFF.quantity).to(
                u.dimensionless_unscaled
            )
        )

    def d_offset_phase_d_PHOFF(self, toas, param, delay):
        """Derivative of the pulse phase w.r.t. PHOFF"""
        return (
            np.zeros(len(toas)) * u.Unit("")
            if toas.tzr
            else -np.ones(len(toas)) * u.Unit("")
        )
