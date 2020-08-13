from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np

from pint.models.parameter import MJDParameter, floatParameter, prefixParameter
from pint.models.timing_model import PhaseComponent, MissingParameter


class Wave(PhaseComponent):
    """This class provides harmonic signals.

    Historically, used for decomposition of timing noise into a series of
    sine/cosine components.

    For consistency with the implementation in tempo2, this signal is treated
    as a time series, but trivially converted into phase by multiplication by
    F0, which could makes changes to PEPOCH fragile if there is strong spin
    frequency evolution.
    """

    register = True
    category = "wave"

    def __init__(self):
        super(Wave, self).__init__()

        self.add_param(
            floatParameter(
                name="WAVE_OM",
                description="Base frequency of wave solution",
                units="1/d",
            )
        )
        self.add_param(
            prefixParameter(
                name="WAVE1",
                units="s",
                description="Wave components",
                type_match="pair",
                long_double=True,
                parameter_type="pair",
            )
        )
        self.add_param(
            MJDParameter(
                name="WAVEEPOCH",
                description="Reference epoch for wave solution",
                time_scale="tdb",
            )
        )
        self.phase_funcs_component += [self.wave_phase]

    def setup(self):
        super(Wave, self).setup()
        self.wave_terms = list(self.get_prefix_mapping_component("WAVE").keys())
        self.num_wave_terms = len(self.wave_terms)

    def validate(self):
        super(Wave, self).validate()
        self.setup()
        if self.WAVEEPOCH.quantity is None:
            if self.PEPOCH.quantity is None:
                raise MissingParameter(
                    "Wave",
                    "WAVEEPOCH",
                    "WAVEEPOCH or PEPOCH are required if " "WAVE_OM is set.",
                )
            else:
                self.WAVEEPOCH = self.PEPOCH

        if (not hasattr(self, "F0")) or (self.F0.quantity is None):
            raise MissingParameter(
                "Wave", "F0", "F0 is required if WAVE entries are present."
            )
        self.wave_terms.sort()
        wave_in_order = list(range(1, max(self.wave_terms) + 1))
        if not self.wave_terms == wave_in_order:
            diff = list(set(wave_in_order) - set(self.wave_terms))
            raise MissingParameter("Wave", "WAVE%d" % diff[0])

    def print_par(self,):
        result = ""
        wave_terms = ["WAVE%d" % ii for ii in range(1, self.num_wave_terms + 1)]

        result += self.WAVEEPOCH.as_parfile_line()
        result += self.WAVE_OM.as_parfile_line()
        for ft in wave_terms:
            par = getattr(self, ft)
            result += par.as_parfile_line()

        return result

    def wave_phase(self, toas, delays):
        times = 0
        wave_names = ["WAVE%d" % ii for ii in range(1, self.num_wave_terms + 1)]
        wave_terms = [getattr(self, name) for name in wave_names]
        wave_om = self.WAVE_OM.quantity
        base_phase = (
            wave_om
            * (
                toas.table["tdbld"] * u.day
                - self.WAVEEPOCH.value * u.day
                - delays.to(u.day)
            )
        ).value

        for k, wave_term in enumerate(wave_terms):
            wave_a, wave_b = wave_term.quantity
            wave_phase = (k + 1) * base_phase
            times += wave_a * np.sin(wave_phase)
            times += wave_b * np.cos(wave_phase)

        phase = ((times) * self.F0.quantity).to(u.dimensionless_unscaled)
        return phase
