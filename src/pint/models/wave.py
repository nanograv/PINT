"""Delays expressed as a sum of sinusoids."""
import astropy.units as u
import numpy as np

from pint.models.parameter import MJDParameter, floatParameter, prefixParameter
from pint.models.timing_model import PhaseComponent, MissingParameter


class Wave(PhaseComponent):
    """Delays expressed as a sum of sinusoids.

    Historically, used for decomposition of timing noise into a series of
    sine/cosine components.

    For consistency with the implementation in tempo2, this signal is treated
    as a time series, but trivially converted into phase by multiplication by
    F0, which could makes changes to PEPOCH fragile if there is strong spin
    frequency evolution.

    Parameters supported:

    .. paramtable::
        :class: pint.models.wave.Wave
    """

    register = True
    category = "wave"

    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="WAVEEPOCH",
                description="Reference epoch for wave solution",
                time_scale="tdb",
            )
        )
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
        self.phase_funcs_component += [self.wave_phase]

    def setup(self):
        super().setup()
        self.wave_terms = list(self.get_prefix_mapping_component("WAVE").keys())
        self.num_wave_terms = len(self.wave_terms)

    def validate(self):
        super().validate()
        self.setup()
        if self.WAVEEPOCH.quantity is None:
            if self._parent.PEPOCH.quantity is None:
                raise MissingParameter(
                    "Wave",
                    "WAVEEPOCH",
                    "WAVEEPOCH or PEPOCH are required if " "WAVE_OM is set.",
                )
            else:
                self.WAVEEPOCH.quantity = self._parent.PEPOCH.quantity

        if (not hasattr(self._parent, "F0")) or (self._parent.F0.quantity is None):
            raise MissingParameter(
                "Wave", "F0", "F0 is required if WAVE entries are present."
            )
        self.wave_terms.sort()
        wave_in_order = list(range(1, max(self.wave_terms) + 1))
        if self.wave_terms != wave_in_order:
            diff = list(set(wave_in_order) - set(self.wave_terms))
            raise MissingParameter("Wave", "WAVE%d" % diff[0])

    def print_par(self, format="pint"):
        result = ""
        wave_terms = ["WAVE%d" % ii for ii in range(1, self.num_wave_terms + 1)]

        result += self.WAVEEPOCH.as_parfile_line(format=format)
        result += self.WAVE_OM.as_parfile_line(format=format)
        for ft in wave_terms:
            par = getattr(self, ft)
            result += par.as_parfile_line(format=format)

        return result

    def add_wave_component(self, amps, index=None):
        """Add Wave Component

        Parameters
        ----------

        index : int
            Interger label for Wave components.
        amps : tuple of float or astropy.quantity.Quantity
            Sine and cosine amplitudes

        Returns
        -------

        index :
            Index that has been assigned to new Wave component
        """
        #### If index is None, increment the current max Wave index by 1. Increment using WAVE
        if index is None:
            dct = self.get_prefix_mapping_component("WAVE")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if int(index) in self.get_prefix_mapping_component("WAVE"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another"
            )

        for amp in amps:
            if isinstance(amp, u.quantity.Quantity):
                amp = amp.to_value(u.s)
        self.add_param(
            prefixParameter(
                name=f"WAVE{index}",
                value=amps,
                units="s",
                description="Wave components",
                type_match="pair",
                long_double=True,
                parameter_type="pair",
            )
        )
        self.setup()
        self.validate()
        return f"{index}"

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

        return ((times) * self._parent.F0.quantity).to(u.dimensionless_unscaled)
