"""Delays expressed as a sum of sinusoids."""
import astropy.units as u
import numpy as np

from pint.models.parameter import MJDParameter, floatParameter, prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter


class WaveX(DelayComponent):
    """Implementation of the wave model as a delay correction

    Delays are expressed as a sum of sinusoids.

    Used for decomposition of timing noise into a series of sine/cosine components with the amplitudes as fitted parameters.


    Parameters supported:

    .. paramtable::
        :class: pint.models.wave.Wave

    This is an extension of the L13 method described in Lentati et al., 2013
    """

    register = True

    #  category = "wave"
    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="WXEPOCH",
                description="Reference epoch for wave delay solution",
                time_scale="tdb",
            )
        )
        self.set_special_params(["WXFREQ_0001", "WXSIN_0001", "WXCOS_0001"])
        self.delay_funcs_component += [self.wavex_delay]

    def add_wavex_component(self, wxfreq=1, index=None, frozens=True):
        """Add WaveX component

        Parameters
        ----------

        wxfreq : float or astropy.quantity.Quantity
            Base frequency for WaveX component
        index : int, None
            Interger label for WaveX component. If None, will increment largest used index by 1.
        frozens : iterable of bool or bool
            Indicates whether wavex will be fit

        Returns
        -------

        index : int
            Index that has been assigned to new WaveX component
        """

        #### If index is None, increment the current max WaveX index by 1. Increment using WXFREQ
        if index is None:
            dct = self.get_prefix_mapping_component("WXFREQ_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if int(index) in self.get_prefix_mapping_component("WXFREQ_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another"
            )

        if isinstance(wxfreq, u.quantity.Quantity):
            wxfreq = wxfreq.to_value(u.d**-1)
        self.add_param(
            prefixParameter(
                name="WXFREQ_{i}",
                description="Base frequency of wave delay solution",
                units="1/d",
                value=wxfreq,
                frozen=frozen,
            )
        )
        self.add_param(
            prefixParameter(
                name="WXSIN_{i}",
                description="Sine amplitudes for wave delay function",
                units="s",
            )
        )
        self.add_param(
            prefixParameter(
                name="WXCOS_{i}",
                description="Cosine amplitudes for wave delay function",
                units="s",
            )
        )
        self.setup()
        self.validate()
        return index

    # Initialize setup
    def setup(self):
        super().setup()

    #     self.wave_freqs = list(self.get_prefix_mapping_component("WXFREQ_").keys())
    #     self.num_wave_freqs = len(self.wave_freqs)

    # Placeholder for validation tests
    # def validate(self)

    def wavex_delay(self, toas, delays):
        total_delay = 0
        # wave_freq_params = self.get_prefix_mapping_component("WXFREQ_")
        wave_freqs = self.get_prefix_mapping_component("WXFREQ_").values()
        wave_sins = self.get_prefix_mapping_component("WXSIN_").values()
        wave_cos = self.get_prefix_mapping_component("WXSIN_").values()
        base_phase = (
            toas.table["tbdld"] * u.day - self.WXEPOCH.value * u.day - delays.to(u.day)
        ).value
        for f, freq in enumerate(wave_freqs):
            arg = 2.0 * np.pi * freq.value * base_phase
            total_delay += wave_sins[f] * np.sin(arg) + wave_cos[f] * np.cos(arg)

    # Placeholder for calculations of derivatives
    # def d_wavex_delay
