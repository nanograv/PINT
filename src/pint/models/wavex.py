"""Delays expressed as a sum of sinusoids."""
import astropy.units as u
import numpy as np

from pint.models.parameter import MJDParameter, floatParameter, prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter

class WaveX(DelayComponent):
# These lines are in wave.py and not sure what they are meant to do
#   register = True
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
        self.add_param(
            prefixParameter(
            name="WXFREQ_",
            description="Base frequency of wave delay solution",
            units="1/d",
            )
        )
        self.add_param(
            prefixParameter(
            name="WXSIN_",
            description="Sine amplitudes for wave delay function",
            units="s",
            )
        )
        self.add_param(
            prefixParameter(
            name="WXCOS_",
            description="Cosine amplitudes for wave delay function",
            units="s",
            )
        )
        #self.delay_funcs_component += [self.wavex_delay]

    #Initialize setup 
    # def setup(self):
    #     super().setup()
    #     self.wave_freqs = list(self.get_prefix_mapping_component("WXFREQ_").keys())
    #     self.num_wave_freqs = len(self.wave_freqs)
    
    #Placeholder for validation tests
    # def validate(self)

    def wavex_delay(self,toas,delays):
        total_delay = 0
       # wave_freq_params = self.get_prefix_mapping_component("WXFREQ_")
        wave_freqs = self.get_prefix_mapping_component("WXFREQ_").values()
        wave_sins = self.get_prefix_mapping_component("WXSIN_").values()
        wave_cos = self.get_prefix_mapping_component("WXSIN_").values()
        base_phase = (
            toas.table["tbdld"] * u.day
            - self.WXEPOCH.value * u.day
            - delays.to(u.day)
        ).value
        for f,freq in enumerate(wave_freqs):
            arg = 2. * np.pi * freq.value * base_phase
            total_delay += (wave_sins[f] * np.sin(arg) + wave_cos[f] * np.cos(arg))

    #Placeholder for calculations of derivatives
    # def d_wavex_delay




