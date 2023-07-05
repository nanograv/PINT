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