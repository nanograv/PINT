"""This module implements a simple model of a constant dispersion measure."""
# dispersion.py
# Simple (constant) ISM dispersion measure
from warnings import warn
from .parameter import Parameter
from .timing_model import TimingModel

class Dispersion(TimingModel):
    """This class provides a timing model for a simple constant
    dispersion measure.
    """

    def __init__(self):
        super(Dispersion, self).__init__()

        self.add_param(Parameter(name="DM",
            units="pc cm^-3", value=0.0,
            description="Dispersion measure"))

        self.delay_funcs += [self.dispersion_delay,]
        self.delay_funcs_ld += [self.dispersion_delay_ld,]
        self.delay_funcs_table += [self.dispersion_delay_table,]
    def setup(self):
        super(Dispersion, self).setup()

    def dispersion_delay(self, toa):
        try:
            bfreq = self.barycentric_radio_freq(toa)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toa.freq

        # TODO: name DM constant
        return self.DM.value/2.41e-4/bfreq/bfreq

    def dispersion_delay_ld(self,TOAs):
        try:
            bfreq = self.barycentric_radio_freq_array(TOAs)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = TOAs.freq

        return self.DM.value/2.41e-4/bfreq/bfreq



    def dispersion_delay_table(self,TOAs):
        try:
            bfreq = self.barycentric_radio_freq_table(TOAs)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = TOAs.dataTable['freq'] 
        return (self.DM.value/2.41e-4/bfreq/bfreq).data    
        
        
           
