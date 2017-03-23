"""This module implements a simple model of a constant dispersion measure.
   And DMX dispersion"""
# dispersion.py
# Simple (constant) ISM dispersion measure
from warnings import warn
from . import parameter as p
from .timing_model import TimingModel, Cache
import astropy.units as u
import numpy as np
import pint.utils as ut
import astropy.time as time
from ..toa_select import TOASelect

# The units on this are not completely correct
# as we don't really use the "pc cm^3" units on DM.
# But the time and freq portions are correct
# This value is cited from Duncan Lorimer, Michael Kramer, Handbook of Pulsar
# Astronomy, Second edition, Page 86, Note 1
DMconst = 1.0/2.41e-4 * u.MHz * u.MHz * u.s * u.cm**3 / u.pc

class Dispersion(TimingModel):
    """This class provides a base dispersion timing model. The dm varience will
    be treated linearly.
    """
    register = True
    def __init__(self):
        super(Dispersion, self).__init__()
        self.add_param(p.floatParameter(name="DM",
                       units="pc cm^-3", value=0.0,
                       description="Dispersion measure"))
        self.dm_value_funcs = [self.constant_dm,]
        self.delay_funcs['L1'] += [self.dispersion_delay,]
        self.order_number = 2

    def setup(self):
        super(Dispersion, self).setup()
        self.register_deriv_funcs(self.d_delay_d_DM, 'delay', 'DM')

    def constant_dm(self, toas):
        cdm = np.zeros(len(toas))
        cdm.fill(self.DM.quantity)
        return cdm * self.DM.units

    def dispersion_time_delay(self, DM, freq):
        """Return the dispersion time delay for a set of frequency.
        This equation if cited from Duncan Lorimer, Michael Kramer, Handbook of Pulsar
        Astronomy, Second edition, Page 86, Equation [4.7]
        Here we assume the reference frequency is at infinity and the EM wave
        frequency is much larger than plasma frequency.
        """
        # dm delay
        dmdelay = DM * DMconst / freq**2.0
        return dmdelay

    def dispersion_delay(self, toas):
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']

        dm = np.zeros(len(toas)) * self.DM.units
        for dm_f in self.dm_value_funcs:
            dm += dm_f(toas)

        return self.dispersion_time_delay(dm, bfreq)

    def d_delay_d_DM(self, toas):
        """Derivatives for constant DM
        """
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']

        return DMconst / bfreq**2.0

class DispersionDMX(Dispersion):
    """This class provides a DMX model based on the class of Dispersion.
    """
    register = True
    def __init__(self):
        super(DispersionDMX, self).__init__()
        # DMX is for info output right now
        self.add_param(p.floatParameter(name="DMX",
                       units="pc cm^-3", value=0.0,
                       description="Dispersion measure"))
        self.add_param(p.prefixParameter(name='DMX_0001',
                       units="pc cm^-3", value=0.0,
                       unitTplt=lambda x: "pc cm^-3",
                       description='Dispersion measure variation',
                       descriptionTplt=lambda x: "Dispersion measure",
                       paramter_type='float'))
        self.add_param(p.prefixParameter(name='DMXR1_0001',
                       units="MJD",
                       unitTplt=lambda x: "MJD",
                       description='Beginning of DMX interval',
                       descriptionTplt=lambda x: 'Beginning of DMX interval',
                       parameter_type='MJD', time_scale='utc'))
        self.add_param(p.prefixParameter(name='DMXR2_0001', units="MJD",
                       unitTplt=lambda x: "MJD",
                       description='End of DMX interval',
                       descriptionTplt=lambda x: 'End of DMX interval',
                       parameter_type='MJD', time_scale='utc'))
        self.dm_value_funcs += [self.dmx_dm,]
        self.set_special_params(['DMX_0001', 'DMXR1_0001','DMXR2_0001'])
        self.print_par_func = 'print_par_DMX'

    def setup(self):
        super(DispersionDMX, self).setup()
        # Get DMX mapping.
        DMX_mapping = self.get_prefix_mapping('DMX_')
        DMXR1_mapping = self.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping('DMXR2_')
        if len(DMX_mapping) != len(DMXR1_mapping):
            errorMsg = 'Number of DMX_ parameters is not'
            errorMsg += 'equals to Number of DMXR1_ parameters. '
            errorMsg += 'Please check your prefixed parameters.'
            raise AttributeError(errorMsg)

        if len(DMX_mapping) != len(DMXR2_mapping):
            errorMsg = 'Number of DMX_ parameters is not'
            errorMsg += 'equals to Number of DMXR2_ parameters. '
            errorMsg += 'Please check your prefixed parameters.'
            raise AttributeError(errorMsg)
        # create d_delay_d_dmx functions
        for prefix_par in self.get_params_of_type('prefixParameter'):
            if prefix_par.startswith('DMX_'):
                self.register_deriv_funcs(self.d_delay_d_DMX, 'delay', prefix_par)

    def dmx_dm(self, toas):
        condition = {}
        if not hasattr(self, 'dmx_toas_selector'):
            self.dmx_toas_selector = TOASelect(is_range=True)
        DMX_mapping = self.get_prefix_mapping('DMX_')
        DMXR1_mapping = self.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping('DMXR2_')
        for epoch_ind in DMX_mapping.keys():
            r1 = getattr(self, DMXR1_mapping[epoch_ind]).quantity
            r2 = getattr(self, DMXR2_mapping[epoch_ind]).quantity
            condition[DMX_mapping[epoch_ind]] = (r1.mjd, r2.mjd)
        select_idx = self.dmx_toas_selector.get_select_index(condition, toas['mjd_float'])
        #Get DMX delays
        dm = np.zeros(len(toas)) * self.DM.units
        for k, v in select_idx.items():
           dm[v] = getattr(self, k).quantity
        return dm

    def d_delay_d_DMX(self, toas, param_name):
        condition = {}
        if not hasattr(self, 'dmx_toas_selector'):
            self.dmx_toas_selector = TOASelect(is_range=True)
        param = getattr(self, param_name)
        dmx_index = param.index
        DMXR1_mapping = self.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping('DMXR2_')
        r1 = getattr(self, DMXR1_mapping[dmx_index]).quantity
        r2 = getattr(self, DMXR2_mapping[dmx_index]).quantity
        condition = {param_name:(r1.mjd, r2.mjd)}
        select_idx = self.dmx_toas_selector.get_select_index(condition, toas['mjd_float'])

        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']
        dmx = np.zeros(len(toas))
        for k, v in select_idx.items():
           dmx[v] = 1.0
        return DMconst * dmx / bfreq**2.0

    def print_par_DMX(self,):
        result = ''
        DMX_mapping = self.get_prefix_mapping('DMX_')
        DMXR1_mapping = self.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping('DMXR2_')
        result += getattr(self, 'DM').as_parfile_line()
        result += getattr(self, 'DMX').as_parfile_line()
        sorted_list = sorted(DMX_mapping.keys())
        for ii in sorted_list:
            result += getattr(self, DMX_mapping[ii]).as_parfile_line()
            result += getattr(self, DMXR1_mapping[ii]).as_parfile_line()
            result += getattr(self, DMXR2_mapping[ii]).as_parfile_line()
        return result
