from __future__ import absolute_import, print_function, division
from warnings import warn
from . import parameter as p
from .timing_model import DelayComponent
from .dispersion_model import Dispersion, DMconst
import astropy.units as u
import astropy.constants as const
import numpy as np
import pint.utils as ut
import astropy.time as time
from ..toa_select import TOASelect
from ..utils import taylor_horner, split_prefixed_name

class SolarWindDispersion(Dispersion):
    register = True
    def __init__(self):
        super(SolarWindDispersion, self).__init__()
        self.add_param(p.floatParameter(name="NE_SW",
                       units="cm^-3", value=0.0, aliases=['NE1AU', 'SOLARN0'],
                       description="Solar Wind Parameter"))
        self.add_param(p.floatParameter(name="SWM",
                       value=0.0, units="",
                       description="Solar Wind Model"))
        self.category = 'solar_wind'
        self.delay_funcs_component += [self.solar_wind_delay,]
        self.set_special_params(['NE_SW', 'SWM'])

    def setup(self):
        super(SolarWindDispersion, self).setup()
        self.register_deriv_funcs(self.d_delay_d_ne_sw, 'NE_SW')
    
    def solar_wind_delay(self, toas, acc_delay=None):
        '''Return the solar wind dispersion delay for a set of frequencies
        Eventually different solar wind models will be supported
        '''
        from astropy import log
        log.warning('%s\t%s\t%s' % (type(self), self.NE_SW, self.DM))
        if self.SWM.value == 0:
            tbl = toas.table
            try:
                bfreq = self.barycentric_radio_freq(toas)
            except AttributeError:
                warn("Using topocentric frequency for dedispersion!")
                bfreq = tbl['freq']

            dm = np.zeros(len(tbl)) * self.DM.units
            for dm_f in self.dm_value_funcs:
                dm += dm_f(toas)
            rsa = tbl['obs_sun_pos'].quantity
            pos = self.ssb_to_psb_xyz_ICRS(epoch=tbl['tdbld'].astype(np.float64))
            r = np.sqrt(np.sum(rsa*rsa, axis=1))
            cos_theta = np.sum(rsa*pos, axis=1) / r
            ret =  const.au**2.0 * np.arccos(cos_theta) * DMconst * self.NE_SW.quantity / \
                (r * np.sqrt(1.0 - cos_theta**2.0) * bfreq**2.0)
            ret[bfreq < 1.0 * u.MHz] = 0.0
            return ret
        else:
            #TODO Introduce the You et.al. (2007) Solar Wind Model for SWM=1
            raise NotImplementedError('Solar Dispersion Delay not implemented for SWM %d' % self.SWM.value)
    
    def d_delay_d_ne_sw(self, toas, param_name, acc_delay=None):
        if self.SWM.value == 0:
            tbl = toas.table
            try:
                bfreq = self.barycentric_radio_freq(toas)
            except AttributeError:
                warn('Using topocentric frequency for solar wind dedispersion!')
                bfreq = tbl['freq']

            rsa = tbl['obs_sun_pos'].quantity
            pos = self.ssb_to_psb_xyz_ICRS(epoch=tbl['tdbld'].astype(np.float64))
            r = np.sqrt(np.sum(rsa*rsa, axis=1))
            cos_theta = np.sum(rsa*pos, axis=1) / r
            
            #ret = AUdist**2.0 / const.c * np.arccos(cos_theta) * DMconst / \
            ret = AUdist**2.0 * np.arccos(cos_theta) * DMconst / \
                (r * np.sqrt(1 - cos_theta**2.0) * bfreq**2.0)
            ret[bfreq < 1.0 * u.MHz] = 0.0
            return ret
        else:
            raise NotImplementedError('Solar Dispersion Delay Derivative not implemented for SWM %d' % self.SWM.value)
    
    def print_par(self,):
        result  = ''
        result += getattr(self, 'NE_SW').as_parfile_line()
        result += getattr(self, 'SWM').as_parfile_line()
        return result
