"""This module implements a simple model of a base dispersion delay.
   And DMX dispersion"""
from __future__ import absolute_import, print_function, division
from warnings import warn
from . import parameter as p
from .timing_model import DelayComponent
import astropy.units as u
import astropy.constants as const
import numpy as np
import pint.utils as ut
import astropy.time as time
from ..toa_select import TOASelect
from ..utils import taylor_horner, split_prefixed_name

# The units on this are not completely correct
# as we don't really use the "pc cm^3" units on DM.
# But the time and freq portions are correct
# This value is cited from Duncan Lorimer, Michael Kramer, Handbook of Pulsar
# Astronomy, Second edition, Page 86, Note 1
DMconst = 1.0/2.41e-4 * u.MHz * u.MHz * u.s * u.cm**3 / u.pc


class Dispersion(DelayComponent):
    """This class provides a base dispersion timing model.
    """
    def __init__(self):
        super(Dispersion, self).__init__()
        self.add_param(p.boolParameter(name='DMDATA', value=0,
                                       description="Flag for using the DM data"
                                       " from the Wideband TOAs."))
        self.dm_value_funcs = []
        self.dm_value_derivs = {}

    def dispersion_time_delay(self, DM, freq):
        """Return the dispersion time delay for a set of frequency.
        This equation if cited from Duncan Lorimer, Michael Kramer,
        Handbook of Pulsar Astronomy, Second edition, Page 86, Equation [4.7]
        Here we assume the reference frequency is at infinity and the EM wave
        frequency is much larger than plasma frequency.
        """
        # dm delay
        dmdelay = DM * DMconst / freq**2.0
        return dmdelay

    def dispersion_type_delay(self, toas):
        tbl = toas.table
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = tbl['freq']

        dm = np.zeros(len(tbl)) * self.DM.units
        for dm_f in self.dm_value_funcs:
            dm += dm_f(toas)
        return self.dispersion_time_delay(dm, bfreq)

    def d_dm_d_param(self, toas, param_name, acc_delay=None):
        par = getattr(self, param_name)
        final_unit = self.DM.units / par.units
        if param_name in self.dm_value_derivs.keys():
            result = self.dm_value_derivs[param_name](toas, param_name,
                                                      acc_delay=None)
            return result.to(final_unit)
        else:
            return np.zeros(toas.ntoas) * final_unit


    def dm_designmatrix(self, toas, acc_delay=None, scale_by_F0=True, \
                        incfrozen=False, incoffset=True):
        """Get the designmatrix for DM values. The API has kept the same with
           designmatrix.
        """
        params = ['Offset',] if incoffset else []
        params += [par for par in self._parent.params if incfrozen or
                   not getattr(self, par).frozen]
        units = []
        nparams = len(params)
        M = np.zeros((toas.ntoas, nparams))
        for ii, param in enumerate(params):
            if param == 'Offset':
                M[:,ii] = 0.0
                units.append(self.DM.units / u.day)
            else:
                der = self.d_dm_d_param(toas, param, acc_delay=None)
                M[:,ii] = der
                units.append(der.unit)
        return M, params, units


class DispersionDM(Dispersion):
    """
    This is the DM dispersion model. This model uses Taylor expansion to model
    the DM variation over time. It is also can be used for the constant DM model.
    """
    register = True
    def __init__(self):
        super(DispersionDM, self).__init__()
        self.add_param(p.floatParameter(name="DM", units="pc cm^-3", value=0.0,
                       description="Dispersion measure", long_double=True))
        self.add_param(p.prefixParameter(name="DM1", value=0.0, units='pc cm^-3/yr^1',
                       description="First order time derivative of the dispersion measure",
                       unit_template=self.DM_dervative_unit,
                       description_template=self.DM_dervative_description,
                       type_match='float', long_double=True))
        self.add_param(p.MJDParameter(name="DMEPOCH",
                       description="Epoch of DM measurement"))

        self.dm_value_funcs += [self.base_dm,]
        self.category = 'dispersion_constant'
        self.delay_funcs_component += [self.constant_dispersion_delay,]

    def setup(self):
        super(Dispersion, self).setup()
        # If DM1 is set, we need DMEPOCH
        if self.DM1.value != 0.0:
            if self.DMEPOCH.value is None:
                raise MissingParameter("Dispersion", "DMEPOCH",
                        "DMEPOCH is required if DM1 or higher are set")
        base_dms = list(self.get_prefix_mapping_component('DM').values())
        base_dms += ['DM',]

        for dm_name in base_dms:
            self.register_deriv_funcs(self.d_delay_d_DMs, dm_name)
            self.dm_value_derivs.update({dm_name: self.d_dm_d_DMs})
    def DM_dervative_unit(self, n):
        return "pc cm^-3/yr^%d" % n if n else "pc cm^-3"

    def DM_dervative_description(self, n):
        return "%d'th time derivative of the dispersion measure" % n

    def get_DM_terms(self):
        """Return a list of the DM term values in the model: [DM, DM1, ..., DMn]
        """
        prefix_dm = list(self.get_prefix_mapping_component('DM').values())
        dm_terms = [self.DM.quantity,]
        dm_terms += [getattr(self, x).quantity for x in prefix_dm]
        return dm_terms

    def base_dm(self, toas):
        tbl = toas.table
        dm = np.zeros(len(tbl))
        dm_terms = self.get_DM_terms()
        if self.DMEPOCH.value is None:
            DMEPOCH = tbl['tdbld'][0]
        else:
            DMEPOCH = self.DMEPOCH.value
        dt = (tbl['tdbld'] - DMEPOCH) * u.day
        dt_value = (dt.to(u.yr)).value
        dm_terms_value = [d.value for d in dm_terms]
        dm = taylor_horner(dt_value, dm_terms_value)
        return dm * self.DM.units

    def constant_dispersion_delay(self, toas, acc_delay=None):
        """ This is a wrapper function for interacting with the TimingModel class
        """
        return self.dispersion_type_delay(toas)

    def print_par(self,):
        # TODO we need to have a better design for print out the parameters in
        # an inhertance class.
        result  = ''
        prefix_dm = list(self.get_prefix_mapping_component('DM').values())
        dms = ['DM'] + prefix_dm
        for dm in dms:
            result += getattr(self, dm).as_parfile_line()
        if hasattr(self, 'components'):
            all_params = self.components['DispersionDM'].params
        else:
            all_params = self.params
        for pm in all_params:
            if pm not in dms:
                result += getattr(self, pm).as_parfile_line()
        return result

    def d_dm_d_DMs(self, toas, param_name, acc_delay=None):
        """Derivatives : d_dm_d_dm_params
        """
        par = getattr(self, param_name)
        unit = par.units
        if param_name == 'DM':
            order = 0
        else:
            pn, idxf, order = split_prefixed_name(param_name)
        dms = self.get_DM_terms()
        dm_terms = np.longdouble(np.zeros(len(dms)))
        dm_terms[order] = np.longdouble(1.0)
        if self.DMEPOCH.value is None:
            DMEPOCH = tbl['tdbld'][0]
        else:
            DMEPOCH = self.DMEPOCH.value
        dt = (tbl['tdbld'] - DMEPOCH) * u.day
        dt_value = (dt.to(u.yr)).value
        d_dm_d_DM = taylor_horner(dt_value,
                                  dm_terms) * (self.DM.units / par.units)
        return d_dm_d_DM

    def d_delay_d_DMs(self, toas, param_name, acc_delay=None): # NOTE we should have a better name for this.
        """Derivatives: d_delay_d_DMs
        """
        tbl = toas.table
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = tbl['freq']
        d_dm_d_DMs = self.d_dm_d_DMs(toas, param_name, acc_delay=None)
        return DMconst * d_dm_d_DMs/ bfreq**2.0


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
                       unit_template=lambda x: "pc cm^-3",
                       description='Dispersion measure variation',
                       description_template=lambda x: "Dispersion measure",
                       paramter_type='float'))
        self.add_param(p.prefixParameter(name='DMXR1_0001',
                       units="MJD",
                       unit_template=lambda x: "MJD",
                       description='Beginning of DMX interval',
                       description_template=lambda x: 'Beginning of DMX interval',
                       parameter_type='MJD', time_scale='utc'))
        self.add_param(p.prefixParameter(name='DMXR2_0001', units="MJD",
                       unit_template=lambda x: "MJD",
                       description='End of DMX interval',
                       description_template=lambda x: 'End of DMX interval',
                       parameter_type='MJD', time_scale='utc'))
        self.dm_value_funcs += [self.dmx_dm,]
        self.set_special_params(['DMX_0001', 'DMXR1_0001','DMXR2_0001'])
        self.delay_funcs_component += [self.DMX_dispersion_delay,]
        self.category = "dispersion_dmx"

    def setup(self):
        super(DispersionDMX, self).setup()
        # Get DMX mapping.
        DMX_mapping = self.get_prefix_mapping_component('DMX_')
        DMXR1_mapping = self.get_prefix_mapping_component('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping_component('DMXR2_')
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
                self.register_deriv_funcs(self.d_delay_d_DMX, prefix_par)
                self.dm_value_derivs.update({prefix_par: self.d_dm_d_DMX})

    def dmx_dm(self, toas):
        condition = {}
        tbl = toas.table
        if not hasattr(self, 'dmx_toas_selector'):
            self.dmx_toas_selector = TOASelect(is_range=True)
        DMX_mapping = self.get_prefix_mapping_component('DMX_')
        DMXR1_mapping = self.get_prefix_mapping_component('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping_component('DMXR2_')
        for epoch_ind in DMX_mapping.keys():
            r1 = getattr(self, DMXR1_mapping[epoch_ind]).quantity
            r2 = getattr(self, DMXR2_mapping[epoch_ind]).quantity
            condition[DMX_mapping[epoch_ind]] = (r1.mjd, r2.mjd)
        select_idx = self.dmx_toas_selector.get_select_index(condition, tbl['mjd_float'])
        #Get DMX delays
        dm = np.zeros(len(tbl)) * self.DM.units
        for k, v in select_idx.items():
           dm[v] = getattr(self, k).quantity
        return dm

    def DMX_dispersion_delay(self, toas, acc_delay=None):
        """ This is a wrapper function for interacting with the TimingModel class
        """
        return self.dispersion_type_delay(toas)

    def d_dm_d_DMX(self, toas, param_name, acc_delay=None):
        """Derivatives : d_dm_d_dmx
        """
        condition = {}
        tbl = toas.table
        if not hasattr(self, 'dmx_toas_selector'):
            self.dmx_toas_selector = TOASelect(is_range=True)
        param = getattr(self, param_name)
        dmx_index = param.index
        DMXR1_mapping = self.get_prefix_mapping_component('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping_component('DMXR2_')
        r1 = getattr(self, DMXR1_mapping[dmx_index]).quantity
        r2 = getattr(self, DMXR2_mapping[dmx_index]).quantity
        condition = {param_name:(r1.mjd, r2.mjd)}
        select_idx = self.dmx_toas_selector.get_select_index(condition, tbl['mjd_float'])
        d_dm_d_dmx = np.zeros(len(tbl))
        for k, v in select_idx.items():
           d_dm_d_dmx[v] = 1.0
        return d_dm_d_dmx

    def d_delay_d_DMX(self, toas, param_name, acc_delay=None):
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = tbl['freq']
        d_dm_d_dmx = self.d_dm_d_DMX(toas, param_name, acc_delay=None)
        return DMconst * d_dm_d_dmx / bfreq**2.0

    def print_par(self,):
        result = ''
        DMX_mapping = self.get_prefix_mapping_component('DMX_')
        DMXR1_mapping = self.get_prefix_mapping_component('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping_component('DMXR2_')
        result += getattr(self, 'DMX').as_parfile_line()
        sorted_list = sorted(DMX_mapping.keys())
        for ii in sorted_list:
            result += getattr(self, DMX_mapping[ii]).as_parfile_line()
            result += getattr(self, DMXR1_mapping[ii]).as_parfile_line()
            result += getattr(self, DMXR2_mapping[ii]).as_parfile_line()
        return result
