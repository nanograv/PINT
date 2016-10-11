"""This module implements a simple model of a constant dispersion measure.
   And DMX dispersion"""
# dispersion.py
# Simple (constant) ISM dispersion measure
from warnings import warn
import parameter as p
from .timing_model import TimingModel, Cache
import astropy.units as u
import numpy as np
import pint.utils as ut
import astropy.time as time
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
    def __init__(self):
        super(Dispersion, self).__init__()
        self.add_param(p.floatParameter(name="DM",
                       units="pc cm^-3", value=0.0,
                       description="Dispersion measure"))
        self.dm_value_funcs = [self.constant_dm,]
        self.delay_funcs['L1'] += [self.dispersion_delay,]
        self.delay_derivs += [self.d_delay_dispersion_d_DM,]

    def setup(self):
        super(Dispersion, self).setup()

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

    def d_delay_dispersion_d_DM(self, toas):
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
                       value=time.Time(0.0, scale='utc', format='mjd'),
                       unitTplt=lambda x: "MJD",
                       description='Beginning of DMX interval',
                       descriptionTplt=lambda x: 'Beginning of DMX interval',
                       parameter_type='MJD', time_scale='utc'))
        self.add_param(p.prefixParameter(name='DMXR2_0001', units="MJD",
                       value=time.Time(0.0, scale='utc', format='mjd'),
                       unitTplt=lambda x: "MJD",
                       description='End of DMX interval',
                       descriptionTplt=lambda x: 'End of DMX interval',
                       parameter_type='MJD', time_scale='utc'))
        self.dm_value_funcs += [self.dmx_dm,]
        self.set_special_params(['DMX_0001', 'DMXR1_0001','DMXR2_0001'])

    def setup(self):
        super(Dispersion, self).setup()
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

    def dmx_dm(self, toas):
        # Set toas to the right DMX peiod.
        DMX_mapping = self.get_prefix_mapping('DMX_')
        DMXR1_mapping = self.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.get_prefix_mapping('DMXR2_')
        if 'DMX_section' not in toas.keys():
            toas['DMX_section'] = np.zeros_like(toas['index'])
            epoch_ind = 1
            while epoch_ind in DMX_mapping:
                # Get the parameters
                r1 = getattr(self, DMXR1_mapping[epoch_ind]).quantity
                r2 = getattr(self, DMXR2_mapping[epoch_ind]).quantity
                msk = np.logical_and(toas['mjd_float'] >= r1.mjd, toas['mjd_float'] <= r2.mjd)
                toas['DMX_section'][msk] = epoch_ind
                epoch_ind = epoch_ind + 1

        # Get DMX delays
        dm = np.zeros(len(toas)) * self.DM.units
        DMX_group = toas.group_by('DMX_section')
        for ii, key in enumerate(DMX_group.groups.keys):
            keyval = key.as_void()[0]
            if keyval != 0:
                dmx = getattr(self, DMX_mapping[keyval]).quantity
                ind = DMX_group.groups[ii]['index']
                dm[ind] = dmx
        return dm

    def d_delay_dmx_d_DMX(self, toas):
        pass
