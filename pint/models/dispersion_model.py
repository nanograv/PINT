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
DMconst = 1.0/2.41e-4 * u.MHz * u.MHz * u.s

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
        self.delay_funcs['L1'] += [self.dedispersion_delay,]

    def setup(self):
        super(Dispersion, self).setup()

    def constant_dm(self, toas):
        cdm = np.zeros(len(toas))
        cdm.fill(self.DM.num_value)
        return cdm * self.DM.num_unit

    def dispersion_time_delay(self, DM, freq):
        """Return the dispersion time delay for a set of frequency."""
        # dm delay
        dmdelay = DM * DMconst / freq**2.0
        return dmdelay

    def dedispersion_delay(self, toas):
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']

        dm = np.zeros(len(toas)) * self.DM.num_unit
        for dm_f in self.dm_value_funcs:
            dm += dm_f(toas)

        return self.dispersion_time_delay(dm, bfreq)


class Dispersion_DMX(Dispersion):
    """This class provides a DMX model based on the class of Dispersion.
    """
    def __init__(self):
        super(Dispersion_DMX, self).__init__()
        # DMX is for info output right now
        self.add_param(p.floatParameter(name="DMX",
                       units="pc cm^-3", value=0.0,
                       description="Dispersion measure"))
        self.add_param(p.prefixParameter(prefix='DMX_', indexformat='0000',
                       units="pc cm^-3", value=0.0,
                       unitTplt=lambda x: "pc cm^-3",
                       description='Dispersion measure variation',
                       descriptionTplt=lambda x: "Dispersion measure",
                       type_match='float'))
        self.add_param(p.prefixParameter(prefix='DMXR1_', indexformat='0000',
                       units="MJD",
                       value=time.Time(0.0, scale='utc', format='mjd'),
                       unitTplt=lambda x: "MJD",
                       description='Beginning of DMX interval',
                       descriptionTplt=lambda x: 'Beginning of DMX interval',
                       type_match='MJD', time_scale='utc'))
        self.add_param(p.prefixParameter(prefix='DMXR2_', indexformat='0000',
                       units="MJD",
                       value=time.Time(0.0, scale='utc', format='mjd'),
                       unitTplt=lambda x: "MJD",
                       description='End of DMX interval',
                       descriptionTplt=lambda x: 'End of DMX interval',
                       type_match='MJD', time_scale='utc'))
        self.dm_value_funcs += [self.dmx_dm,]

    def setup(self):
        super(Dispersion, self).setup()
        # Get DMX mapping.
        self.get_prefix_mapping('DMX_')
        self.get_prefix_mapping('DMXR1_')
        self.get_prefix_mapping('DMXR2_')
        if len(self.DMX_mapping) != len(self.DMXR1_mapping):
            errorMsg = 'Number of DMX_ parameters is not'
            errorMsg += 'equals to Number of DMXR1_ parameters. '
            errorMsg += 'Please check your prefixed parameters.'
            raise AttributeError(errorMsg)

        if len(self.DMX_mapping) != len(self.DMXR2_mapping):
            errorMsg = 'Number of DMX_ parameters is not'
            errorMsg += 'equals to Number of DMXR2_ parameters. '
            errorMsg += 'Please check your prefixed parameters.'
            raise AttributeError(errorMsg)

    def dmx_dm(self, toas):
        # Set toas to the right DMX peiod.
        if 'DMX_section' not in toas.keys():
            toas['DMX_section'] = np.zeros_like(toas['index'])
            epoch_ind = 1
            while epoch_ind in self.DMX_mapping:
                # Get the parameters
                r1 = getattr(self, self.DMXR1_mapping[epoch_ind]).value
                r2 = getattr(self, self.DMXR2_mapping[epoch_ind]).value
                msk = np.logical_and(toas['mjd'] >= r1, toas['mjd'] <= r2)
                toas['DMX_section'][msk] = epoch_ind
                epoch_ind = epoch_ind + 1

        # Get DMX delays
        dm = np.zeros(len(toas)) * self.DM.num_unit
        DMX_group = toas.group_by('DMX_section')
        for ii, key in enumerate(DMX_group.groups.keys):
            keyval = key.as_void()[0]
            if keyval != 0:
                dmx = getattr(self, self.DMX_mapping[keyval]).value
                ind = DMX_group.groups[ii]['index']
                dm[ind] = dmx
        return dm
