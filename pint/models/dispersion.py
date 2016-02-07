"""This module implements a simple model of a constant dispersion measure.
   And DMX dispersion"""
# dispersion.py
# Simple (constant) ISM dispersion measure
from warnings import warn
from .parameter import Parameter, prefixParameter
from .timing_model import TimingModel, Cache
import astropy.units as u
import numpy as np
import pint.utils as ut
import astropy.time as time
# The units on this are not completely correct
# as we don't really use the "pc cm^3" units on DM.
# But the time and freq portions are correct
DMconst = 1.0/2.41e-4 * u.MHz * u.MHz * u.s
# TODO split simple dispersion and DMX dispearion?


class Dispersion(TimingModel):
    """This class provides a timing model for a simple constant
    dispersion measure.
    """

    def __init__(self):
        super(Dispersion, self).__init__()
        self.add_param(Parameter(name="DM",
                       units="pc cm^-3", value=0.0,
                       description="Dispersion measure"))

        # DMX is for info output right now
        self.add_param(Parameter(name="DMX",
                       units="pc cm^-3", value=0.0,
                       description="Dispersion measure"))
        self.add_param(prefixParameter(prefix='DMX_', indexformat='0000',
                       units="pc cm^-3", value=0.0,
                       unitTplt=lambda x: "pc cm^-3",
                       description='Dispersion measure variation',
                       descriptionTplt=lambda x: "Dispersion measure"))
        self.add_param(prefixParameter(prefix='DMXR1_', indexformat='0000',
                       units="MJD",
                       value=time.Time(0.0, scale='tdb', format='mjd'),
                       unitTplt=lambda x: "MJD",
                       description='Beginning of DMX interval',
                       descriptionTplt=lambda x: 'Beginning of DMX interval'))
        self.add_param(prefixParameter(prefix='DMXR2_', indexformat='0000',
                       units="MJD",
                       value=time.Time(0.0, scale='tdb', format='mjd'),
                       unitTplt=lambda x: "MJD",
                       description='End of DMX interval',
                       descriptionTplt=lambda x: 'End of DMX interval'))

        self.delay_funcs['L1'] += [self.dispersion_delay]

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

    def dispersion_delay(self, toas):
        """Return the dispersion delay at each toa."""
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']

        # Constant dm delay
        dmdelay = self.DM.value * DMconst / bfreq**2
        # Set toas to the right DMX peiod.
        if 'DMX_section' not in toas.keys():
            toas['DMX_section'] = np.zeros_like(toas['index'])
            epoch_ind = 1
            while epoch_ind in self.DMX_mapping:
                # Get the parameters
                r1 = getattr(self, self.DMXR1_mapping[epoch_ind]).value
                r2 = getattr(self, self.DMXR2_mapping[epoch_ind]).value
                msk = np.logical_and(toas['tdbld'] >= ut.time_to_longdouble(r1),
                                     toas['tdbld'] <= ut.time_to_longdouble(r2))
                toas['DMX_section'][msk] = epoch_ind
                epoch_ind = epoch_ind + 1

        # Get DMX delays
        DMX_group = toas.group_by('DMX_section')
        for ii, key in enumerate(DMX_group.groups.keys):
            keyval = key.as_void()[0]
            if keyval != 0:
                dmx = getattr(self, self.DMX_mapping[keyval]).value
                ind = DMX_group.groups[ii]['index']
                # Apply the DMX delays
                dmdelay[ind] += dmx * DMconst / bfreq[ind]**2

        return dmdelay

    @Cache.use_cache
    def d_delay_d_DM(self, toas):
        """Return the dispersion delay at each toa."""
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']
        return DMconst / bfreq**2
