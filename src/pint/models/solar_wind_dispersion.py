"""Dispersion due to the solar wind."""
from __future__ import absolute_import, division, print_function

from warnings import warn

import astropy.constants as const
import astropy.units as u
import numpy as np

import pint.utils as ut
from pint.models.dispersion_model import Dispersion, DMconst
from pint.models.parameter import floatParameter
from pint.toa_select import TOASelect


class SolarWindDispersion(Dispersion):
    """Dispersion due to the solar wind (basic model).

    The model is a simple spherically-symmetric model that varies
    only in its amplitude.

    References
    ----------
    Madison et al. 2019, ApJ, 872, 150; Section 3.1
    Edwards et al. 2006, MNRAS, 372, 1549; Setion 2.5.4

    """

    register = True
    category = "solar_wind"

    def __init__(self):
        super(SolarWindDispersion, self).__init__()
        self.add_param(
            floatParameter(
                name="NE_SW",
                units="cm^-3",
                value=0.0,
                aliases=["NE1AU", "SOLARN0"],
                description="Solar Wind Parameter",
            )
        )
        self.add_param(
            floatParameter(
                name="SWM", value=0.0, units="", description="Solar Wind Model"
            )
        )
        self.delay_funcs_component += [self.solar_wind_delay]
        self.set_special_params(["NE_SW", "SWM"])

    def setup(self):
        super(SolarWindDispersion, self).setup()
        self.register_deriv_funcs(self.d_delay_d_ne_sw, "NE_SW")

    def validate(self):
        super(SolarWindDispersion, self).validate()

    def solar_wind_delay(self, toas, acc_delay=None):
        """Return the solar wind dispersion delay for a set of frequencies
        Eventually different solar wind models will be supported

        Implements equations 29, 30 of Edwards et al. 2006,
        where their rho is given as theta here

        rvec: radial vector from observatory to the center of the Sun
        pos: pulsar position
        """
        if self.SWM.value == 0:
            tbl = toas.table
            try:
                bfreq = self.barycentric_radio_freq(toas)
            except AttributeError:
                warn("Using topocentric frequency for dedispersion!")
                bfreq = tbl["freq"]

            rvec = tbl["obs_sun_pos"].quantity
            pos = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"].astype(np.float64))
            r = np.sqrt(np.sum(rvec * rvec, axis=1))
            cos_theta = (np.sum(rvec * pos, axis=1) / r).to(u.Unit("")).value
            ret = (
                const.au ** 2.0
                * np.arccos(cos_theta)
                * DMconst
                * self.NE_SW.quantity
                / (r * np.sqrt(1.0 - cos_theta ** 2.0) * bfreq ** 2.0)
            )
            ret[bfreq < 1.0 * u.MHz] = 0.0
            return ret
        else:
            # TODO Introduce the You et.al. (2007) Solar Wind Model for SWM=1
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )

    def d_delay_d_ne_sw(self, toas, param_name, acc_delay=None):
        if self.SWM.value == 0:
            tbl = toas.table
            try:
                bfreq = self.barycentric_radio_freq(toas)
            except AttributeError:
                warn("Using topocentric frequency for solar wind dedispersion!")
                bfreq = tbl["freq"]

            rvec = tbl["obs_sun_pos"].quantity
            pos = self.ssb_to_psb_xyz_ICRS(epoch=tbl["tdbld"].astype(np.float64))
            r = np.sqrt(np.sum(rvec * rvec, axis=1))
            cos_theta = np.sum(rvec * pos, axis=1) / r

            # ret = AUdist**2.0 / const.c * np.arccos(cos_theta) * DMconst / \
            ret = (
                AUdist ** 2.0
                * np.arccos(cos_theta)
                * DMconst
                / (r * np.sqrt(1 - cos_theta ** 2.0) * bfreq ** 2.0)
            )
            ret[bfreq < 1.0 * u.MHz] = 0.0
            return ret
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay Derivative not implemented for SWM %d"
                % self.SWM.value
            )

    def print_par(self,):
        result = ""
        result += getattr(self, "NE_SW").as_parfile_line()
        result += getattr(self, "SWM").as_parfile_line()
        return result
