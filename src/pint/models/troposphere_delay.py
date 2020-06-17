"""Delay due to Earth's troposphere"""
from __future__ import absolute_import, division, print_function

from warnings import warn

import astropy.constants as const
import astropy.units as u
import numpy as np
import pint.utils as ut
from astropy import log
from astropy.coordinates import AltAz, SkyCoord
from pint.models.timing_model import DelayComponent
from pint.observatory import get_observatory
from pint.toa_select import TOASelect


class TroposphereDelay(DelayComponent):
    """Dispersion due to the solar wind (basic model).

    The model is a simple spherically-symmetric model that varies
    only in its amplitude.

    References
    ----------
    Madison et al. 2019, ApJ, 872, 150; Section 3.1
    Edwards et al. 2006, MNRAS, 372, 1549; Setion 2.5.4

    """

    register = True
    category = "troposphere"

    # zero padding will provide constant within 15degrees of the poles or equator
    A_AVG = (
        np.array([0.0, 1.2769934, 1.2683230, 1.2465397, 1.2196049, 1.2045996, 0.0])
        * 1e-3
    )
    B_AVG = (
        np.array([0.0, 2.9153695, 2.9152299, 2.9288445, 2.9022565, 2.9024912, 0.0])
        * 1e-3
    )
    C_AVG = (
        np.array([0.0, 62.610505, 62.837393, 63.721774, 63.824265, 64.258455, 0.0])
        * 1e-3
    )

    A_AMP = np.array([0.0, 0.0, 1.2709626, 2.6523662, 3.4000452, 4.1202191, 0.0]) * 1e-5

    B_AMP = np.array([0.0, 0.0, 2.1414979, 3.0160779, 7.2562722, 11.723375, 0.0]) * 1e-5

    C_AMP = np.array([0.0, 0.0, 9.0128400, 4.3497037, 84.795348, 170.37206, 0.0]) * 1e-5

    A_HT = 2.53e-5
    B_HT = 5.49e-3
    C_HT = 1.14e-3

    LAT = np.array([0, 15, 30, 45, 60, 75, 90]) * u.deg  # in degrees

    DOY_OFFSET = -28  # add this into the MJD value to get the right phase

    def __init__(self):
        super(TroposphereDelay, self).__init__()
        # copy over the
        for array in [
            self.A_AVG,
            self.B_AVG,
            self.C_AVG,
            self.A_AMP,
            self.B_AMP,
            self.C_AMP,
        ]:
            array[0] = array[1]
            array[-1] = array[-2]

    def setup(self):
        super(TroposphereDelay, self).setup()

    def validate(self):
        super(TroposphereDelay, self).validate()

    def _get_target_altitude(self, obs, grp, radec):
        transformAltaz = AltAz(location=obs, obstime=grp["mjd"])
        alt = radec.transform_to(transformAltaz).alt  # * u.deg
        return alt

    def troposphere_delay(self, toas, model):
        # must include model to get ra/dec of target, plus maybe proper motion?
        tab = toas.table

        radec = SkyCoord(
            model.RAJ.value * model.RAJ.units, model.DECJ.value * model.DECJ.units
        )  # just do this once instead of adjusting over time

        # okie so I need to do this more efficiently, so i'll group by the observatory

        # for loop copied from solary_system_shapiro_delay
        tbl = toas.table
        delay = np.zeros(len(tbl))
        for ii, key in enumerate(tbl.groups.keys):
            grp = tbl.groups[ii]
            loind, hiind = tbl.groups.indices[ii : ii + 2]
            if key["obs"].lower() == "barycenter":
                log.debug("Skipping Troposphere delay for Barycentric TOAs")
                continue

            obs = get_observatory(tbl.groups.keys[ii]["obs"]).earth_location_itrf()

            alt = self._get_target_altitude(obs, grp, radec)

            # now actually calculate the atmospheric delay based on the models
            # start with 7.7 ns and plane atmosphere

            # delay[loind:hiind] = 7.7 * u.nanosecond / np.sin(alt)

            delay[loind:hiind] = self.delay_model(
                alt, obs.lat, obs.height, grp["tdbld"]
            )

        return delay

    def delay_model(self, alt, lat, H, mjd):

        return self.zenith_delay(lat, H.to(u.km)) * self.mapping_function(
            alt, lat, H, mjd
        )

    def zenith_delay(self, lat, H):
        # return 7.7 * u.ns

        p = 101
        return (
            (p / 43.921)
            / (const.c.value * (1 - 0.00266 * np.cos(2 * lat) + 0.00028 * H.value))
            * u.second
        )

    def _coefficient_func(self, average, amplitudes, yearFraction):
        return average + 0 * amplitudes * np.cos(2 * np.pi * yearFraction)

    def _find_latitude_index(self, lat):
        """
        find the index corresponding to the upper bound on latitude
        for nearest neighbor interpolation in the mapping function
        """
        absLat = np.abs(lat)
        for lInd in range(len(self.LAT)):
            if absLat >= self.LAT[lInd - 1]:
                return lInd
        # else this is an invalid latitude... huh?
        raise ValueError("Invaid latitude: %s must be between -90 and 90 degrees" % lat)

    def mapping_function(self, alt, lat, H, mjd):

        # yearFraction = np.mod(jyear, 1)
        # for now just set the year fraction to 0 and don't change anything
        # yearFraction = np.zeros(len(mjd))  # come back and fix this later

        yearFraction = self._get_year_fraction_fast(mjd, lat)
        """
        according to Niell, the way to use latitude interpolation is to interpolate
        between the nearest definite latitude coefficient functions.  So that means
        I need to calcualte the function, then I can go back and interpolate between the results.
        I figure the easiest way to do this will be to calculate the function on the entire array
        """

        # first I need to find the nearest latitude neighbors
        latIndex = self._find_latitude_index(lat)

        aNeighbors = [
            self._coefficient_func(self.A_AVG[i], self.A_AMP[i], yearFraction)
            for i in range(latIndex, latIndex + 2)
        ]
        bNeighbors = [
            self._coefficient_func(self.B_AVG[i], self.B_AMP[i], yearFraction)
            for i in range(latIndex, latIndex + 2)
        ]
        cNeighbors = [
            self._coefficient_func(self.C_AVG[i], self.C_AMP[i], yearFraction)
            for i in range(latIndex, latIndex + 2)
        ]

        # now time to interpolate between them
        latNeighbors = self.LAT[latIndex : latIndex + 2]

        a = self._interp(np.abs(lat), latNeighbors, aNeighbors)
        b = self._interp(np.abs(lat), latNeighbors, bNeighbors)
        c = self._interp(np.abs(lat), latNeighbors, cNeighbors)

        # now finally the mapping formula
        # return 1 / np.sin(alt)

        baseMap = 1 / (
            (1 / (1 + a / (1 + b / (1 + c))))
            / (1 / (np.sin(alt) + a / (np.sin(alt) + b / (np.sin(alt) + c))))
        )

        # now add in the mapping correction based on height
        fcorrection = 1 / (
            (1 / (1 + self.A_HT / (1 + self.B_HT / (1 + self.C_HT))))
            / (
                1
                / (
                    np.sin(alt)
                    + self.A_HT / (np.sin(alt) + self.B_HT / (np.sin(alt) + self.C_HT))
                )
            )
        )

        return baseMap + (1 / np.sin(alt) - fcorrection) * H.to(u.km).value

        """
        for i in range(len(tab["obs"])):
            obs = get_observatory(tab["obs"][i]).earth_location_itrf()
            transAltaz = AltAz(location=obs, obstime=tab["mjd"][i])
            altaz = radec.transform_to(transAltaz)
            altitudes[i] = altaz.alt.value
        return altitudes
        """

    @staticmethod
    def _interp(x, xn, yn):
        # vectorized 1d interpolation for 2 points only
        return (x - xn[0]) * (yn[1] - yn[0]) / (xn[1] - xn[0]) + yn[0]

    def _get_year_fraction_slow(self, mjd, lat):
        """
        use python for loop and astropy to calculate the year fraction
        but it's more slow because of the looping
        """

        seasonOffset = 0.0
        if lat < 0:
            seasonOffset = 0.5

        yearFraction = np.array(
            [(i.jyear + seasonOffset + self.DOY_OFFSET / 365.25) % 1.0 for i in mjd]
        )
        return yearFraction

    def _get_year_fraction_fast(self, tdbld, lat):
        """
        use numpy array arithmetic to calculate the year fraction more quickly
        """
        seasonOffset = 0.0
        if lat < 0:
            seasonOffset = 0.5
        return np.mod(
            2000.0 + (tdbld - 51544.5 + self.DOY_OFFSET) / (365.25) + seasonOffset, 1.0
        )
