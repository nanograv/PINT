"""Delay due to Earth's troposphere"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.interpolate
from astropy.coordinates import AltAz, SkyCoord
from loguru import logger as log

from pint.models.parameter import boolParameter
from pint.models.timing_model import DelayComponent
from pint.observatory import get_observatory
from pint.observatory.topo_obs import TopoObs


class TroposphereDelay(DelayComponent):
    """Model for accounting for the troposphere delay for topocentric TOAs.

    Based on Davis zenith hydrostatic delay (Davis et al., 1985, Appendix A)
    Niell Mapping Functions (Niell, 1996, Eq 4)
    additional altitude correction to atmospheric pressure
    from CRC Handbook Chapter 14 page 19 "US Standard Atmosphere"

    The Zenith delay is the actual time delay for radio waves arriving directly
    overhead the observatory.  The mapping function is a dimensionless number that
    scales the zenith delay to recover the correct delay for sources anywhere else
    in the sky (closer to the horizon).

    The hydrostatic delay is best described as the relatively non-changing component to the
    delay, depending primarily on atmospheric pressure.
    The wet delay represents changes due to dynamical variation in the
    atmosphere (ie changing water vapor) and is usually around 10% of the hydrostatic delay

    Parameters supported:

    .. paramtable::
        :class: pint.models.troposphere_delay.TroposphereDelay
    """

    register = True
    category = "troposphere"  # is this the correct category?

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

    AW = (
        np.array([0.0, 5.8021897, 5.6794847, 5.8118019, 5.9727542, 6.1641693, 0.0])
        * 1e-4
    )
    BW = (
        np.array([0.0, 1.4275268, 1.5138625, 1.4572752, 1.5007428, 1.7599082, 0.0])
        * 1e-3
    )
    CW = (
        np.array([0.0, 4.3472961, 4.6729510, 4.3908931, 4.4626982, 5.4736038, 0.0])
        * 1e-2
    )

    LAT = np.array([0, 15, 30, 45, 60, 75, 90]) * u.deg  # in degrees

    DOY_OFFSET = -28  # add this into the MJD value to get the right phase

    EARTH_R = 6356766 * u.m  # earth radius at 45 degree latitude

    @staticmethod
    def _herring_map(alt, a, b, c):
        """equation 4 from the Niell mapping function.
        It is a modification to the plane-parallel atmosphere model (1 / sin(alt))
        The coefficients a b and c provide the correction for the correct map
        near the horizon, while still producing the correct mapping at zenith (1)
        """
        sinAlt = np.sin(alt)
        return 1 / (
            (1 / (1 + a / (1 + b / (1 + c))))
            / (1 / (sinAlt + a / (sinAlt + b / (sinAlt + c))))
        )

    def __init__(self):
        super().__init__()
        self.add_param(
            boolParameter(
                name="CORRECT_TROPOSPHERE",
                value="Y",
                description="Enable Troposphere Delay Model",
            )
        )

        self.delay_funcs_component += [self.troposphere_delay]

        # copy over the arrays to provide constant values within 15 deg
        # of the poles and equator
        for array in [
            self.A_AVG,
            self.B_AVG,
            self.C_AVG,
            self.A_AMP,
            self.B_AMP,
            self.C_AMP,
            self.AW,
            self.BW,
            self.CW,
        ]:
            array[0] = array[1]
            array[-1] = array[-2]

    def _get_target_altitude(self, obs, grp, radec):
        """convert the sky coordinates of the target to the angular altitude at each TOA"""
        transformAltaz = AltAz(location=obs, obstime=grp["mjd"])
        alt = radec.transform_to(transformAltaz).alt  # * u.deg
        return alt

    def _get_target_skycoord(self):
        """return the sky coordinates for the target, either from equatorial or ecliptic coordinates"""
        try:
            radec = SkyCoord(
                self._parent.RAJ.value * self._parent.RAJ.units,
                self._parent.DECJ.value * self._parent.DECJ.units,
            )  # just do this once instead of adjusting over time
        except AttributeError:
            radec = SkyCoord(
                self._parent.ELONG.value * self._parent.ELONG.units,
                self._parent.ELAT.value * self._parent.ELAT.units,
                frame="barycentricmeanecliptic",
            )
        return radec

    def troposphere_delay(self, toas, acc_delay=None):
        """This is the main function for the troposphere delay.
        Pass in the TOAs and it will calculate the delay for each TOA,
        accounting for the observatory location, target coordinates, and time of observation
        """
        tbl = toas.table
        delay = np.zeros(len(tbl))

        # if not correcting for troposphere, return the default zero delay
        if self.CORRECT_TROPOSPHERE.value:
            radec = self._get_target_skycoord()

            # the only python for loop is to iterate through the unique observatory locations
            # all other math is computed through numpy
            for key, grp in toas.get_obs_groups():
                obsobj = get_observatory(key)

                # exclude non topocentric observations
                if not isinstance(obsobj, TopoObs):
                    log.debug(
                        f"Skipping Troposphere delay for non Topocentric TOA: {obsobj.name}"
                    )
                    continue

                obs = obsobj.earth_location_itrf()

                alt = self._get_target_altitude(obs, tbl[grp], radec)

                # now actually calculate the atmospheric delay based on the models

                delay[grp] = self.delay_model(
                    alt, obs.lat, obs.height, tbl[grp]["tdbld"]
                )
        return delay * u.s

    def _validate_altitudes(self, alt, obs=""):
        """This method checks if any of the TOAs occur at invalid altitudes
        for example, if the pulsar position is incorrect, it would likely
        result in negative altitudes.
        To correct for this is two steps: first make a numpy boolean array
        to store whether each TOA is valid or not, to let me know for later.
        The boolean array is returned at the end of the function.
        Then, to allow for fast numpy math, correct the individual invalid TOAs
        to make them appear at the zenith, then afterwards make that part of
        the delay be zero.
        This altitude correction is applied to the alt numpy array passed in as argument
        optionally pass obs to list which observatory the invalid altitudes are from

        This has been tested and it does work, even though it's slightly convoluted
        """
        isPositive = np.greater_equal(alt, 0 * u.deg)
        isLessThan90 = np.less_equal(alt, 90 * u.deg)
        isValid = np.logical_and(isPositive, isLessThan90)

        # now make corrections to alt based on the valid status
        # if not valid, make them appear at the zenith to make the math sensible
        if not np.all(isValid):
            # it's probably helpful to count how many are invalid
            numInvalid = len(isValid) - np.count_nonzero(isValid)
            message = "Invalid altitude calculated for %i TOAS" % numInvalid
            if obs:
                message += f" from observatory {obs}"
            log.warning(message)

            # now correct the values
            # first make the invalid altitudes zeros
            alt *= isValid  # multiply valids by 1, else make zero
            alt += (
                90 * u.deg * np.logical_not(isValid)
            )  # increase the invalid ones to 90 deg (zenith)
            # this will prevent unexpected behavior from occurring for negative altitudes
        return isValid

    def delay_model(self, alt, lat, H, mjd):
        """validate the observed altitudes, then combine dry and wet delays"""
        # make sure the altitudes are reasonable values, warn if not
        altIsValid = self._validate_altitudes(alt)

        delay = self.zenith_delay(lat, H.to(u.km)) * self.mapping_function(
            alt, lat, H, mjd
        ) + self.wet_zenith_delay() * self.wet_map(alt, lat)

        # modify the delay if any of the altitudes are invalid
        if not np.all(altIsValid):
            delay *= altIsValid  # this will make the invalid delays zero
        return delay

    def pressure_from_altitude(self, H):
        """From CRC Handbook Chapter 14 page 19 US Standard Atmosphere"""
        gph = self.EARTH_R * H / (self.EARTH_R + H)  # geopotential height
        if gph > 11 * u.km:
            log.warning("Pressure approximation invalid for elevations above 11 km")
        T = 288.15 - 0.0065 * H.to(u.m).value  # temperature lapse
        return 101.325 * (288.15 / T) ** -5.25575 * u.kPa

    def zenith_delay(self, lat, H):
        """Calculate the hydrostatic zenith delay"""
        p = self.pressure_from_altitude(H)
        return (p / (43.921 * u.kPa)) / (
            const.c.value * (1 - 0.00266 * np.cos(2 * lat) - 0.00028 * H.value)
        )

    def wet_zenith_delay(self):
        """calculate the wet delay at zenith"""
        return 0.0  # this method will be updated in the future to
        # either allow explicit specification of the wet zenith delay
        # or approximate it from weather data
        # default for TEMPO2 is zero wet delay if not specified

    def _coefficient_func(self, average, amplitudes, yearFraction):
        """from the Niell mapping function with annual variations"""
        return average + amplitudes * np.cos(2 * np.pi * yearFraction)

    def _find_latitude_index(self, lat):
        """find the index corresponding to the upper bound on latitude
        for nearest neighbor interpolation in the mapping function
        """
        absLat = np.abs(lat)
        for lInd in range(1, len(self.LAT)):
            if absLat <= self.LAT[lInd]:
                return lInd - 1
        # else this is an invalid latitude... huh?
        raise ValueError(f"Invaid latitude: {lat} must be between -90 and 90 degrees")

    def mapping_function(self, alt, lat, H, mjd):
        """this implements the Niell mapping function for hydrostatic delays"""

        yearFraction = self._get_year_fraction_fast(mjd, lat)
        """
        according to Niell, the way to use latitude interpolation is to interpolate
        between the nearest definite latitude coefficient functions.  So that means
        I need to calculate the function, then I can go back and interpolate between the results.
        I figure the easiest way to do this will be to calculate the function on the entire array
        """

        # first I need to find the nearest latitude neighbors
        latIndex = self._find_latitude_index(lat)

        aNeighbors = np.array(
            [
                self._coefficient_func(self.A_AVG[i], self.A_AMP[i], yearFraction)
                for i in range(latIndex, latIndex + 2)
            ]
        ).transpose()
        bNeighbors = np.array(
            [
                self._coefficient_func(self.B_AVG[i], self.B_AMP[i], yearFraction)
                for i in range(latIndex, latIndex + 2)
            ]
        ).transpose()
        cNeighbors = np.array(
            [
                self._coefficient_func(self.C_AVG[i], self.C_AMP[i], yearFraction)
                for i in range(latIndex, latIndex + 2)
            ]
        ).transpose()

        # now time to interpolate between them
        latNeighbors = self.LAT[latIndex : latIndex + 2]

        a = self._interp(np.abs(lat), latNeighbors, aNeighbors)
        b = self._interp(np.abs(lat), latNeighbors, bNeighbors)
        c = self._interp(np.abs(lat), latNeighbors, cNeighbors)

        # the base mapping function
        baseMap = self._herring_map(alt, a, b, c)

        # now add in the mapping correction based on height
        fcorrection = self._herring_map(alt, self.A_HT, self.B_HT, self.C_HT)

        return baseMap + (1 / np.sin(alt) - fcorrection) * H.to(u.km).value

    def wet_map(self, alt, lat):
        """This is very similar to the normal mapping function except it uses different
        coefficients.  In addition, there is no height correction.  From Niell (1996):

        "This does not apply to the wet mapping function since the
        water vapor is not in hydrostatic equilibrium, and the
        height distribution of the water vapor is not expected
        to be predictable from the station height"
        """

        latIndex = self._find_latitude_index(lat)  # latitude dependent

        aNeighbors = self.AW[latIndex : latIndex + 2]
        bNeighbors = self.BW[latIndex : latIndex + 2]
        cNeighbors = self.CW[latIndex : latIndex + 2]

        latNeighbors = self.LAT[latIndex : latIndex + 2]

        a = self._interp(np.abs(lat), latNeighbors, aNeighbors)
        b = self._interp(np.abs(lat), latNeighbors, bNeighbors)
        c = self._interp(np.abs(lat), latNeighbors, cNeighbors)

        return self._herring_map(alt, a, b, c)

    @staticmethod
    def _interp(x, xn, yn):
        """vectorized 1d interpolation for 2 points only"""
        # return (x - xn[0]) * (yn[1] - yn[0]) / (xn[1] - xn[0]) + yn[0]
        f = scipy.interpolate.interp1d(xn, yn)
        return f(x)

    def _get_year_fraction_slow(self, mjd, lat):
        """
        use python for loop and astropy to calculate the year fraction
        but it's more slow because of the looping
        """

        seasonOffset = 0.5 if lat < 0 else 0.0
        return np.array(
            [(i.jyear + seasonOffset + self.DOY_OFFSET / 365.25) % 1.0 for i in mjd]
        )

    def _get_year_fraction_fast(self, tdbld, lat):
        """
        use numpy array arithmetic to calculate the year fraction more quickly
        """
        seasonOffset = 0.5 if lat < 0 else 0.0
        return np.mod(
            2000.0 + (tdbld - 51544.5 + self.DOY_OFFSET) / (365.25) + seasonOffset, 1.0
        )
