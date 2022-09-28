"""Dispersion due to the solar wind."""
from warnings import warn

import astropy.constants as const
import astropy.units as u
import numpy as np
import scipy.special

from pint.models.dispersion_model import Dispersion, DMconst
from pint.models.parameter import floatParameter


def _dm_p_int(b, z, p):
    """Integral function for DM calculation
    from https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/chromatic/solar_wind.py#L299

    See Figure 1 of Hazboun et al. (2022) for definitions of b, z

    Parameters
    ----------
    b : astropy.quantity.Quantity
        Impact parameter
    z : astropy.quantity.Quantity
        distance from Earth to closest point to the Sun
    p : power-law index

    Returns
    -------
    astropy.quantity.Quantity
    """
    return (z / b) * scipy.special.hyp2f1(
        0.5, p / 2.0, 1.5, -((z**2) / b**2).decompose().value
    )


class SolarWindDispersion(Dispersion):
    """Dispersion due to the solar wind (basic model).

    The model is a simple spherically-symmetric model that is fit
    only in its constant amplitude.

    For ``SWM==0`` it assumes a power-law index of 2 (Edwards et al.)

    For ``SWM==1`` it can have any power-law index (You et al., Hazboun et al.)

    Parameters supported:

    .. paramtable::
        :class: pint.models.solar_wind_dispersion.SolarWindDispersion

    References
    ----------
    Edwards et al. 2006, MNRAS, 372, 1549; Setion 2.5.4
    Madison et al. 2019, ApJ, 872, 150; Section 3.1.
    Hazboun et al. (2022, ApJ, 929, 39)
    You et al. (2012, MNRAS, 422, 1160)
    """

    register = True
    category = "solar_wind"

    def __init__(self):
        super().__init__()
        self.add_param(
            floatParameter(
                name="NE_SW",
                units="cm^-3",
                value=0.0,
                aliases=["NE1AU", "SOLARN0"],
                description="Solar Wind density at 1 AU",
            )
        )
        self.add_param(
            floatParameter(
                name="SWP",
                value=2.0,
                units="",
                description="Solar Wind Model radial power-law index (only for SWM=1)",
            )
        )
        self.add_param(
            floatParameter(
                name="SWM",
                value=0.0,
                units="",
                description="Solar Wind Model (0 is from Edwards+ 2006, 1 is from You+2007,2012/Hazboun+ 2022)",
            )
        )
        self.dm_value_funcs += [self.solar_wind_dm]
        self.delay_funcs_component += [self.solar_wind_delay]
        self.set_special_params(["NE_SW", "SWM", "SWP"])

    def setup(self):
        super().setup()
        self.register_dm_deriv_funcs(self.d_dm_d_ne_sw, "NE_SW")
        self.register_deriv_funcs(self.d_delay_d_ne_sw, "NE_SW")

    def solar_wind_geometry(self, toas):
        """Return the geometry of solar wind dispersion.

        For SWM==0:
            Implements the geometry part of equations 29, 30 of Edwards et al. 2006,
            (i.e., without the n0, the solar wind DM amplitude part.)
            Their rho is given as theta here.

            rvec: radial vector from observatory to the center of the Sun
            pos: pulsar position
        For SWM==1:
            Implements Eqn. 12 of Hazboun et al. (2022)

        Parameters
        ----------
        toas : pint.toa.TOAs

        Returns
        -------
        astropy.quantity.Quantity
        """
        if self.SWM.value == 0:
            angle, r = self._parent.sun_angle(toas, also_distance=True)
            rho = np.pi - angle.value
            solar_wind_geometry = const.au**2.0 * rho / (r * np.sin(rho))
            return solar_wind_geometry
        elif self.SWM.value == 1:
            p = self.SWP.value
            # get elongation angle, distance from Earth to Sun
            theta, r = self._parent.sun_angle(toas, also_distance=True)
            # impact parameter
            b = r * np.sin(theta)
            # distance from the Earth to the impact point
            z_sun = r * np.cos(theta)
            # a big value for comparison
            # this is what Enterprise uses
            z_p = (1e14 * u.s * const.c).to(b.unit)
            if p > 1:
                solar_wind_geometry = (
                    (1 / b.to_value(u.AU)) ** p
                    * b
                    * (_dm_p_int(b, z_p, p) - _dm_p_int(b, -z_sun, p))
                )
            else:
                raise NotImplementedError(
                    "Solar Dispersion Delay not implemented for power-law index p <= 1"
                )
            return solar_wind_geometry
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )

    def solar_wind_dm(self, toas):
        """Return the solar wind dispersion measure.

        SWM==0:
            Uses equations 29, 30 of Edwards et al. 2006.
        SWM==1:
            Hazboun et al. 2022
        """
        if self.NE_SW.value == 0:
            return np.zeros(len(toas)) * u.pc / u.cm**3
        if self.SWM.value == 0 or self.SWM.value == 1:
            solar_wind_geometry = self.solar_wind_geometry(toas)
            solar_wind_dm = self.NE_SW.quantity * solar_wind_geometry
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )
        return solar_wind_dm.to(u.pc / u.cm**3)

    def solar_wind_delay(self, toas, acc_delay=None):
        """This is a wrapper function to compute solar wind dispersion delay."""
        if self.NE_SW.value == 0:
            return np.zeros(len(toas)) * u.s
        return self.dispersion_type_delay(toas)

    def d_dm_d_ne_sw(self, toas, param_name, acc_delay=None):
        """Derivative of of DM wrt the solar wind dm amplitude."""
        if self.SWM.value == 0 or self.SWM.value == 1:
            solar_wind_geometry = self.solar_wind_geometry(toas)
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )
        return solar_wind_geometry

    def d_delay_d_ne_sw(self, toas, param_name, acc_delay=None):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]
        deriv = self.d_delay_d_dmparam(toas, "NE_SW")
        deriv[bfreq < 1.0 * u.MHz] = 0.0
        return deriv

    def print_par(self, format="pint"):
        result = ""
        result += getattr(self, "NE_SW").as_parfile_line(format=format)
        result += getattr(self, "SWM").as_parfile_line(format=format)
        if self.SWM.value == 1:
            result += getattr(self, "SWP").as_parfile_line(format=format)
        return result
