"""Dispersion due to the solar wind."""
from warnings import warn

import astropy.constants as const
import astropy.units as u
import astropy.time
import numpy as np
import scipy.special

from pint.models.dispersion_model import Dispersion
from pint.models.parameter import floatParameter, prefixParameter
import pint.utils
from pint.models.timing_model import MissingTOAs
from pint.toa_select import TOASelect


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
    p : float
        power-law index

    Returns
    -------
    astropy.quantity.Quantity
    """
    return (z / b) * scipy.special.hyp2f1(
        0.5, p / 2.0, 1.5, -((z**2) / b**2).decompose().value
    )


def _gamma_function(p):
    """The second term in Eqn. 12 of Hazboun et al. involving Gamma functions
    Note that this needs to be multiplied by b*sqrt(pi)/2

    Parameters
    ----------
    p : float
        power-law index

    Returns
    -------
    float
    """
    return scipy.special.gamma(p / 2 - 0.5) / scipy.special.gamma(p / 2)


def _d_gamma_function_dp(p):
    """Derivative of the second term in Eqn. 12 of Hazboun et al. involving Gamma functions wrt p
    Note that this needs to be multiplied by b*sqrt(pi)/2

    Parameters
    ----------
    p : float
        power-law index

    Returns
    -------
    float
    """
    return scipy.special.gamma(p / 2 - 0.5) * scipy.special.polygamma(
        0, p / 2 - 0.5
    ) / 2 / scipy.special.gamma(p / 2) - scipy.special.gamma(
        p / 2 - 0.5
    ) * scipy.special.polygamma(
        0, p / 2
    ) / 2 / scipy.special.gamma(
        p / 2
    )


def _hypergeom_function(b, z, p):
    """The first term in Eqn. 12 of Hazboun et al. involving hypergeometric functions
    Note that this needs to be multiplied by b

    Parameters
    ----------
    b : astropy.quantity.Quantity
        Impact parameter
    z : astropy.quantity.Quantity
        distance from Earth to closest point to the Sun
    p : float
        power-law index

    Returns
    -------
    astropy.quantity.Quantity
    """
    theta = np.arctan2(b, z)
    return (1 / np.tan(theta)) * scipy.special.hyp2f1(
        0.5, p / 2.0, 1.5, -((1 / np.tan(theta)).value ** 2)
    )


def _d_hypergeom_function_dp(b, z, p):
    """Derivative of the first term in Eqn. 12 of Hazboun et al. involving hypergeometric functions wrt p
    Note that this needs to be multiplied by b

    Parameters
    ----------
    b : astropy.quantity.Quantity
        Impact parameter
    z : astropy.quantity.Quantity
        distance from Earth to closest point to the Sun
    p : float
        power-law index

    Returns
    -------
    astropy.quantity.Quantity
    """
    theta = np.arctan2(b, z)
    x = theta.to_value(u.rad) - np.pi / 2
    # the result of a order 4,4 Pade expansion of
    # cot(theta) * hypergeom([1/2, p/2],[3/2],-cot(theta)**2)
    # near theta=Pi/2
    # in Maple:
    # with(numapprox):
    # with(CodeGeneration):
    # Python(simplify(subs(theta = x + Pi/2, diff(pade(cot(theta)*hypergeom([1/2, p/2], [3/2], -cot(theta)^2), theta = Pi/2, [4, 4]), p))));
    return (
        8580
        * x**3
        * (
            (
                p**4
                - 0.76e2 / 0.11e2 * p**3
                + 0.2996e4 / 0.429e3 * p**2
                + 0.5248e4 / 0.429e3 * p
                - 0.1984e4 / 0.429e3
            )
            * x**4
            + 0.84e2
            / 0.11e2
            * (p**2 - 0.115e3 / 0.39e2 * p - 0.44e2 / 0.39e2)
            * (p + 4)
            * x**2
            + 0.1960e4 / 0.143e3 * (p + 4) ** 2
        )
        / (
            39 * x**4 * p**3
            - 186 * x**4 * p**2
            + 200 * x**4 * p
            + 360 * x**2 * p**2
            + 32 * x**4
            - 480 * x**2 * p
            - 1440 * x**2
            + 840 * p
            + 3360
        )
        ** 2
    )


def _solar_wind_geometry(r, theta, p):
    """Solar wind geometry factor (integral of path length)

    For the models with variable power-law index (You et al., Hazboun et al.)

    Parameters
    ----------
    r : astropy.quantity.Quantity
        Distance from the Earth to the Sun
    theta : astropy.quantity.Quantity
        Solar elongation
    p : float
        Power-law index

    Returns
    -------
    astropy.quantity.Quantity
    """
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
            "Solar Wind geometry not implemented for power-law index p <= 1"
        )
    return solar_wind_geometry


def _d_solar_wind_geometry_d_p(r, theta, p):
    """Derivative of solar_wind_geometry (path length) wrt power-law index p

    The evaluation is done using Eqn. 12 in Hazboun et al. (2022).  The first term
    involving hypergeometric functions (:func:`_hypergeom_function`)
    has the derivative computed approximately using a Pade expansion (:func:`_d_hypergeom_function_dp`).
    The second uses gamma functions (:func:`_gamma_function`) and has the derivative computed
    using polygamma functions (:func:`_d_gamma_function_dp`).

    """
    # impact parameter
    b = r * np.sin(theta)
    # distance from the Earth to the impact point
    z_sun = r * np.cos(theta)
    # a big value for comparison
    # this is what Enterprise uses
    z_p = (1e14 * u.s * const.c).to(b.unit)
    if p > 1:
        return (1 / b.to_value(u.AU)) ** p * (
            b * _d_hypergeom_function_dp(b, z_sun, p)
            + (b * np.sqrt(np.pi) / 2) * _d_gamma_function_dp(p)
        ) - (1 / b.to_value(u.AU)) ** p * np.log(b.to_value(u.AU)) * (
            b * _hypergeom_function(b, z_sun, p)
            + (b * np.sqrt(np.pi) / 2) * _gamma_function(p)
        )
    else:
        return np.inf * np.ones(len(theta)) * u.pc


def _get_reference_time(
    model,
    params=["POSEPOCH", "PEPOCH", "DMEPOCH"],
    default=astropy.time.Time(50000, format="mjd"),
):
    """Return a reference time for other calculations

    Go through a list of possible times in a model, and return the first one that is not None.

    If none is found, return the default.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
    params : list
        Names of parameters to search through
    default : astropy.time.Time

    Returns
    -------
    astropy.time.Time
    """
    for p in params:
        if getattr(model, p).value is not None:
            return getattr(model, p).quantity
    return default


class SolarWindDispersionBase(Dispersion):
    """Abstract base class for solar wind dispersion components."""

    pass


class SolarWindDispersion(SolarWindDispersionBase):
    """Dispersion due to the solar wind (basic model).

    The model is a simple spherically-symmetric model that is fit
    in its constant amplitude.

    For ``SWM==0`` it assumes a power-law index of 2 (Edwards et al.)

    For ``SWM==1`` it can have any power-law index (You et al., Hazboun et al.), which can also be fit

    Parameters supported:

    .. paramtable::
        :class: pint.models.solar_wind_dispersion.SolarWindDispersion

    References
    ----------
    - Edwards et al. 2006, MNRAS, 372, 1549; Setion 2.5.4
    - Madison et al. 2019, ApJ, 872, 150; Section 3.1.
    - Hazboun et al. (2022, ApJ, 929, 39)
    - You et al. (2012, MNRAS, 422, 1160)
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
        self.register_dm_deriv_funcs(self.d_dm_d_swp, "SWP")
        self.register_deriv_funcs(self.d_delay_d_swp, "SWP")

    def solar_wind_geometry(self, toas):
        """Return the geometry of solar wind dispersion.

        For SWM==0:
            Implements the geometry part of equations 29, 30 of Edwards et al. 2006,
            (i.e., without the n0, the solar wind DM amplitude part.)
            Their rho is given as theta here.

            rvec: radial vector from observatory to the center of the Sun
            pos: pulsar position
        For SWM==1:
            Implements Eqn. 11 of Hazboun et al. (2022)

        Parameters
        ----------
        toas : pint.toa.TOAs

        Returns
        -------
        astropy.quantity.Quantity
        """
        swm = self.SWM.value
        p = self.SWP.value

        if swm == 0:
            angle, r = self._parent.sun_angle(toas, also_distance=True)
            rho = np.pi - angle.to_value(u.rad)
            solar_wind_geometry = const.au**2.0 * rho / (r * np.sin(rho))
            return solar_wind_geometry.to(u.pc)
        elif swm == 1:
            # get elongation angle, distance from Earth to Sun
            theta, r = self._parent.sun_angle(toas, also_distance=True)
            return _solar_wind_geometry(r, theta, p).to(u.pc)
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % swm
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
        if self.SWM.value not in [0, 1]:
            raise NotImplementedError(
                f"Solar Dispersion Delay not implemented for SWM {self.SWM.value}"
            )
        solar_wind_geometry = self.solar_wind_geometry(toas)
        solar_wind_dm = self.NE_SW.quantity * solar_wind_geometry
        return solar_wind_dm.to(u.pc / u.cm**3)

    def solar_wind_delay(self, toas, acc_delay=None):
        """This is a wrapper function to compute solar wind dispersion delay."""
        if self.NE_SW.value == 0:
            return np.zeros(len(toas)) * u.s
        return self.dispersion_type_delay(toas)

    def d_dm_d_ne_sw(self, toas, param_name, acc_delay=None):
        """Derivative of of DM wrt the solar wind ne amplitude."""
        if self.SWM.value in [0, 1]:
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

    def d_solar_wind_geometry_d_swp(self, toas, param_name, acc_delay=None):
        """Derivative of solar_wind_geometry (path length) wrt power-law index p

        The evaluation is done using Eqn. 12 in Hazboun et al. (2022).  The first term
        involving hypergeometric functions (:func:`_hypergeom_function`)
        has the derivative computed approximately using a Pade expansion (:func:`_d_hypergeom_function_dp`).
        The second uses gamma functions (:func:`_gamma_function`) and has the derivative computed
        using polygamma functions (:func:`_d_gamma_function_dp`).

        """
        if self.SWM.value == 0:
            raise ValueError(
                "Solar Wind power-law index not valid for SWM %d" % self.SWM.value
            )
        elif self.SWM.value == 1:
            # get elongation angle, distance from Earth to Sun
            theta, r = self._parent.sun_angle(toas, also_distance=True)
            return _d_solar_wind_geometry_d_p(r, theta, self.SWP.value)
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )

    def d_dm_d_swp(self, toas, param_name, acc_delay=None):
        d_geometry_dp = self.d_solar_wind_geometry_d_swp(
            toas, param_name, acc_delay=acc_delay
        )
        return self.NE_SW.quantity * d_geometry_dp

    def d_delay_d_swp(self, toas, param_name, acc_delay=None):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]
        deriv = self.d_delay_d_dmparam(toas, "SWP")
        deriv[bfreq < 1.0 * u.MHz] = 0.0
        return deriv

    def print_par(self, format="pint"):
        result = ""
        result += getattr(self, "NE_SW").as_parfile_line(format=format)
        result += getattr(self, "SWM").as_parfile_line(format=format)
        if self.SWM.value == 1:
            result += getattr(self, "SWP").as_parfile_line(format=format)
        return result

    def get_max_dm(self):
        """Return approximate maximum DM from the Solar Wind (at conjunction)

        Simplified model that assumes a circular orbit

        Returns
        -------
        astropy.quantity.Quantity
        """
        coord = self._parent.get_psr_coords()
        t0 = _get_reference_time(self._parent)
        # time and distance of conjunction
        t0, elongation = pint.utils.get_conjunction(coord, t0, precision="high")
        if self.SWM.value == 0:
            r = 1 * u.AU
            rho = (180 * u.deg - elongation).to(u.rad)
            return (
                self.NE_SW.quantity * (const.au**2.0 * rho / (r * np.sin(rho)))
            ).to(u.pc / u.cm**3, equivalencies=u.dimensionless_angles())
        elif self.SWM.value == 1:
            p = self.SWP.value
            theta = elongation
            r = 1 * u.AU
            return (_solar_wind_geometry(r, theta, p) * self.NE_SW.quantity).to(
                u.pc / u.cm**3
            )
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )

    def get_min_dm(self):
        """Return approximate minimum DM from the Solar Wind (180deg away from conjunction)

        Simplified model that assumes a circular orbit

        Returns
        -------
        astropy.quantity.Quantity
        """
        coord = self._parent.get_psr_coords()
        t0 = _get_reference_time(self._parent)
        # time and distance of conjunction
        t0, elongation = pint.utils.get_conjunction(coord, t0, precision="high")
        if self.SWM.value == 0:
            r = 1 * u.AU
            rho = (elongation).to(u.rad)
            return (
                self.NE_SW.quantity * (const.au**2.0 * rho / (r * np.sin(rho)))
            ).to(u.pc / u.cm**3, equivalencies=u.dimensionless_angles())
        elif self.SWM.value == 1:
            p = self.SWP.value
            # for the min
            theta = 180 * u.deg - elongation
            r = 1 * u.AU
            return (_solar_wind_geometry(r, theta, p) * self.NE_SW.quantity).to(
                u.pc / u.cm**3
            )
        else:
            raise NotImplementedError(
                "Solar Dispersion Delay not implemented for SWM %d" % self.SWM.value
            )


class SolarWindDispersionX(SolarWindDispersionBase):
    """This class provides a SWX model - multiple Solar Wind segments.

    This model lets the user specify time ranges and fit for a different
    SWXDM (max solar wind DM) value and SWXP (radial power-law index) in each time range.

    The default radial power-law index value of 2 corresponds to the Edwards et al. model.
    Other values are for the You et al./Hazboun et al. model.

    Note that unlike the standard model, this model goes to 0 at opposition (the minimum).
    So it really represents a Delta DM.  This is to make it easier to join multiple segments.
    However, to get the peak to still be the requested max DM the values are scaled compared
    to the standard model: the standard model goes from opposition (min) to conjunction (max),
    while this model goes from 0 to conjunction (max), so the scaling is ``((conjunction - opposition)/conjuction)``.

    See `Solar Wind Examples <examples/solar_wind.html>`_.

    To compare against a standard model:

    Example
    -------

        >>> swxmodel.SWXDM_0001.quantity = model.get_max_dm() * swxmodel.get_swscalings()[0]
        >>> np.allclose(swxmodel.get_min_dms(),model.get_min_dm()*swxmodel.get_swscalings())
        >>> toas = pint.simulation.make_fake_toas_uniform(54000, 54000 + 2 * 365, 500, model=model, obs="gbt")
        >>> np.allclose((swxmodel.swx_dm(toas) + model.get_min_dm()), model.solar_wind_dm(toas))

    Parameters supported:

    .. paramtable::
        :class: pint.models.solar_wind_dispersion.SolarWindDispersionX

    References
    ----------
    - Edwards et al. 2006, MNRAS, 372, 1549; Setion 2.5.4
    - Madison et al. 2019, ApJ, 872, 150; Section 3.1.
    - Hazboun et al. (2022, ApJ, 929, 39)
    - You et al. (2012, MNRAS, 422, 1160)
    """

    register = True
    category = "solar_windx"

    def __init__(self):
        super().__init__()

        self.add_swx_range(None, None, swxdm=0, swxp=2, frozen=False, index=1)

        self.set_special_params(["SWXDM_0001", "SWXP_0001", "SWXR1_0001", "SWXR2_0001"])
        self.dm_value_funcs += [self.swx_dm]
        self.delay_funcs_component += [self.swx_delay]
        self._theta0 = None

    def solar_wind_geometry(self, toas, p):
        """Return the geometry of solar wind dispersion (integrated path-length)

        Implements Eqn. 11 of Hazboun et al. (2022)

        Parameters
        ----------
        toas : pint.toa.TOAs
        p : float
            Radial power-law index

        Returns
        -------
        astropy.quantity.Quantity
        """

        # get elongation angle, distance from Earth to Sun
        theta, r = self._parent.sun_angle(toas, also_distance=True)
        return _solar_wind_geometry(r, theta, p).to(u.pc)

    def conjunction_solar_wind_geometry(self, p):
        """Return the geometry (integrated path-length) of solar wind dispersion at conjunction

        Implements Eqn. 11 of Hazboun et al. (2022)

        Parameters
        ----------
        p : float
            Radial power-law index

        Returns
        -------
        astropy.quantity.Quantity
        """
        r0 = 1 * u.AU
        return _solar_wind_geometry(r0, self.theta0, p).to(u.pc)

    def opposition_solar_wind_geometry(self, p):
        """Return the geometry (integrated path-length) of solar wind dispersion at opposition

        Implements Eqn. 11 of Hazboun et al. (2022)

        Parameters
        ----------
        p : float
            Radial power-law index

        Returns
        -------
        astropy.quantity.Quantity
        """
        r0 = 1 * u.AU
        return _solar_wind_geometry(r0, 180 * u.deg - self.theta0, p).to(u.pc)

    def d_solar_wind_geometry_d_swxp(self, toas, p):
        """Derivative of solar_wind_geometry (path length) wrt power-law index p

        The evaluation is done using Eqn. 12 in Hazboun et al. (2022).  The first term
        involving hypergeometric functions (:func:`_hypergeom_function`)
        has the derivative computed approximately using a Pade expansion (:func:`_d_hypergeom_function_dp`).
        The second uses gamma functions (:func:`_gamma_function`) and has the derivative computed
        using polygamma functions (:func:`_d_gamma_function_dp`).

        Parameters
        ----------
        toas : pint.toa.TOAs
        p : float, optional
            Radial power-law index

        Returns
        -------
        astropy.quantity.Quantity
        """
        # get elongation angle, distance from Earth to Sun
        theta, r = self._parent.sun_angle(toas, also_distance=True)
        return _d_solar_wind_geometry_d_p(r, theta, p)

    def d_conjunction_solar_wind_geometry_d_swxp(self, p):
        """Derivative of conjunction_solar_wind_geometry (path length) wrt power-law index p

        The evaluation is done using Eqn. 12 in Hazboun et al. (2022).  The first term
        involving hypergeometric functions (:func:`_hypergeom_function`)
        has the derivative computed approximately using a Pade expansion (:func:`_d_hypergeom_function_dp`).
        The second uses gamma functions (:func:`_gamma_function`) and has the derivative computed
        using polygamma functions (:func:`_d_gamma_function_dp`).

        """
        # fiducial values
        r0 = 1 * u.AU
        return _d_solar_wind_geometry_d_p(r0, self.theta0, p)

    def d_opposition_solar_wind_geometry_d_swxp(self, p):
        """Derivative of opposition_solar_wind_geometry (path length) wrt power-law index p

        The evaluation is done using Eqn. 12 in Hazboun et al. (2022).  The first term
        involving hypergeometric functions (:func:`_hypergeom_function`)
        has the derivative computed approximately using a Pade expansion (:func:`_d_hypergeom_function_dp`).
        The second uses gamma functions (:func:`_gamma_function`) and has the derivative computed
        using polygamma functions (:func:`_d_gamma_function_dp`).

        """
        # fiducial values
        r0 = 1 * u.AU
        return _d_solar_wind_geometry_d_p(r0, 180 * u.deg - self.theta0, p)

    def add_swx_range(
        self, mjd_start, mjd_end, index=None, swxdm=0, swxp=2, frozen=True
    ):
        """Add SWX range to a dispersion model with specified start/end MJD, SWXDM, and power-law index

        Parameters
        ----------
        mjd_start : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for beginning of DMX event.
        mjd_end : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for end of DMX event.
        index : int, None
            Integer label for DMX event. If None, will increment largest used index by 1.
        swxdm : float or astropy.quantity.Quantity
            Max solar wind DM
        swxp : float or astropy.quantity.Quantity
            Solar wind power-law index
        frozen : bool
            Indicates whether SWXDM and SWXP will be fit.

        Returns
        -------
        index : int
            Index that has been assigned to new SWX event.

        """

        #### Setting up the SWX title convention. If index is None, want to increment the current max SWX index by 1.
        if index is None:
            dct = self.get_prefix_mapping_component("SWXDM_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if mjd_end is not None and mjd_start is not None:
            if mjd_end < mjd_start:
                raise ValueError("Starting MJD is greater than ending MJD.")
        elif mjd_start != mjd_end:
            raise ValueError("Only one MJD bound is set.")

        if int(index) in self.get_prefix_mapping_component("SWXDM_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another."
            )
        if isinstance(swxdm, u.quantity.Quantity):
            swxdm = swxdm.to_value(u.pc / u.cm**3)
        if isinstance(mjd_start, astropy.time.Time):
            mjd_start = mjd_start.mjd
        elif isinstance(mjd_start, u.quantity.Quantity):
            mjd_start = mjd_start.value
        if isinstance(mjd_end, astropy.time.Time):
            mjd_end = mjd_end.mjd
        elif isinstance(mjd_end, u.quantity.Quantity):
            mjd_end = mjd_end.value
        if isinstance(swxp, u.quantity.Quantity):
            swxp = swxp.value
        self.add_param(
            prefixParameter(
                name=f"SWXDM_{i}",
                units="pc cm^-3",
                value=swxdm,
                description="Max Solar Wind DM",
                parameter_type="float",
                frozen=frozen,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"SWXP_{i}",
                value=swxp,
                description="Solar wind power-law index",
                parameter_type="float",
                frozen=frozen,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"SWXR1_{i}",
                units="MJD",
                description="Beginning of SWX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_start,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"SWXR2_{i}",
                units="MJD",
                description="End of SWX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_end,
            )
        )
        self.setup()
        self.validate()
        return index

    def remove_swx_range(self, index):
        """Removes all SWX parameters associated with a given index/list of indices.

        Parameters
        ----------

        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to SWX indices to be removed from model.
        """

        if isinstance(index, (int, float, np.int64)):
            indices = [index]
        elif isinstance(index, (list, np.ndarray)):
            indices = index
        else:
            raise TypeError(
                f"index must be a float, int, list, or array - not {type(index)}"
            )
        for index in indices:
            index_rf = f"{int(index):04d}"
            for prefix in ["SWXDM_", "SWXP_", "SWXR1_", "SWXR2_"]:
                self.remove_param(prefix + index_rf)
        self.validate()

    def get_indices(self):
        """Returns an array of integers corresponding to SWX parameters.

        Returns
        -------
        inds : np.ndarray
            Array of SWX indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "SWXDM_" in p]
        return np.array(inds)

    def setup(self):
        super().setup()
        # Get SWX mapping.
        # Register the SWX derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("SWXDM_"):
                # check to make sure power-law index is present
                # if not, put in default
                p_name = f"SWXP_{pint.utils.split_prefixed_name(prefix_par)[1]}"
                if not hasattr(self, p_name):
                    self.add_param(
                        prefixParameter(
                            name=p_name,
                            value=2,
                            description="Solar wind power-law index",
                            parameter_type="float",
                        )
                    )
                self.register_deriv_funcs(self.d_delay_d_swxdm, prefix_par)
                self.register_dm_deriv_funcs(self.d_dm_d_swxdm, prefix_par)
            elif prefix_par.startswith("SWXP_"):
                self.register_deriv_funcs(self.d_delay_d_swxp, prefix_par)
                self.register_dm_deriv_funcs(self.d_dm_d_swxp, prefix_par)

    def validate(self):
        """Validate the SWX parameters."""
        super().validate()
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        SWXR1_mapping = self.get_prefix_mapping_component("SWXR1_")
        SWXR2_mapping = self.get_prefix_mapping_component("SWXR2_")
        if SWXDM_mapping.keys() != SWXP_mapping.keys():
            # FIXME: report mismatch
            raise ValueError(
                "SWXDM_ parameters do not "
                "match SWXP_ parameters. "
                "Please check your prefixed parameters."
            )
        if SWXDM_mapping.keys() != SWXR1_mapping.keys():
            # FIXME: report mismatch
            raise ValueError(
                "SWXDM_ parameters do not "
                "match SWXR1_ parameters. "
                "Please check your prefixed parameters."
            )
        if SWXDM_mapping.keys() != SWXR2_mapping.keys():
            raise ValueError(
                "SWXDM_ parameters do not "
                "match SWXR2_ parameters. "
                "Please check your prefixed parameters."
            )

    def get_theta0(self):
        """Get elongation at conjunction

        Returns
        -------
        astropy.quantity.Quantity
        """
        coord = self._parent.get_psr_coords()
        t0 = _get_reference_time(self._parent)
        t0, elongation = pint.utils.get_conjunction(coord, t0, precision="high")
        # approximate elongation at conjunction
        self._theta0 = elongation

    @property
    def theta0(self):
        if self._theta0 is None:
            self.get_theta0()
        return self._theta0

    def validate_toas(self, toas):
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        SWXR1_mapping = self.get_prefix_mapping_component("SWXR1_")
        SWXR2_mapping = self.get_prefix_mapping_component("SWXR2_")
        bad_parameters = []
        for k in SWXR1_mapping.keys():
            if self._parent[SWXDM_mapping[k]].frozen:
                continue
            b = self._parent[SWXR1_mapping[k]].quantity.mjd * u.d
            e = self._parent[SWXR2_mapping[k]].quantity.mjd * u.d
            mjds = toas.get_mjds()
            n = np.sum((b <= mjds) & (mjds < e))
            if n == 0:
                bad_parameters.append(SWXDM_mapping[k])
        if bad_parameters:
            raise MissingTOAs(bad_parameters)

    def swx_dm(self, toas):
        """Return solar wind Delta DM for given TOAs"""
        condition = {}
        p = {}
        tbl = toas.table
        if not hasattr(self, "swx_toas_selector"):
            self.swx_toas_selector = TOASelect(is_range=True)
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        SWXR1_mapping = self.get_prefix_mapping_component("SWXR1_")
        SWXR2_mapping = self.get_prefix_mapping_component("SWXR2_")
        for epoch_ind in SWXDM_mapping.keys():
            r1 = getattr(self, SWXR1_mapping[epoch_ind]).quantity
            r2 = getattr(self, SWXR2_mapping[epoch_ind]).quantity
            condition[SWXDM_mapping[epoch_ind]] = (r1.mjd, r2.mjd)
            p[SWXDM_mapping[epoch_ind]] = getattr(self, SWXP_mapping[epoch_ind]).value
        select_idx = self.swx_toas_selector.get_select_index(
            condition, tbl["mjd_float"]
        )
        # Get SWX delays
        dm = np.zeros(len(tbl)) * self._parent.DM.units
        for k, v in select_idx.items():
            if len(v) > 0:
                dmmax = getattr(self, k).quantity
                dm[v] += (
                    dmmax
                    * (
                        (
                            self.solar_wind_geometry(toas[v], p=p[k])
                            - self.opposition_solar_wind_geometry(p[k])
                        )
                        / (
                            self.conjunction_solar_wind_geometry(p[k])
                            - self.opposition_solar_wind_geometry(p[k])
                        )
                    )
                ).to(u.pc / u.cm**3)
        return dm

    def swx_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.dispersion_type_delay(toas)

    def d_dm_d_swxdm(self, toas, param_name, acc_delay=None):
        condition = {}
        tbl = toas.table
        if not hasattr(self, "swx_toas_selector"):
            self.swx_toas_selector = TOASelect(is_range=True)
        param = getattr(self, param_name)
        swx_index = param.index
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        SWXR1_mapping = self.get_prefix_mapping_component("SWXR1_")
        SWXR2_mapping = self.get_prefix_mapping_component("SWXR2_")
        p = getattr(self, SWXP_mapping[swx_index]).value
        r1 = getattr(self, SWXR1_mapping[swx_index]).quantity
        r2 = getattr(self, SWXR2_mapping[swx_index]).quantity
        condition = {param_name: (r1.mjd, r2.mjd)}
        select_idx = self.swx_toas_selector.get_select_index(
            condition, tbl["mjd_float"]
        )
        deriv = np.zeros(len(tbl)) * u.dimensionless_unscaled
        for k, v in select_idx.items():
            if len(v) > 0:
                deriv[v] += (
                    self.solar_wind_geometry(toas[v], p=p)
                    - self.opposition_solar_wind_geometry(p)
                ) / (
                    self.conjunction_solar_wind_geometry(p)
                    - self.opposition_solar_wind_geometry(p)
                )
        return deriv

    def d_delay_d_swxdm(self, toas, param_name, acc_delay=None):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]
        deriv = self.d_delay_d_dmparam(toas, param_name)
        deriv[bfreq < 1.0 * u.MHz] = 0.0
        return deriv

    def d_dm_d_swxp(self, toas, param_name, acc_delay=None):
        condition = {}
        tbl = toas.table
        # still use the SWX selectors
        if not hasattr(self, "swx_toas_selector"):
            self.swx_toas_selector = TOASelect(is_range=True)
        param = getattr(self, param_name)
        swxp_index = param.index
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        SWXR1_mapping = self.get_prefix_mapping_component("SWXR1_")
        SWXR2_mapping = self.get_prefix_mapping_component("SWXR2_")
        swxdm = getattr(self, SWXDM_mapping[swxp_index]).quantity
        p = getattr(self, SWXP_mapping[swxp_index]).value
        r1 = getattr(self, SWXR1_mapping[swxp_index]).quantity
        r2 = getattr(self, SWXR2_mapping[swxp_index]).quantity

        swx_name = f"SWXDM_{pint.utils.split_prefixed_name(param_name)[1]}"
        condition = {swx_name: (r1.mjd, r2.mjd)}
        select_idx = self.swx_toas_selector.get_select_index(
            condition, tbl["mjd_float"]
        )

        deriv = np.zeros(len(tbl)) * u.pc / u.cm**3
        for k, v in select_idx.items():
            if len(v) > 0:
                geometry = self.solar_wind_geometry(toas[v], p)
                conjunction_geometry = self.conjunction_solar_wind_geometry(p)
                opposition_geometry = self.opposition_solar_wind_geometry(p)
                d_geometry_dp = self.d_solar_wind_geometry_d_swxp(toas[v], p)
                d_conjunction_geometry_dp = (
                    self.d_conjunction_solar_wind_geometry_d_swxp(p)
                )
                d_opposition_geometry_dp = self.d_opposition_solar_wind_geometry_d_swxp(
                    p
                )
                deriv[v] += swxdm * (
                    (d_geometry_dp - d_opposition_geometry_dp)
                    / (conjunction_geometry - opposition_geometry)
                    - (geometry - opposition_geometry)
                    * (d_conjunction_geometry_dp - d_opposition_geometry_dp)
                    / (conjunction_geometry - opposition_geometry) ** 2
                )
        return deriv

    def d_delay_d_swxp(self, toas, param_name, acc_delay=None):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]
        deriv = self.d_delay_d_dmparam(toas, param_name)
        deriv[bfreq < 1.0 * u.MHz] = 0.0
        return deriv

    def print_par(self, format="pint"):
        result = ""
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        SWXR1_mapping = self.get_prefix_mapping_component("SWXR1_")
        SWXR2_mapping = self.get_prefix_mapping_component("SWXR2_")
        sorted_list = sorted(SWXDM_mapping.keys())
        for ii in sorted_list:
            result += getattr(self, SWXDM_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, SWXP_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, SWXR1_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, SWXR2_mapping[ii]).as_parfile_line(format=format)
        return result

    def get_swscalings(self):
        """Return the approximate scaling between the SWX model and the standard model

        Returns
        -------
        np.ndarray"""
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        sorted_list = sorted(SWXDM_mapping.keys())
        scalings = np.zeros(len(sorted_list))
        for j, ii in enumerate(sorted_list):
            swxdm = getattr(self, SWXDM_mapping[ii]).quantity
            p = getattr(self, SWXP_mapping[ii]).value
            scalings[j] = (
                self.conjunction_solar_wind_geometry(p)
                - self.opposition_solar_wind_geometry(p)
            ) / self.conjunction_solar_wind_geometry(p)
        return scalings

    def get_max_dms(self):
        """Return approximate maximum DMs for each segment from the Solar Wind (at conjunction)

        Simplified model that assumes a circular orbit

        Returns
        -------
        astropy.quantity.Quantity
        """
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        sorted_list = sorted(SWXDM_mapping.keys())
        dms = np.zeros(len(sorted_list)) * u.pc / u.cm**3
        for j, ii in enumerate(sorted_list):
            swxdm = getattr(self, SWXDM_mapping[ii]).quantity
            dms[j] = (swxdm).to(u.pc / u.cm**3)
        return dms

    def get_min_dms(self):
        """Return approximate minimum DMs for each segment from the Solar Wind (at opposition).

        Simplified model that assumes a circular orbit

        Note that this value has been subtracted off of the model

        Returns
        -------
        astropy.quantity.Quantity
        """
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        sorted_list = sorted(SWXDM_mapping.keys())
        dms = np.zeros(len(sorted_list)) * u.pc / u.cm**3
        for j, ii in enumerate(sorted_list):
            swxdm = getattr(self, SWXDM_mapping[ii]).quantity
            p = getattr(self, SWXP_mapping[ii]).value
            dms[j] = (
                swxdm
                * self.opposition_solar_wind_geometry(p)
                / self.conjunction_solar_wind_geometry(p)
            ).to(u.pc / u.cm**3)
        return dms

    def get_ne_sws(self):
        """Return Solar Wind electron densities at 1 AU for each segment

        Simplified model that assumes a circular orbit

        Returns
        -------
        astropy.quantity.Quantity
        """
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        sorted_list = sorted(SWXDM_mapping.keys())
        ne_sws = np.zeros(len(sorted_list)) * u.cm**-3
        for j, ii in enumerate(sorted_list):
            swxdm = getattr(self, SWXDM_mapping[ii]).quantity
            p = getattr(self, SWXP_mapping[ii]).value
            ne_sws[j] = (swxdm / self.conjunction_solar_wind_geometry(p)).to(u.cm**-3)
        return ne_sws

    def set_ne_sws(self, ne_sws):
        """Set the DMMAXs based on an input NE_SW values (electron density at 1 AU)

        Parameters
        ----------
        ne_sws : astropy.quantity.Quantity
            Desired NE_SWs (should be scalar or the same length as the number of segments)

        Raises
        ------
        ValueError : If length of input does ont match number of segments
        """
        ne_sws = np.atleast_1d(ne_sws)
        SWXDM_mapping = self.get_prefix_mapping_component("SWXDM_")
        SWXP_mapping = self.get_prefix_mapping_component("SWXP_")
        sorted_list = sorted(SWXDM_mapping.keys())
        if len(ne_sws) == 1:
            ne_sws = ne_sws[0] * np.ones(len(sorted_list))
        if len(sorted_list) != len(ne_sws):
            raise ValueError(
                f"Length of input NE_SW values ({len(ne_sws)}) must match number of SWX segments ({len(sorted_list)})"
            )
        for j, ii in enumerate(sorted_list):
            p = getattr(self, SWXP_mapping[ii]).value
            getattr(self, SWXDM_mapping[ii]).quantity = (
                self.conjunction_solar_wind_geometry(p) * ne_sws[j]
            ).to(u.pc / u.cm**3)
