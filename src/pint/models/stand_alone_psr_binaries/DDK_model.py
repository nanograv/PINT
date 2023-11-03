"""Kopeikin corrected DD model."""

import astropy.units as u
import numpy as np
import warnings

from .DD_model import DDmodel


class DDKmodel(DDmodel):
    """DDK model, a Kopeikin method corrected DD model.

    The main difference is that DDK model considers the effects on the pulsar binary parameters from the annual parallax of earth and the proper motion of the pulsar.

    From Kopeikin (1995) this includes :math:`\Delta_{\pi M}` (Equation 17), the mixed annual-orbital parallax term, which changes :math:`a_1` and :math:`\omega`
    (:meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_a1_parallax` and :meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_omega_parallax`).

    It does not include :math:`\Delta_{\pi P}`, the pure pulsar orbital parallax term (Equation 14).

    From Kopeikin (1996) this includes apparent changes in :math:`\omega`, :math:`a_1`, and :math:`i` due to the proper motion
    (:meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_omega_proper_motion`, :meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_a1_proper_motion`,
    :meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_kin_proper_motion`) (Equations 8, 9, 10).


    Special parameters are:

       KIN
            the inclination angle: :math:`i`
       KOM
            the longitude of the ascending node, Kopeikin (1995) Eq 9: :math:`\Omega`
       K96
            flag for Kopeikin binary model proper motion correction

    It also removes:

       SINI
            use ``KIN`` instead

    Notes
    -----
    This model defines KOM with reference to east, either equatorial or ecliptic depending on how the model is defined.
    KOM and KIN are defined in the Damour & Taylor (1992) convention (DT92), where

        KIN = 180 deg means the orbital angular momentum vector points toward the Earth, and KIN = 0 means the orbital angular momentum vector points away from the Earth.

        KOM is 0 toward the East and increases clockwise on the sky; it is measured "East through North."


    References
    ----------
    - Kopeikin (1995), ApJ, 439, L5 [1]_
    - Kopeikin (1996), ApJ, 467, L93 [2]_
    - Damour & Taylor (1992), Phys Rev D, 45, 1840 [3]_

    .. [1] https://ui.adsabs.harvard.edu/abs/1995ApJ...439L...5K/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/1996ApJ...467L..93K/abstract
    .. [3] https://ui.adsabs.harvard.edu/abs/1992PhRvD..45.1840D/abstract
    """

    def __init__(self, t=None, input_params=None):
        super().__init__()
        self.binary_name = "DDK"
        # Add parameter that specific for DD model, with default value and units
        self.param_default_value.update(
            {
                "KIN": 0 * u.deg,
                "PMLONG_DDK": 0 * u.mas / u.year,
                "PMLAT_DDK": 0.5 * u.mas / u.year,
                "PX": 0 * u.mas,
                "KOM": 0 * u.deg,
                "K96": True,
            }
        )

        # If any parameter has aliases, it should be updated
        # self.param_aliases.update({})
        self.binary_params = list(self.param_default_value.keys())
        # Remove unused parameter SINI
        del self.param_default_value["SINI"]
        self.set_param_values()
        if input_params is not None:
            self.update_input(param_dict=input_params)

    @property
    def KOM(self):
        return self._KOM

    @KOM.setter
    def KOM(self, val):
        self._KOM = val
        self._sin_KOM = np.sin(self._KOM)
        self._cos_KOM = np.cos(self._KOM)

    @property
    def sin_KOM(self):
        return self._sin_KOM

    @property
    def cos_KOM(self):
        return self._cos_KOM

    @property
    def psr_pos(self):
        return self._psr_pos

    @psr_pos.setter
    def psr_pos(self, val):
        """The pointing unit vector of a pulsar.
        long and lat are alpha and delta (if equatorial)
        or elong and elat (if ecliptic)
        described in
        (Kopeikin 1995 Eq 10)
        """
        self._psr_pos = val
        # it appears that this is the vector K0
        # K0 = (cos(alpha)*cos(delta), sin(alpha)*cos(delta), sin(delta))
        # in equatorial coords
        self._sin_lat = self._psr_pos[:, 2]
        self._cos_lat = np.cos(np.arcsin(self._sin_lat))
        self._sin_long = self._psr_pos[:, 1] / self._cos_lat
        self._cos_long = self._psr_pos[:, 0] / self._cos_lat

    @property
    def sin_lat(self):
        return self._sin_lat

    @property
    def cos_lat(self):
        return self._cos_lat

    @property
    def sin_long(self):
        return self._sin_long

    @property
    def cos_long(self):
        return self._cos_long

    @property
    def SINI(self):
        return np.sin(self.kin()) if hasattr(self, "_tt0") else np.sin(self.KIN)

    @SINI.setter
    def SINI(self, val):
        warnings.warn(
            "DDK model uses KIN as inclination angle. SINI will not be "
            "used. This happens every time a DDK model is constructed."
        )

    # @property
    # def SINI(self):
    #     return np.sin()
    # The code below is to apply the KOPEIKIN correction due to pulser proper motion
    # Reference:  KOPEIKIN. 1996 Eq 7 -> Eq 10.
    # Update binary parameters due to the pulser proper motion

    def delta_kin_proper_motion(self):
        """The time dependent inclination angle
        (Kopeikin 1996 Eq 10):

        .. math::

            ki = KIN + \delta_{KIN}

            \delta_{KIN} = (-\mu_{long} \sin(KOM) + \mu_{lat} \cos(KOM)) (t-T_0)

        """
        d_KIN = (
            -self.PMLONG_DDK * self.sin_KOM + self.PMLAT_DDK * self.cos_KOM
        ) * self.tt0
        return d_KIN.to(self.KIN.unit)

    def kin(self):
        return self.KIN + self.delta_kin_proper_motion() if self.K96 else self.KIN

    def d_SINI_d_KIN(self):
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        return np.cos(self.kin()).to(u.Unit("") / self.KIN.unit)

    def d_SINI_d_KOM(self):
        if not self.K96:
            return np.cos(self.kin()) * u.Unit("") / self.KOM.unit
        d_si_d_kom = (
            (-self.PMLONG_DDK * self.cos_KOM - self.PMLAT_DDK * self.sin_KOM)
            * self.tt0
            * np.cos(self.kin())
        )
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        return d_si_d_kom.to(u.Unit("") / self.KOM.unit)

    def d_SINI_d_T0(self):
        if not self.K96:
            return np.ones(len(self.tt0)) * u.Unit("") / self.T0.unit
        d_si_d_kom = -(-self.PMLONG_DDK * self.sin_KOM + self.PMLAT_DDK * self.cos_KOM)
        return d_si_d_kom.to(u.Unit("") / self.T0.unit)

    def d_SINI_d_par(self, par):
        par_obj = getattr(self, par)
        try:
            ko_func = getattr(self, f"d_SINI_d_{par}")
        except Exception:
            ko_func = lambda: np.zeros(len(self.tt0)) * u.Unit("") / par_obj.unit
        return ko_func()

    def d_kin_proper_motion_d_KOM(self):
        d_DKIN_d_KOM = (
            -self.PMLONG_DDK * self.cos_KOM - self.PMLAT_DDK * self.sin_KOM
        ) * self.tt0
        return d_DKIN_d_KOM.to(
            self.KIN.unit / self.KOM.unit, equivalencies=u.dimensionless_angles()
        )

    def d_kin_proper_motion_d_T0(self):
        d_DKIN_d_T0 = -1 * (
            -self.PMLONG_DDK * self.sin_KOM + self.PMLAT_DDK * self.cos_KOM
        )
        return d_DKIN_d_T0.to(
            self.KIN.unit / self.T0.unit, equivalencies=u.dimensionless_angles()
        )

    def d_kin_d_par(self, par):
        if par == "KIN":
            return np.ones_like(self.tt0)
        par_obj = getattr(self, par)
        if not self.K96:
            return np.zeros(len(self.tt0)) * self.KIN / par_obj.unit
        try:
            func = getattr(self, f"d_kin_proper_motion_d_{par}")
        except Exception:
            func = lambda: np.zeros(len(self.tt0)) * self.KIN / par_obj.unit
        return func()

    def delta_a1_proper_motion(self):
        """The correction on a1 (projected semi-major axis)
        due to the pulsar proper motion
        (Kopeikin 1996 Eq 8):

        .. math::

            \delta_x = a_1 \cot(kin)  (-\mu_{long}\sin(KOM) + \mu_{lat}\cos(KOM)) (t-T_0)

            \delta_{kin} = (-\mu_{long} \sin(KOM) + \mu_{lat} \cos(KOM)) (t-T_0)

            \delta_x = a_1 \delta_{kin}  \cot(kin)

        """
        a1 = self.a1_k(False, False)
        kin = self.kin()
        tan_kin = np.tan(kin)
        d_kin = self.delta_kin_proper_motion()
        d_a1 = a1 * d_kin / tan_kin
        return d_a1.to(a1.unit, equivalencies=u.dimensionless_angles())

    def d_delta_a1_proper_motion_d_KIN(self):
        a1 = self.a1_k(False, False)
        kin = self.kin()
        d_kin = self.delta_kin_proper_motion()
        d_delta_a1_proper_motion_d_KIN = -a1 * d_kin / np.sin(kin) ** 2
        return d_delta_a1_proper_motion_d_KIN.to(
            a1.unit / kin.unit, equivalencies=u.dimensionless_angles()
        )

    def d_delta_a1_proper_motion_d_KOM(self):
        a1 = self.a1_k(False, False)
        kin = self.kin()
        d_kin_d_KOM = self.d_kin_proper_motion_d_KOM()
        tan_kin = np.tan(kin)
        d_kin = self.delta_kin_proper_motion()
        # Since kin = KIN + d_kin
        # dkin/dKOM = dKIN/dKOM + d (d_kin)/dKOM
        # dKIN/dKOM == 0
        # dkin/dKOM = d (d_kin)/dKOM
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        d_delta_a1_proper_motion_d_KOM = (
            a1 * d_kin_d_KOM * (-1.0 / np.sin(kin) ** 2 * d_kin + 1.0 / tan_kin)
        )
        return d_delta_a1_proper_motion_d_KOM.to(a1.unit / self.KOM.unit)

    def d_delta_a1_proper_motion_d_T0(self):
        a1 = self.a1_k(False, False)
        kin = self.kin()
        d_kin_d_T0 = self.d_kin_proper_motion_d_T0()
        tan_kin = np.tan(kin)
        d_kin = self.delta_kin_proper_motion()
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        d_delta_a1_proper_motion_d_T0 = (
            a1 * d_kin_d_T0 * (-1.0 / np.sin(kin) ** 2 * d_kin + 1.0 / tan_kin)
        )
        return d_delta_a1_proper_motion_d_T0.to(a1.unit / self.T0.unit)

    def delta_omega_proper_motion(self):
        """The correction on omega (Longitude of periastron)
        due to the pulsar proper motion
        (Kopeikin 1996 Eq 9):

        .. math::

            \delta_{\Omega} = \csc(i) (\mu_{long}\cos(KOM) + \mu_{lat} \sin(KOM)) (t-T_0)

        """
        kin = self.kin()
        sin_kin = np.sin(kin)
        omega_dot = (
            1.0
            / sin_kin
            * (self.PMLONG_DDK * self.cos_KOM + self.PMLAT_DDK * self.sin_KOM)
        )
        return (omega_dot * self.tt0).to(self.OM.unit)

    def d_delta_omega_proper_motion_d_KIN(self):
        kin = self.kin()
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        d_omega_dot = (
            -cos_kin
            / sin_kin**2
            * (self.PMLONG_DDK * self.cos_KOM + self.PMLAT_DDK * self.sin_KOM)
        )
        return (d_omega_dot * self.tt0).to(
            self.OM.unit / self.KIN.unit, equivalencies=u.dimensionless_angles()
        )

    def d_delta_omega_proper_motion_d_KOM(self):
        kin = self.kin()
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        d_kin_d_KOM = self.d_kin_proper_motion_d_KOM()
        d_omega_dot = (
            -cos_kin
            / sin_kin**2
            * d_kin_d_KOM
            * (self.PMLONG_DDK * self.cos_KOM + self.PMLAT_DDK * self.sin_KOM)
            + (-self.PMLONG_DDK * self.sin_KOM + self.PMLAT_DDK * self.cos_KOM)
            / sin_kin
        )
        return (d_omega_dot * self.tt0).to(
            self.OM.unit / self.KOM.unit, equivalencies=u.dimensionless_angles()
        )

    def d_delta_omega_proper_motion_d_T0(self):
        kin = self.kin()
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        d_kin_d_T0 = self.d_kin_proper_motion_d_T0()
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        d_omega_d_T0 = (
            -cos_kin / sin_kin**2 * d_kin_d_T0 * self.tt0 - 1.0 / sin_kin
        ) * (self.PMLONG_DDK * self.cos_KOM + self.PMLAT_DDK * self.sin_KOM)
        return d_omega_d_T0.to(self.OM.unit / self.T0.unit)

    # The code below is to implement the binary model parameter correction due
    # to the parallax.
    # Reference KOPEIKIN. 1995 Eq 18 -> Eq 19.

    def delta_I0(self):
        """
        :math:`\Delta_{I0}`

        Reference: (Kopeikin 1995 Eq 15)
        """
        return -self.obs_pos[:, 0] * self.sin_long + self.obs_pos[:, 1] * self.cos_long

    def delta_J0(self):
        """
        :math:`\Delta_{J0}`

        Reference: (Kopeikin 1995 Eq 16)
        """
        return (
            -self.obs_pos[:, 0] * self.sin_lat * self.cos_long
            - self.obs_pos[:, 1] * self.sin_lat * self.sin_long
            + self.obs_pos[:, 2] * self.cos_lat
        )

    def delta_sini_parallax(self):
        """Reference (Kopeikin 1995 Eq 18).  Computes:

        .. math::

            x_{obs} = \\frac{a_p  \sin(i)_{obs}}{c}

        Since :math:`a_p` and :math:`c` will not be changed by parallax:

        .. math::

            x_{obs} = \\frac{a_p}{c}(\sin(i)_{\\rm intrisic} + \delta_{\sin(i)})

            \delta_{\sin(i)} = \sin(i)_{\\rm intrisic}  \\frac{\cot(i)_{\\rm intrisic}}{d} (\Delta_{I0} \sin KOM - \Delta_{J0}  \cos KOM)

        """
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        delta_sini = (
            np.cos(self.KIN)
            / PX_kpc
            * (self.delta_I0() * self.sin_KOM - self.delta_J0() * self.cos_KOM)
        )
        return delta_sini.to("")

    def delta_a1_parallax(self):
        """
        Reference: (Kopeikin 1995 Eq 18)
        """
        p_motion = bool(self.K96)
        a1 = self.a1_k(proper_motion=p_motion, parallax=False)
        kin = self.kin()
        tan_kin = np.tan(kin)
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        delta_a1 = (
            a1
            / tan_kin
            / PX_kpc
            * (self.delta_I0() * self.sin_KOM - self.delta_J0() * self.cos_KOM)
        )
        return delta_a1.to(a1.unit)

    def d_delta_a1_parallax_d_KIN(self):
        p_motion = bool(self.K96)
        a1 = self.a1_k(proper_motion=p_motion, parallax=False)
        d_a1_d_kin = self.d_a1_k_d_par("KIN", proper_motion=p_motion, parallax=False)
        kin = self.kin()
        tan_kin = np.tan(kin)
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        d_delta_a1_d_KIN = (
            d_a1_d_kin / tan_kin / PX_kpc - a1 / PX_kpc / np.sin(kin) ** 2
        ) * (self.delta_I0() * self.sin_KOM - self.delta_J0() * self.cos_KOM)
        return d_delta_a1_d_KIN.to(a1.unit / kin.unit)

    def d_delta_a1_parallax_d_KOM(self):
        p_motion = bool(self.K96)
        a1 = self.a1_k(proper_motion=p_motion, parallax=False)
        d_a1_d_kom = self.d_a1_k_d_par("KOM", proper_motion=p_motion, parallax=False)
        kin = self.kin()
        tan_kin = np.tan(kin)
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        d_kin_d_kom = self.d_kin_d_par("KOM")
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        kom_projection = self.delta_I0() * self.sin_KOM - self.delta_J0() * self.cos_KOM
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        d_delta_a1_d_KOM = (
            d_a1_d_kom / tan_kin / PX_kpc * kom_projection
            - a1 * d_kin_d_kom / PX_kpc / sin_kin**2 * kom_projection
            + a1
            / tan_kin
            / PX_kpc
            * (self.delta_I0() * self.cos_KOM + self.delta_J0() * self.sin_KOM)
        )
        return d_delta_a1_d_KOM.to(a1.unit / self.KOM.unit)

    def d_delta_a1_parallax_d_T0(self):
        p_motion = bool(self.K96)
        a1 = self.a1_k(proper_motion=p_motion, parallax=False)
        d_a1_d_T0 = self.d_a1_k_d_par("T0", proper_motion=p_motion, parallax=False)
        kin = self.kin()
        tan_kin = np.tan(kin)
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        d_kin_d_T0 = self.d_kin_d_par("T0")
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        kom_projection = self.delta_I0() * self.sin_KOM - self.delta_J0() * self.cos_KOM
        # with u.set_enabled_equivalencies(u.dimensionless_angles()):
        d_delta_a1_d_T0 = (
            d_a1_d_T0 / tan_kin / PX_kpc - a1 * d_kin_d_T0 / PX_kpc / sin_kin**2
        ) * kom_projection
        return d_delta_a1_d_T0.to(a1.unit / self.T0.unit)

    def delta_omega_parallax(self):
        """
        Reference: (Kopeikin 1995 Eq 19)
        """
        kin = self.kin()
        sin_kin = np.sin(kin)
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        delta_omega = (
            -1.0
            / sin_kin
            / PX_kpc
            * (self.delta_I0() * self.cos_KOM + self.delta_J0() * self.sin_KOM)
        )
        return delta_omega.to(self.OM.unit, equivalencies=u.dimensionless_angles())

    def d_delta_omega_parallax_d_KIN(self):
        kin = self.kin()
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        kom_projection = self.delta_I0() * self.cos_KOM + self.delta_J0() * self.sin_KOM
        d_delta_omega_d_KIN = cos_kin / sin_kin**2 / PX_kpc * kom_projection
        return d_delta_omega_d_KIN.to(
            self.OM.unit / kin.unit, equivalencies=u.dimensionless_angles()
        )

    def d_delta_omega_parallax_d_KOM(self):
        kin = self.kin()
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        kom_projection = self.delta_I0() * self.cos_KOM + self.delta_J0() * self.sin_KOM
        d_kin_d_KOM = self.d_kin_d_par("KOM")
        d_delta_omega_d_KOM = (
            cos_kin / sin_kin**2 / PX_kpc * d_kin_d_KOM * kom_projection
            - 1.0
            / sin_kin
            / PX_kpc
            * (-self.delta_I0() * self.sin_KOM + self.delta_J0() * self.cos_KOM)
        )
        return d_delta_omega_d_KOM.to(
            self.OM.unit / self.KOM.unit, equivalencies=u.dimensionless_angles()
        )

    def d_delta_omega_parallax_d_T0(self):
        kin = self.kin()
        sin_kin = np.sin(kin)
        cos_kin = np.cos(kin)
        PX_kpc = self.PX.to(u.kpc, equivalencies=u.parallax())
        kom_projection = self.delta_I0() * self.cos_KOM + self.delta_J0() * self.sin_KOM
        d_kin_d_T0 = self.d_kin_d_par("T0")
        d_delta_omega_d_T0 = (
            cos_kin / sin_kin**2 / PX_kpc * d_kin_d_T0 * kom_projection
        )
        return d_delta_omega_d_T0.to(
            self.OM.unit / self.T0.unit, equivalencies=u.dimensionless_angles()
        )

    def a1_k(self, proper_motion=True, parallax=True):
        """Compute Kopeikin corrected projected semi-major axis.

        Parameters
        ----------
        proper_motion: boolean, optional, default True
            Flag for proper_motion correction
        parallax: boolean, optional, default True
            Flag for parallax correction
        """
        a1 = super().a1()
        corr_funs = [self.delta_a1_proper_motion, self.delta_a1_parallax]
        mask = [proper_motion, parallax]
        for ii, cf in enumerate(corr_funs):
            if mask[ii]:
                a1 += cf()
        return a1

    def a1(self):
        return self.a1_k() if self.K96 else self.a1_k(proper_motion=False)

    def d_a1_k_d_par(self, par, proper_motion=True, parallax=True):
        result = super().d_a1_d_par(par)
        ko_func_name = ["d_delta_a1_proper_motion_d_", "d_delta_a1_parallax_d_"]
        for ii, flag in enumerate([proper_motion, parallax]):
            if flag:
                try:
                    ko_func = getattr(self, ko_func_name[ii] + par)
                except Exception:
                    ko_func = lambda: np.zeros(len(self.tt0)) * result.unit
                result += ko_func()
        return result

    def d_a1_d_par(self, par):
        if self.K96:
            return self.d_a1_k_d_par(par)
        else:
            return self.d_a1_k_d_par(par, proper_motion=False)

    def omega_k(self, proper_motion=True, parallax=True):
        """A function to compute Kopeikin corrected projected omega.

        Parameters
        ----------
        proper_motion: boolean, optional, default True
            Flag for proper_motion correction
        parallax: boolean, optional, default True
            Flag for parallax correction
        """
        omega = super().omega()
        corr_funs = [self.delta_omega_proper_motion, self.delta_omega_parallax]
        mask = [proper_motion, parallax]
        for ii, cf in enumerate(corr_funs):
            if mask[ii]:
                omega += cf()
        return omega

    def omega(self):
        return self.omega_k() if self.K96 else self.omega_k(proper_motion=False)

    def d_omega_k_d_par(self, par, proper_motion=True, parallax=True):
        result = super().d_omega_d_par(par)
        ko_func_name = ["d_delta_omega_proper_motion_d_", "d_delta_omega_parallax_d_"]
        for ii, flag in enumerate([proper_motion, parallax]):
            if flag:
                try:
                    ko_func = getattr(self, ko_func_name[ii] + par)
                except:
                    ko_func = lambda: np.zeros(len(self.tt0)) * result.unit
                result += ko_func()
        return result

    def d_omega_d_par(self, par):
        if self.K96:
            return self.d_omega_k_d_par(par)
        else:
            return self.d_omega_k_d_par(par, proper_motion=False)
