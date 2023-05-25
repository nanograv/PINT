"""The ELL1 model for approximately handling near-circular orbits."""
import astropy.constants as c
import astropy.units as u
import numpy as np

from .binary_generic import PSR_BINARY


class ELL1BaseModel(PSR_BINARY):
    """This is a class for base ELL1 pulsar binary model.

    ELL1 model is BT model in the small eccentricity case.
    The shapiro delay is computed differently by different subclass of
    ELL1Base.
    """

    def __init__(self):
        super().__init__()
        self.binary_name = "ELL1Base"
        self.param_default_value.update(
            {
                "EPS1": 0 * u.Unit(""),
                "EPS2": 0 * u.Unit(""),
                "EPS1DOT": 0 / u.second,
                "EPS2DOT": 0 / u.second,
                "TASC": np.longdouble(54000.0) * u.day,
            }
        )
        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values()  # Set parameters to default values.
        self.ELL1_interVars = ["eps1", "eps2", "Phi", "Dre", "Drep", "Drepp", "nhat"]
        self.add_inter_vars(self.ELL1_interVars)
        self.orbits_func = self.orbits_ELL1

    @property
    def tt0(self):
        return self.ttasc()

    ###############################

    def ttasc(self):
        """
        ttasc = t - TASC
        """
        t = self.t
        if not hasattr(self.t, "unit") or self.t.unit is None:
            t = self.t * u.day
        return (t - self.TASC.value * u.day).to("second")

    def a1(self):
        """ELL1 model a1 calculation.

        This method overrides the a1() method in pulsar_binary.py. Instead of tt0,
        it uses ttasc.
        """
        return self.A1 + self.ttasc() * self.A1DOT

    def d_a1_d_A1(self):
        return np.longdouble(np.ones(len(self.ttasc()))) * u.Unit("")

    def d_a1_d_T0(self):
        result = np.empty(len(self.ttasc()))
        result.fill(-self.A1DOT.value)
        return result * u.Unit(self.A1DOT.unit)

    def d_a1_d_A1DOT(self):
        return self.ttasc()

    def eps1(self):
        return self.EPS1 + self.ttasc() * self.EPS1DOT

    def d_eps1_d_EPS1(self):
        return np.longdouble(np.ones(len(self.t))) * u.Unit("")

    def d_eps1_d_TASC(self):
        result = np.empty(len(self.t))
        result.fill(-self.EPS1DOT.value)
        return result * u.Unit(self.EPS1DOT.unit)

    def d_eps1_d_EPS1DOT(self):
        return self.ttasc()

    def eps2(self):
        return self.EPS2 + self.ttasc() * self.EPS2DOT

    def d_eps2_d_EPS2(self):
        return np.longdouble(np.ones(len(self.t))) * u.Unit("")

    def d_eps2_d_TASC(self):
        result = np.empty(len(self.t))
        result.fill(-self.EPS2DOT.value)
        return result * u.Unit(self.EPS2DOT.unit)

    def d_eps2_d_EPS2DOT(self):
        return self.ttasc()

    # NOTE in ELL1 the orbit phase is modeled as Phi.
    # But pulsar_binary function M() is a generic function ot computes the
    # orbit phase in the range [0,1], So Phi can be computed by M(). But
    # the attribute .orbits_func needs to be set as orbits_ELL1
    def Phi(self):
        """Orbit phase in ELL1 model. Using TASC"""
        return self.M()

    def orbits_ELL1(self):
        PB = (self.pb()).to("second")
        PBDOT = self.pbdot()
        ttasc = self.ttasc()
        return (ttasc / PB - 0.5 * PBDOT * (ttasc / PB) ** 2).decompose()

    def d_Phi_d_TASC(self):
        """dPhi/dTASC"""
        PB = self.pb().to("second")
        PBDOT = self.pbdot()
        ttasc = self.ttasc()
        return (PBDOT * ttasc / PB - 1.0) * 2 * np.pi * u.rad / PB

    def d_Phi_d_par(self, par):
        """The derivative of Phi with respect to parameter

        Parameters
        ----------
        par : string
              parameter name

        Returns
        -------
        Derivative of Phi respect to par
        """
        if par not in self.binary_params:
            raise ValueError(f"{par} is not in binary parameter list.")

        par_obj = getattr(self, par)
        try:
            func = getattr(self, f"d_Phi_d_{par}")
            return func()
        except Exception:
            return self.d_M_d_par(par)

    def delayI(self):
        """Inverse time delay formula.

        The treatment is similar to the one
        in DD model (T. Damour & N. Deruelle (1986) equation [46-52])::

            Dre = a1*(sin(Phi)+eps1/2*sin(2*Phi)+eps1/2*cos(2*Phi))
            Drep = dDre/dt
            Drepp = d^2 Dre/dt^2
            nhat = dPhi/dt = 2pi/pb
            nhatp = d^2Phi/dt^2 = 0
            Dre(t-Dre(t-Dre(t)))  = Dre(Phi) - Drep(Phi)*nhat*Dre(t-Dre(t))
                                  = Dre(Phi) - Drep(Phi)*nhat*(Dre(Phi)-Drep(Phi)*nhat*Dre(t))
                                    + 1/2 (Drepp(u)*nhat^2 + Drep(u) * nhat * nhatp) * (Dre(t)-...)^2
                                  = Dre(Phi)(1 - nhat*Drep(Phi) + (nhat*Drep(Phi))^2
                                    + 1/2*nhat^2* Dre*Drepp)
        """
        Dre = self.delayR()
        Drep = self.Drep()
        Drepp = self.Drepp()
        PB = self.pb().to("second")
        nhat = 2 * np.pi / self.pb()
        return (
            Dre
            * (1 - nhat * Drep + (nhat * Drep) ** 2 + 1.0 / 2 * nhat**2 * Dre * Drepp)
        ).decompose()

    def nhat(self):
        return 2 * np.pi / self.pb()

    def d_nhat_d_par(self, par):
        return -2 * np.pi / self.pb() ** 2 * self.d_pb_d_par(par)

    def d_delayI_d_par(self, par):
        """Delay derivative.

        Computes::

            delayI = Dre*(1 - nhat*Drep + (nhat*Drep)**2 + 1.0/2*nhat**2*Dre*Drepp)
            d_delayI_d_par = d_delayI_d_Dre * d_Dre_d_par + d_delayI_d_Drep * d_Drep_d_par +
                             d_delayI_d_Drepp * d_Drepp_d_par + d_delayI_d_nhat * d_nhat_d_par
        """
        Dre = self.delayR()
        Drep = self.Drep()
        Drepp = self.Drepp()
        PB = self.pb().to("second")
        nhat = 2 * np.pi / self.pb()

        d_delayI_d_Dre = (
            1 - nhat * Drep + (nhat * Drep) ** 2 + 1.0 / 2 * nhat**2 * Dre * Drepp
        ) + Dre * 1.0 / 2 * nhat**2 * Drepp
        d_delayI_d_Drep = -Dre * nhat + 2 * (nhat * Drep) * nhat * Dre
        d_delayI_d_Drepp = 1.0 / 2 * (nhat * Dre) ** 2
        d_delayI_d_nhat = Dre * (-Drep + 2 * (nhat * Drep) * Drep + nhat * Dre * Drepp)
        d_nhat_d_par = self.prtl_der("nhat", par)
        d_Dre_d_par = self.d_Dre_d_par(par)
        d_Drep_d_par = self.d_Drep_d_par(par)
        d_Drepp_d_par = self.d_Drepp_d_par(par)

        return (
            d_delayI_d_Dre * d_Dre_d_par
            + d_delayI_d_Drep * d_Drep_d_par
            + d_delayI_d_Drepp * d_Drepp_d_par
            + d_delayI_d_nhat * d_nhat_d_par
        )

    def ELL1_om(self):
        # arctan(om)
        om = np.arctan2(self.eps1(), self.eps2())
        return om.to(u.deg, equivalencies=u.dimensionless_angles())

    def ELL1_ecc(self):
        return np.sqrt(self.eps1() ** 2 + self.eps2() ** 2)

    def ELL1_T0(self):
        return self.TASC + self.pb() / (2 * np.pi) * (
            np.arctan(self.eps1() / self.eps2())
        ).to(u.Unit(""), equivalencies=u.dimensionless_angles())

    ###############################
    def d_delayR_da1(self):
        """ELL1 Roemer delay in proper time divided by a1/c, including third order corrections

        typo corrected from Zhu et al., following:
        https://github.com/nanograv/tempo/blob/master/src/bnryell1.f
        """
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        return (
            np.sin(Phi)
            + 0.5 * (eps2 * np.sin(2 * Phi) - eps1 * np.cos(2 * Phi))
            - (1.0 / 8)
            * (
                5 * eps2**2 * np.sin(Phi)
                - 3 * eps2**2 * np.sin(3 * Phi)
                - 2 * eps2 * eps1 * np.cos(Phi)
                + 6 * eps2 * eps1 * np.cos(3 * Phi)
                + 3 * eps1**2 * np.sin(Phi)
                + 3 * eps1**2 * np.sin(3 * Phi)
            )
            - (1.0 / 12)
            * (
                5 * eps2**3 * np.sin(2 * Phi)
                + 3 * eps1**2 * eps2 * np.sin(2 * Phi)
                - 6 * eps1 * eps2**2 * np.cos(2 * Phi)
                - 4 * eps1**3 * np.cos(2 * Phi)
                - 4 * eps2**3 * np.sin(4 * Phi)
                + 12 * eps1**2 * eps2 * np.sin(4 * Phi)
                + 12 * eps1 * eps2**2 * np.cos(4 * Phi)
                - 4 * eps1**3 * np.cos(4 * Phi)
            )
        )

    def d_d_delayR_dPhi_da1(self):
        """d (ELL1 Roemer delay)/dPhi in proper time divided by a1/c"""
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        return (
            np.cos(Phi)
            + eps1 * np.sin(2 * Phi)
            + eps2 * np.cos(2 * Phi)
            - (1.0 / 8)
            * (
                5 * eps2**2 * np.cos(Phi)
                - 9 * eps2**2 * np.cos(3 * Phi)
                + 2 * eps1 * eps2 * np.sin(Phi)
                - 18 * eps1 * eps2 * np.sin(3 * Phi)
                + 3 * eps1**2 * np.cos(Phi)
                + 9 * eps1**2 * np.cos(3 * Phi)
            )
            - (1.0 / 12)
            * (
                10 * eps2**3 * np.cos(2 * Phi)
                + 6 * eps1**2 * eps2 * np.cos(2 * Phi)
                + 12 * eps1 * eps2**2 * np.sin(2 * Phi)
                + 8 * eps1**3 * np.sin(2 * Phi)
                - 16 * eps2**3 * np.cos(4 * Phi)
                + 48 * eps1**2 * eps2 * np.cos(4 * Phi)
                - 48 * eps1 * eps2**2 * np.sin(4 * Phi)
                + 16 * eps1**3 * np.sin(4 * Phi)
            )
        )

    def d_dd_delayR_dPhi_da1(self):
        """d^2 (ELL1 Roemer delay)/dPhi^2 in proper time divided by a1/c"""
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        return (
            -np.sin(Phi)
            + 2 * eps1 * np.cos(2 * Phi)
            - 2 * eps2 * np.sin(2 * Phi)
            - (1.0 / 8)
            * (
                -5 * eps2**2 * np.sin(Phi)
                + 27 * eps2**2 * np.sin(3 * Phi)
                + 2 * eps1 * eps2 * np.cos(Phi)
                - 54 * eps1 * eps2 * np.cos(3 * Phi)
                - 3 * eps1**2 * np.sin(Phi)
                - 27 * eps1**2 * np.sin(3 * Phi)
            )
            - (1.0 / 12)
            * (
                -20 * eps2**3 * np.sin(2 * Phi)
                - 12 * eps1**2 * eps2 * np.sin(2 * Phi)
                + 24 * eps1 * eps2**2 * np.cos(2 * Phi)
                + 16 * eps1**3 * np.cos(2 * Phi)
                + 64 * eps2**3 * np.sin(4 * Phi)
                - 192 * eps1**2 * eps2 * np.sin(4 * Phi)
                - 192 * eps1 * eps2**2 * np.cos(4 * Phi)
                + 64 * eps1**3 * np.cos(4 * Phi)
            )
        )

    def delayR(self):
        """ELL1 Roemer delay in proper time.
        Include terms up to third order in eccentricity
        Zhu et al. (2019), Eqn. 1
        Fiore et al. (2023), Eqn. 4
        """
        return ((self.a1() / c.c) * self.d_delayR_da1()).decompose()

    def d_Dre_d_par(self, par):
        """Derivative computation.

        Computes::

            Dre = delayR = a1/c.c*(sin(phi) - 0.5* eps1*cos(2*phi) +  0.5* eps2*sin(2*phi) + ...)
            d_Dre_d_par = d_a1_d_par /c.c*(sin(phi) - 0.5* eps1*cos(2*phi) +  0.5* eps2*sin(2*phi)) +
                          d_Dre_d_Phi * d_Phi_d_par + d_Dre_d_eps1*d_eps1_d_par + d_Dre_d_eps2*d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der("a1", par)
        d_Dre_d_Phi = self.Drep()
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Dre_d_eps1 = (
            a1
            / c.c
            * (
                -0.5 * np.cos(2 * Phi)
                - (1.0 / 8)
                * (
                    -2 * eps2 * np.cos(Phi)
                    + 6 * eps2 * np.cos(3 * Phi)
                    + 6 * eps1 * np.sin(Phi)
                    + 6 * eps1 * np.sin(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    6 * eps1 * eps2 * np.sin(2 * Phi)
                    - 6 * eps2**2 * np.cos(2 * Phi)
                    - 12 * eps1**2 * np.cos(2 * Phi)
                    + 24 * eps1 * eps2 * np.sin(4 * Phi)
                    + 12 * eps2**2 * np.cos(4 * Phi)
                    - 12 * eps1**2 * np.cos(4 * Phi)
                )
            )
        )

        d_Dre_d_eps2 = (
            a1
            / c.c
            * (
                0.5 * np.sin(2 * Phi)
                - (1.0 / 8)
                * (
                    -2 * eps1 * np.cos(Phi)
                    + 6 * eps1 * np.cos(3 * Phi)
                    + 10 * eps2 * np.sin(Phi)
                    - 6 * eps2 * np.sin(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    15 * eps2**2 * np.sin(2 * Phi)
                    + 3 * eps1**2 * np.sin(2 * Phi)
                    - 12 * eps1 * eps2 * np.cos(2 * Phi)
                    - 12 * eps2**2 * np.sin(4 * Phi)
                    + 12 * eps1**2 * np.sin(4 * Phi)
                    + 24 * eps1 * eps2 * np.cos(4 * Phi)
                )
            )
        )

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Dre_d_par = (
                d_a1_d_par / c.c * self.d_delayR_da1()
                + d_Dre_d_Phi * d_Phi_d_par
                + d_Dre_d_eps1 * self.prtl_der("eps1", par)
                + d_Dre_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Dre_d_par

    def Drep(self):
        """dDre/dPhi"""
        a1 = self.a1()
        # Here we are using full d Dre/dPhi. But Tempo and Tempo2 ELL1 model
        # does not have terms beyond the first one. This will result a difference in
        # the order of magnitude of 1e-8s level.
        return a1 / c.c * self.d_d_delayR_dPhi_da1()

    def d_Drep_d_par(self, par):
        """Derivative computation.

        Computes::

            Drep = d_Dre_d_Phi = a1/c.c*(cos(Phi) + eps1 * sin(Phi) + eps2 * cos(Phi) + ...)
            d_Drep_d_par = ...
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der("a1", par)
        d_Drep_d_Phi = self.Drepp()
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Drep_d_eps1 = (
            a1
            / c.c
            * (
                np.sin(2.0 * Phi)
                - (1.0 / 8)
                * (
                    6 * eps1 * np.cos(Phi)
                    + 18 * eps1 * np.cos(3 * Phi)
                    + 2 * eps2 * np.sin(Phi)
                    - 18 * eps2 * np.sin(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    12 * eps1 * eps2 * np.cos(2 * Phi)
                    + 12 * eps2**2 * np.sin(2 * Phi)
                    + 16 * eps1**2 * np.sin(2 * Phi)
                    + 96 * eps1 * eps2 * np.cos(4 * Phi)
                    - 48 * eps2**2 * np.sin(4 * Phi)
                    + 48 * eps1**2 * np.sin(4 * Phi)
                )
            )
        )

        d_Drep_d_eps2 = (
            a1
            / c.c
            * (
                np.cos(2.0 * Phi)
                - (1.0 / 8)
                * (
                    2 * eps1 * np.sin(Phi)
                    - 18 * eps1 * np.sin(3 * Phi)
                    + 10 * eps2 * np.cos(Phi)
                    - 18 * eps2 * np.cos(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    30 * eps2**2 * np.cos(2 * Phi)
                    + 6 * eps1**2 * np.cos(2 * Phi)
                    + 24 * eps1 * eps2 * np.sin(2 * Phi)
                    - 48 * eps2**2 * np.cos(4 * Phi)
                    + 48 * eps1**2 * np.cos(4 * Phi)
                    - 96 * eps1 * eps2 * np.sin(4 * Phi)
                )
            )
        )

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Drep_d_par = (
                d_a1_d_par / c.c * self.d_d_delayR_dPhi_da1()
                + d_Drep_d_Phi * d_Phi_d_par
                + d_Drep_d_eps1 * self.prtl_der("eps1", par)
                + d_Drep_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Drep_d_par

    def Drepp(self):
        """d^2Dre/dPhi^2"""
        a1 = self.a1()
        return a1 / c.c * self.d_dd_delayR_dPhi_da1()

    def d_Drepp_d_par(self, par):
        """Derivative computation

        Computes::

            Drepp = d_Drep_d_Phi = ...
            d_Drepp_d_par = ...
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der("a1", par)
        d_Drepp_d_Phi = (
            a1
            / c.c
            * (
                -np.cos(Phi)
                - 4.0 * (eps1 * np.sin(2.0 * Phi) + eps2 * np.cos(2.0 * Phi))
                - (1.0 / 8)
                * (
                    -5 * eps2**2 * np.cos(Phi)
                    + 81 * eps2**2 * np.cos(3 * Phi)
                    - 2 * eps1 * eps2 * np.sin(Phi)
                    + 162 * eps1 * eps2 * np.sin(3 * Phi)
                    - 3 * eps1**2 * np.cos(Phi)
                    - 81 * eps1**2 * np.cos(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    -40 * eps2**3 * np.cos(2 * Phi)
                    - 24 * eps1**2 * eps2 * np.cos(2 * Phi)
                    - 48 * eps1 * eps2**2 * np.sin(2 * Phi)
                    - 32 * eps1**3 * np.sin(2 * Phi)
                    + 256 * eps2**3 * np.cos(4 * Phi)
                    - 768 * eps1**2 * eps2 * np.cos(4 * Phi)
                    + 768 * eps1 * eps2**2 * np.sin(4 * Phi)
                    - 256 * eps1**3 * np.sin(4 * Phi)
                )
            )
        )

        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Drepp_d_eps1 = (
            a1
            / c.c
            * (
                2.0 * np.cos(2.0 * Phi)
                - (1.0 / 8)
                * (
                    -6 * eps1 * np.sin(Phi)
                    - 54 * eps1 * np.sin(3 * Phi)
                    + 2 * eps2 * np.cos(Phi)
                    - 54 * eps2 * np.cos(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    -24 * eps1 * eps2 * np.sin(2 * Phi)
                    + 24 * eps2**2 * np.cos(2 * Phi)
                    + 48 * eps1**2 * np.cos(2 * Phi)
                    - 384 * eps1 * eps2 * np.sin(4 * Phi)
                    - 192 * eps2**2 * np.cos(4 * Phi)
                    + 192 * eps1**2 * np.cos(4 * Phi)
                )
            )
        )
        d_Drepp_d_eps2 = (
            a1
            / c.c
            * (
                -2.0 * np.sin(2.0 * Phi)
                - (1.0 / 8)
                * (
                    2 * eps1 * np.cos(Phi)
                    - 54 * eps1 * np.cos(3 * Phi)
                    - 10 * eps2 * np.sin(Phi)
                    + 54 * eps2 * np.sin(3 * Phi)
                )
                - (1.0 / 12)
                * (
                    -60 * eps2**2 * np.sin(2 * Phi)
                    - 12 * eps1**2 * np.sin(2 * Phi)
                    + 48 * eps1 * eps2 * np.cos(2 * Phi)
                    + 192 * eps2**2 * np.sin(4 * Phi)
                    - 192 * eps1**2 * np.sin(4 * Phi)
                    - 384 * eps1 * eps2 * np.cos(4 * Phi)
                )
            )
        )

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Drepp_d_par = (
                d_a1_d_par / c.c * self.d_dd_delayR_dPhi_da1()
                + d_Drepp_d_Phi * d_Phi_d_par
                + d_Drepp_d_eps1 * self.prtl_der("eps1", par)
                + d_Drepp_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Drepp_d_par


class ELL1model(ELL1BaseModel):
    """This is a ELL1 model using M2 and SINI as the Shapiro delay parameters.

    References
    ----------
    - Lange et al. (2001), MNRAS, 326 (1), 274–282 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..274L/abstract
    """

    def __init__(self):
        super().__init__()
        self.binary_name = "ELL1"
        self.binary_delay_funcs = [self.ELL1delay]
        self.d_binarydelay_d_par_funcs = [self.d_ELL1delay_d_par]

    def delayS(self):
        """ELL1 Shapiro delay. Lange et al 2001 eq. A16"""
        TM2 = self.TM2()
        Phi = self.Phi()
        return -2 * TM2 * np.log(1 - self.SINI * np.sin(Phi))

    def d_delayS_d_par(self, par):
        """Derivative for binary Shapiro delay.

        Computes::

            delayS = -2 * TM2 * np.log(1 - self.SINI * np.sin(Phi))
            d_delayS_d_par = d_delayS_d_TM2 * d_TM2_d_par + d_delayS_d_SINI*d_SINI_d_par +
                             d_delayS_d_Phi * d_Phi_d_par
        """
        TM2 = self.TM2()
        Phi = self.Phi()
        d_delayS_d_TM2 = -2 * np.log(1 - self.SINI * np.sin(Phi))
        d_delayS_d_SINI = (
            -2 * TM2 * 1.0 / (1 - self.SINI * np.sin(Phi)) * (-np.sin(Phi))
        )
        d_delayS_d_Phi = -2 * TM2 * 1.0 / (1 - self.SINI * np.sin(Phi)) * (-self.SINI)
        d_TM2_d_par = self.prtl_der("TM2", par)
        d_SINI_d_par = self.prtl_der("SINI", par)
        d_Phi_d_par = self.prtl_der("Phi", par)

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_delayS_d_par = (
                d_delayS_d_TM2 * d_TM2_d_par
                + d_delayS_d_SINI * d_SINI_d_par
                + d_delayS_d_Phi * d_Phi_d_par
            )
        return d_delayS_d_par

    def ELL1delay(self):
        # TODO need add aberration delay
        return self.delayI() + self.delayS()

    def d_ELL1delay_d_par(self, par):
        return self.d_delayI_d_par(par) + self.d_delayS_d_par(par)
