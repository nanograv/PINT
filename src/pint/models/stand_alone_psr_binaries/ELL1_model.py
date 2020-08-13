"""The ELL1 model for approximately handling near-circular orbits."""
from __future__ import absolute_import, division, print_function

import astropy.constants as c
import astropy.units as u
import numpy as np

from pint import GMsun, Tsun, ls

from .binary_generic import PSR_BINARY


class ELL1BaseModel(PSR_BINARY):
    """This is a class for base ELL1 pulsar binary model.

    ELL1 model is BT model in the small eccentricity case.
    The shapiro delay is computed differently by different subclass of
    ELL1Base.
    """

    def __init__(self):
        super(ELL1BaseModel, self).__init__()
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
        if not hasattr(self.t, "unit") or self.t.unit == None:
            t = self.t * u.day
        t = self.t
        ttasc = (t - self.TASC).to("second")
        return ttasc

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
        """Orbit phase in ELL1 model. Using TASC
        """
        phase = self.M()
        return phase

    def orbits_ELL1(self):
        PB = (self.pb()).to("second")
        PBDOT = self.pbdot()
        ttasc = self.ttasc()
        orbits = (ttasc / PB - 0.5 * PBDOT * (ttasc / PB) ** 2).decompose()
        return orbits

    def d_Phi_d_TASC(self):
        """dPhi/dTASC
        """
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
         Derivitve of Phi respect to par
        """
        if par not in self.binary_params:
            errorMesg = par + " is not in binary parameter list."
            raise ValueError(errorMesg)

        par_obj = getattr(self, par)
        try:
            func = getattr(self, "d_Phi_d_" + par)
            return func()
        except:
            return self.d_M_d_par(par)

    def d_Dre_d_par(self, par):
        """Derivative computation.

        Computes::

            Dre = delayR = a1/c.c*(sin(phi) - 0.5* eps1*cos(2*phi) +  0.5* eps2*sin(2*phi))
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
        d_Dre_d_eps1 = a1 / c.c * (-0.5 * np.cos(2 * Phi))
        d_Dre_d_eps2 = a1 / c.c * (0.5 * np.sin(2 * Phi))

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Dre_d_par = (
                d_a1_d_par
                / c.c
                * (
                    np.sin(Phi)
                    - 0.5 * eps1 * np.cos(2 * Phi)
                    + 0.5 * eps2 * np.sin(2 * Phi)
                )
                + d_Dre_d_Phi * d_Phi_d_par
                + d_Dre_d_eps1 * self.prtl_der("eps1", par)
                + d_Dre_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Dre_d_par

    def Drep(self):
        """ dDre/dPhi
        """
        a1 = self.a1()
        eps1 = self.eps1()
        eps2 = self.eps2()
        Phi = self.Phi()
        # Here we are using full d Dre/dPhi. But Tempo and Tempo2 ELL1 model
        # does not have the last two terms. This will result a difference in
        # the order of magnitude of 1e-8s level.
        return (
            a1
            / c.c
            * (np.cos(Phi) + eps1 * np.sin(2.0 * Phi) + eps2 * np.cos(2.0 * Phi))
        )

    def d_Drep_d_par(self, par):
        """Derivative computation.

        Computes::

            Drep = d_Dre_d_Phi = a1/c.c*(cos(Phi) + eps1 * sin(Phi) + eps2 * cos(Phi))
            d_Drep_d_par = d_a1_d_par /c.c*(cos(Phi) + eps1 * sin(Phi) + eps2 * cos(Phi)) +
                          d_Drep_d_Phi * d_Phi_d_par + d_Drep_d_eps1*d_eps1_d_par +
                          d_Drep_d_eps2*d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der("a1", par)
        d_Drep_d_Phi = self.Drepp()
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Drep_d_eps1 = a1 / c.c * np.sin(2.0 * Phi)
        d_Drep_d_eps2 = a1 / c.c * np.cos(2.0 * Phi)

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Drep_d_par = (
                d_a1_d_par
                / c.c
                * (np.cos(Phi) + eps1 * np.sin(2.0 * Phi) + eps2 * np.cos(2.0 * Phi))
                + d_Drep_d_Phi * d_Phi_d_par
                + d_Drep_d_eps1 * self.prtl_der("eps1", par)
                + d_Drep_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Drep_d_par

    def Drepp(self):
        a1 = self.a1()
        eps1 = self.eps1()
        eps2 = self.eps2()
        Phi = self.Phi()
        return (
            a1
            / c.c
            * (
                -np.sin(Phi)
                + 2.0 * (eps1 * np.cos(2.0 * Phi) - eps2 * np.sin(2.0 * Phi))
            )
        )

    def d_Drepp_d_par(self, par):
        """Derivative computation

        Computes::

            Drepp = d_Drep_d_Phi = a1/c.c*(-sin(Phi) + 2.0* (eps1 * cos(2.0*Phi) - eps2 * sin(2.0*Phi)))
            d_Drepp_d_par = d_a1_d_par /c.c*(-sin(Phi) + 2.0* (eps1 * cos(2.0*Phi) - eps2 * sin(2.0*Phi))) +
                          d_Drepp_d_Phi * d_Phi_d_par + d_Drepp_d_eps1*d_eps1_d_par +
                          d_Drepp_d_eps2*d_eps2_d_par
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
            )
        )
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Drepp_d_eps1 = a1 / c.c * 2.0 * np.cos(2.0 * Phi)
        d_Drepp_d_eps2 = -a1 / c.c * 2.0 * np.sin(2.0 * Phi)

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Drepp_d_par = (
                d_a1_d_par
                / c.c
                * (
                    -np.sin(Phi)
                    + 2.0 * (eps1 * np.cos(2.0 * Phi) - eps2 * np.sin(2.0 * Phi))
                )
                + d_Drepp_d_Phi * d_Phi_d_par
                + d_Drepp_d_eps1 * self.prtl_der("eps1", par)
                + d_Drepp_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Drepp_d_par

    def delayR(self):
        """ELL1 Roemer delay in proper time. Ch. Lange,1 F. Camilo, 2001 eq. A6 """
        Phi = self.Phi()
        return (
            self.a1()
            / c.c
            * (
                np.sin(Phi)
                + 0.5 * (self.eps2() * np.sin(2 * Phi) - self.eps1() * np.cos(2 * Phi))
            )
        ).decompose()

    def delayI(self):
        """Inverse time delay formular.

        The treatment is similar to the one
        in DD model(T. Damour and N. Deruelle(1986)equation [46-52])::

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
            * (1 - nhat * Drep + (nhat * Drep) ** 2 + 1.0 / 2 * nhat ** 2 * Dre * Drepp)
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
            1 - nhat * Drep + (nhat * Drep) ** 2 + 1.0 / 2 * nhat ** 2 * Dre * Drepp
        ) + Dre * 1.0 / 2 * nhat ** 2 * Drepp
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


class ELL1model(ELL1BaseModel):
    """ This is a ELL1 model using M2 and SINI as the Shapiro delay parameters.
    """

    def __init__(self):
        super(ELL1model, self).__init__()
        self.binary_name = "ELL1"
        self.binary_delay_funcs = [self.ELL1delay]
        self.d_binarydelay_d_par_funcs = [self.d_ELL1delay_d_par]

    def delayS(self):
        """ELL1 Shaprio delay. Ch. Lange,1 F. Camilo, 2001 eq. A16
        """
        TM2 = self.TM2()
        Phi = self.Phi()
        sDelay = -2 * TM2 * np.log(1 - self.SINI * np.sin(Phi))
        return sDelay

    def d_delayS_d_par(self, par):
        """Derivative for bianry Shaprio delay.

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
