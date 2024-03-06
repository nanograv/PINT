"""Damour and Deruelle binary model."""
import astropy.constants as c
import astropy.units as u
import numpy as np
from loguru import logger as log

from pint import Tsun

from .binary_generic import PSR_BINARY


class DDmodel(PSR_BINARY):
    """DD binary model.

    This is a class independent from PINT platform for pulsar DD binary model.
    Reference: T. Damour and N. Deruelle (1986)
    It is a subclass of PSR_BINARY class defined in file binary_generic.py in
    the same directory. This class is designed for PINT platform but can be used
    as an independent module for binary delay calculation.
    To interact with PINT platform, a pulsar_binary wrapper is needed.
    See the source file pint/models/pulsar_binary_dd.py

    Example
    -------

        >>> import numpy as np
        >>> t = np.linspace(54200.0,55000.0,800)
        >>> binary_model = DDmodel()
        >>> parameters_dict = {'A0':0.5,'ECC':0.01}
        >>> binary_model.update_input(t, parameters_dict)

    Here the binary has time input and parameters input, the delay can be
    calculated.
    """

    def __init__(self, t=None, input_params=None):
        super().__init__()
        self.binary_name = "DD"
        # Add parameter that specific for DD model, with default value and units
        self.param_default_value.update(
            {
                "A0": 0 * u.second,
                "B0": 0 * u.second,
                "DR": 0 * u.Unit(""),
                "DTH": 0 * u.Unit(""),
            }
        )
        # If any parameter has aliases, it should be updated
        # self.param_aliases.update({})
        self.binary_params = list(self.param_default_value.keys())

        self.dd_interVars = [
            "er",
            "eTheta",
            "beta",
            "alpha",
            "Dre",
            "Drep",
            "Drepp",
            "nhat",
            "TM2",
        ]
        self.add_inter_vars(self.dd_interVars)
        self.set_param_values()  # Set parameters to default values.
        self.binary_delay_funcs = [self.DDdelay]
        self.d_binarydelay_d_par_funcs = [self.d_DDdelay_d_par]
        if t is not None:
            self.t = t
        if input_params is not None:
            self.update_input(param_dict=input_params)

    # calculations for delays in DD model

    @property
    def k(self):
        # separate this into a property so it can be calculated correctly in DDGR
        # note that this include self.pb() in the calculation of k
        # and self.pb() is PB + PBDOT*dt, so it can vary slightly
        # compared to a definition that does not include PBDOT
        # I am not certain about how this should be done
        # but this is keeping the behavior consistent
        return self.OMDOT.to(u.rad / u.second) / (2 * np.pi * u.rad / self.pb())

    # DDmodel special omega.
    def omega(self):
        """T. Damour and N. Deruelle (1986) equation [25]

        Calculates::

           omega = OM+nu*k
           k = OMDOT/n

        (T. Damour and N. Deruelle (1986) equation between Eq 16 Eq 17)
        """
        return (self.OM + self.nu() * self.k).to(u.rad)

    def d_omega_d_par(self, par):
        """derivative for omega respect to user input Parameter.

        Calculates::

           if par is not 'OM','OMDOT','PB'
           dOmega/dPar =  k*dAe/dPar
           k = OMDOT/n

        Parameters
        ----------
        par : string
             parameter name

        Returns
        -------
        Derivative of omega respect to par
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)
        par_obj = getattr(self, par)

        PB = self.pb()
        OMDOT = self.OMDOT
        OM = self.OM
        nu = self.nu()
        if par in ["OM", "OMDOT"]:
            dername = f"d_omega_d_{par}"
            return getattr(self, dername)()
        elif par in self.orbits_cls.orbit_params:
            d_nu_d_par = self.d_nu_d_par(par)
            d_pb_d_par = self.d_pb_d_par(par)
            return d_nu_d_par * self.k + d_pb_d_par * nu * OMDOT.to(
                u.rad / u.second
            ) / (2 * np.pi * u.rad)
        else:
            # For parameters only in nu
            return (self.k * self.d_nu_d_par(par)).to(
                OM.unit / par_obj.unit, equivalencies=u.dimensionless_angles()
            )

    def d_omega_d_OM(self):
        """dOmega/dOM = 1"""
        return np.ones(len(self.tt0), dtype=np.longdouble) * u.Unit("")

    def d_omega_d_OMDOT(self):
        """Derivative.

        Calculates::

            dOmega/dOMDOT = 1/n*nu
            n = 2*pi/PB
            dOmega/dOMDOT = PB/2*pi*nu
        """
        PB = (self.pb()).to("second")
        nu = self.nu()

        return PB / (2 * np.pi * u.rad) * nu

    ############################################################
    # Calculate er
    def er(self):
        return self.ecc() * (1 + self.DR)

    def d_er_d_DR(self):
        return np.longdouble(np.ones(len(self.tt0))) * self.ecc()

    def d_er_d_par(self, par):
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)

        if par in ["DR"]:
            dername = f"d_er_d_{par}"
            return getattr(self, dername)()
        else:
            dername = f"d_ecc_d_{par}"
            if hasattr(self, dername):
                return getattr(self, dername)()
            par_obj = getattr(self, par)
            return np.zeros(len(self.tt0), dtype=np.longdouble) * (
                u.Unit("") / par_obj.unit
            )

    ##########
    def eTheta(self):
        return self.ecc() * (1 + self.DTH)

    def d_eTheta_d_DTH(self):
        return np.longdouble(np.ones(len(self.tt0))) * self.ecc()

    def d_eTheta_d_par(self, par):
        if par not in self.binary_params:
            errorMesg = f"{par} is not in parameter list."
            raise ValueError(errorMesg)
        par_obj = getattr(self, par)

        if par in ["DTH"]:
            dername = f"d_eTheta_d_{par}"
            return getattr(self, dername)()
        else:
            dername = f"d_ecc_d_{par}"
            if hasattr(self, dername):
                return getattr(self, dername)()
            else:
                return (
                    np.longdouble(np.zeros(len(self.tt0))) * u.Unit("") / par_obj.unit
                )

    ##########
    def alpha(self):
        """Alpha defined in T. Damour and N. Deruelle (1986) equation [46]

        Computes::

            alpha = A1/c*sin(omega)
        """
        sinOmg = np.sin(self.omega())
        return self.a1() / c.c * sinOmg

    def d_alpha_d_par(self, par):
        """T. Damour and N. Deruelle (1986) equation [46]

        Computes::

           alpha = a1/c*sin(omega)
           dAlpha/dpar = d_a1_d_par /c * sin(omega) + a1/c*cos(omega)*dOmega/dPar
        """

        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)
        par_obj = getattr(self, par)
        alpha = self.alpha()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())
        a1 = self.a1()
        d_a1_d_par = self.d_a1_d_par(par)
        d_omega_d_par = self.d_omega_d_par(par)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dAlpha_dpar = d_a1_d_par / c.c * sinOmg + a1 / c.c * cosOmg * d_omega_d_par
        #
        # if par in ['A1','A1DOT']:
        #     dername = 'd_alpha_d_'+par
        #     return getattr(self,dername)()
        #
        # else:
        #     dername = 'd_omega_d_'+par # For parameters only in Ae
        #     if hasattr(self,dername):
        #         cosOmg=np.cos(self.omega())
        #         return self.a1()/c.c*cosOmg*getattr(self,dername)()
        #     else:
        #         return np.longdouble(np.zeros(len(self.tt0)))
        return dAlpha_dpar.to(alpha.unit / par_obj.unit)

    # def d_alpha_d_A1(self):
    #     sinOmg = np.sin(self.omega())
    #     return 1.0/c.c*sinOmg
    #
    # def d_alpha_d_A1DOT(self):
    #     sinOmg = np.sin(self.omega())
    #     return self.tt0/c.c*sinOmg
    ##############################################

    def beta(self):
        """Beta defined in T. Damour and N. Deruelle (1986) equation [47]

        Computes::

           beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        return self.a1() / c.c * (1 - eTheta**2) ** 0.5 * cosOmg

    def d_beta_d_par(self, par):
        """Derivative.

        Computes::

           beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
           eTheta = ecc+Dth  ??
           dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
           dBeta/dECC = A1/c*((-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           dBeta/dEDOT = A1/c*((-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)*de/dEDOT-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           dBeta/dDth = A1/c*(-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)
           Other parameters
           dBeta/dPar = -A1/c*(1-eTheta**2)**0.5*sin(omega)*dOmega/dPar
        """
        if par not in self.binary_params:
            errorMesg = f"{par} is not in binary parameter list."
            raise ValueError(errorMesg)
        par_obj = getattr(self, par)
        beta = self.beta()
        eTheta = self.eTheta()
        a1 = self.a1()
        omega = self.omega()
        sinOmg = np.sin(omega)
        cosOmg = np.cos(omega)
        d_a1_d_par = self.d_a1_d_par(par)
        d_omega_d_par = self.d_omega_d_par(par)
        d_eTheta_d_par = self.d_eTheta_d_par(par)

        d_beta_d_a1 = 1.0 / c.c * (1 - eTheta**2) ** 0.5 * cosOmg
        d_beta_d_omega = -a1 / c.c * (1 - eTheta**2) ** 0.5 * sinOmg
        d_beta_d_eTheta = a1 / c.c * (-eTheta) / np.sqrt(1 - eTheta**2) * cosOmg
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return (
                d_beta_d_a1 * d_a1_d_par
                + d_beta_d_omega * d_omega_d_par
                + d_beta_d_eTheta * d_eTheta_d_par
            ).to(beta.unit / par_obj.unit)

        # if par in ['A1','ECC','EDOT','DTH','A1DOT']:
        #     dername = 'd_beta_d_'+par
        #     return getattr(self,dername)()
        #
        # else:
        #     dername = 'd_omega_d_'+par # For parameters only in omega
        #     if hasattr(self,dername):
        #         eTheta = self.eTheta()
        #         a1 = self.a1()
        #         sinOmg = np.sin(self.omega())
        #         return -a1/c.c*(1-eTheta**2)**0.5*sinOmg*getattr(self,dername)()
        #     else:
        #         return np.longdouble(np.zeros(len(self.tt0)))

    def d_beta_d_A1(self):
        """Derivative.

        Computes::

            dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega) * d_a1_d_A1
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        d_a1_d_A1 = self.d_a1_d_A1()
        return d_a1_d_A1 / c.c * (1 - eTheta**2) ** 0.5 * cosOmg

    def d_beta_d_A1DOT(self):
        """Derivative.

        Computes::

            dBeta/dA1DOT = * d_a1_d_A1DOT/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        d_a1_d_A1DOT = self.d_a1_d_A1DOT()
        return d_a1_d_A1DOT / c.c * (1 - eTheta**2) ** 0.5 * cosOmg

    def d_beta_d_T0(self):
        """Derivative.

        Computes::

            dBeta/dT0 = * d_a1_d_T0/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        d_a1_d_T0 = self.d_a1_d_T0()
        return d_a1_d_T0 / c.c * (1 - eTheta**2) ** 0.5 * cosOmg

    def d_beta_d_ECC(self):
        """Derivative.

        Computes::

           dBeta/dECC = A1/c*((-(e+dtheta)/sqrt(1-(e+dtheta)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           de/dECC = 1
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return (
                a1
                / c.c
                * (
                    (-eTheta) / np.sqrt(1 - eTheta**2) * cosOmg
                    - (1 - eTheta**2) ** 0.5 * sinOmg * self.d_omega_d_par("ECC")
                )
            )

    def d_beta_d_EDOT(self):
        """Derivative.

        Computes::
           dBeta/dEDOT = A1/c*((-(e+dtheta)/sqrt(1-(e+dtheta)**2)*cos(omega)*de/dEDOT- \
           (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           de/dEDOT = tt0
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return (
                a1
                / c.c
                * (
                    (-eTheta) / np.sqrt(1 - eTheta**2) * cosOmg * self.tt0
                    - (1 - eTheta**2) ** 0.5 * sinOmg * self.d_omega_d_par("EDOT")
                )
            )

    def d_beta_d_DTH(self):
        """Derivative.

        Computes::

            dBeta/dDth = a1/c*((-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())

        return self.a1() / c.c * (-eTheta) / np.sqrt(1 - eTheta**2) * cosOmg

    ##################################################

    def delayR(self):
        """Roemer delay defined in T. Damour and N. Deruelle (1986)

        Computes::

            delayR = alpha*(cos(E)-er) + beta*sin(E)
        """
        er = self.er()
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return self.alpha() * (cosE - er) + self.beta() * sinE

    ##################################################
    def Dre(self):
        """Dre defined in T. Damour and N. Deruelle (1986) equation [48]

        Computes::

            delayR = alpha*(cos(E)-er) + beta*sin(E)
            delayE = gamma*sin(E)
            Dre = delayR + delayE
        """

        return self.delayR() + self.delayE()

    def d_Dre_d_par(self, par):
        """Derivative.

        Computes::

           Dre = alpha*(cos(E)-er)+(beta+gamma)*sin(E)
           dDre = alpha*(-der-dE*sin(E)) + (cos[E]-er)*dalpha +
                  (dBeta+dGamma)*sin(E) + (beta+gamma)*cos(E)*dE
           dDre/dpar = alpha*(-der/dpar-dE/dpar*sin(E)) +
                       (cos[E]-er)*dalpha/dpar +
                       (dBeta/dpar+dGamma/dpar)*sin(E) +
                       (beta+gamma)*cos(E)*dE/dpar
            er = e + Dr
        """
        Dre = self.Dre()
        par_obj = getattr(self, par)
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            # First term
            term1 = self.alpha() * (
                -self.prtl_der("er", par) - self.prtl_der("E", par) * sinE
            )
            # Second term
            term2 = (cosE - self.er()) * self.prtl_der("alpha", par)
            # Third term
            term3 = (self.prtl_der("beta", par) + self.prtl_der("GAMMA", par)) * sinE
            # Fourth term
            term4 = (self.beta() + self.GAMMA) * cosE * self.prtl_der("E", par)

            return (term1 + term2 + term3 + term4).to(Dre.unit / par_obj.unit)

    #################################################
    def Drep(self):
        """Derivative of Dre respect to T. Damour and N. Deruelle (1986) equation [49]

        Computes::

           Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return -self.alpha() * sinE + (self.beta() + self.GAMMA) * cosE

    def d_Drep_d_par(self, par):
        """Derivative.

        Computes::

           Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
           dDrep = -alpha*cos(E)*dE + cos(E)*(dbeta+dgamma)
                   -(beta+gamma)*dE*sin(E)-dalpha*sin(E)
           dDrep/dPar = -sin(E)*dalpha/dPar
                        -(alpha*cos(E)+(beta+gamma)*sin(E))*dE/dPar
                        + cos(E)(dbeta/dPar+dgamma/dPar)

        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        Drep = self.Drep()
        par_obj = getattr(self, par)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            # first term
            term1 = -sinE * self.prtl_der("alpha", par)
            # second term
            term2 = -(
                self.alpha() * cosE + (self.beta() + self.GAMMA) * sinE
            ) * self.prtl_der("E", par)
            # Third term
            term3 = cosE * (self.prtl_der("beta", par) + self.prtl_der("GAMMA", par))

            return (term1 + term2 + term3).to(Drep.unit / par_obj.unit)

    #################################################
    def Drepp(self):
        """Derivative of Drep respect to T. Damour and N. Deruelle (1986) equation [50]

        Computes::

           Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return -self.alpha() * cosE - (self.beta() + self.GAMMA) * sinE

    def d_Drepp_d_par(self, par):
        """Derivative.

        Computes::

            Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
            dDrepp = -(beta+gamma)*cos(E)*dE - cos(E)*dalpha
                     +alpha*sin(E)*dE - (dbeta+dgamma)*sin(E)
            dDrepp/dPar = -cos(E)*dalpha/dPar
                          +(alpha*sin(E)-(beta+gamma)*cos(E))*dE/dPar
                          -(dbeta/dPar+dgamma/dPar)*sin(E)
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        Drepp = self.Drepp()
        par_obj = getattr(self, par)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            # first term
            term1 = -cosE * self.prtl_der("alpha", par)
            # second term
            term2 = (
                self.alpha() * sinE - (self.beta() + self.GAMMA) * cosE
            ) * self.prtl_der("E", par)
            # Third term
            term3 = -sinE * (self.prtl_der("beta", par) + self.prtl_der("GAMMA", par))

            return (term1 + term2 + term3).to(Drepp.unit / par_obj.unit)

    #################################################

    def nhat(self):
        """nhat defined as T. Damour and N. Deruelle (1986) equation [51]

        Computes::

           nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()
        """
        cosE = np.cos(self.E())
        return 2.0 * np.pi / self.pb().to("second") / (1 - self.ecc() * cosE)

    def d_nhat_d_par(self, par):
        """Derivative.

        Computes::

           nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()?
           dnhat = -2*pi*dPB/PB^2*(1-ecc*cos(E))
                   -2*pi*(-cos(E)*decc+ecc*sin(E)*dE)/PB*(1-ecc*cos(E))^2
           dnhat/dPar = -2*pi/(PB*(1-ecc*cos(E))*((dPB/dPar)/PB -
                        (-cos(E)*decc/dPar+ecc*sin(E)*dE/dpar)/(1-e*cos(E)))
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            oneMeccTcosE = 1 - self.ecc() * cosE
            fctr = -2 * np.pi / self.pb() / oneMeccTcosE

            return fctr * (
                self.prtl_der("PB", par) / self.pb()
                - (
                    cosE * self.prtl_der("ecc", par)
                    - self.ecc() * sinE * self.prtl_der("E", par)
                )
                / oneMeccTcosE
            )

    #################################################
    def delayInverse(self):
        """DD model Inverse timing delay.

        T. Damour and N. Deruelle (1986) equation [46-52]

        This part is convert the delay argument from proper time to coordinate
        time. The Roemer delay and Einstein are included in the calculation.
        It uses there iterations to approximate the Roemer delay and Einstein
        delay.

        T. Damour and N. Deruelle (1986) equation [43]. The equation [52] gives a
        taylor expansion of equation [43].

        Computes::

            u - e*sin(u) = n(t-T0)
            nhat = du/dt
            nhatp  = d^2u/dt^2
            Drep = dDre/du
            Drepp = d^2Dre/du^2
            Dre(t-Dre(t-Dre(t)))  =  Dre(u) - Drep(u)*nhat*Dre(t-Dre(t))
                                  =  Dre(u) - Drep(u)*nhat*(Dre(u)-Drep(u)*nhat*Dre(t))
                                     + 1/2 (Drepp(u)*nhat^2 + Drep(u) * nhat * nhatp) * (Dre(t)-...)^2
                                  = Dre(t)*(1 - nhat * Drep(u) + (nhat*Drep)^2 +
                                    1/2*nhat^2* Dre*Drepp - 1/2*e*sin(u)/(1-e*cos(u)*nhat^2*Drep*Drep))

        Here u is equivalent to E in the function.
        """
        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        e = self.ecc()
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return (
            Dre
            * (
                1
                - nHat * Drep
                + (nHat * Drep) ** 2
                + 1.0 / 2 * nHat**2 * Dre * Drepp
                - 1.0 / 2 * e * sinE / (1 - e * cosE) * nHat**2 * Dre * Drep
            )
        ).decompose()

    def d_delayI_d_par(self, par):
        """Derivative on delay inverse."""
        e = self.ecc()
        sE = np.sin(self.E())
        cE = np.cos(self.E())
        dE_dpar = self.prtl_der("E", par)
        decc_dpar = self.prtl_der("ecc", par)

        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        delayI = self.delayInverse()

        dDre_dpar = self.d_Dre_d_par(par)
        dDrep_dpar = self.d_Drep_d_par(par)
        dDrepp_dpar = self.d_Drepp_d_par(par)
        dnhat_dpar = self.d_nhat_d_par(par)
        oneMeccTcosE = 1 - e * cE  # 1-e*cos(E)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            x = -1.0 / 2.0 * e * sE / oneMeccTcosE  # -1/2*e*sin(E)/(1-e*cos(E))

            dx_dpar = (
                -sE / (2 * oneMeccTcosE**2) * decc_dpar
                + e * (e - cE) / (2 * oneMeccTcosE**2) * dE_dpar
            )

            diDelay_dDre = (
                1
                + (Drep * nHat) ** 2
                + Dre * Drepp * nHat**2
                + Drep * nHat * (2 * Dre * nHat * x - 1)
            )
            diDelay_dDrep = Dre * nHat * (2 * Drep * nHat + Dre * nHat * x - 1)
            diDelay_dDrepp = (Dre * nHat) ** 2 / 2
            diDelay_dnhat = Dre * (
                -Drep
                + 2 * Drep**2 * nHat
                + nHat * Dre * Drepp
                + 2 * x * nHat * Dre * Drep
            )
            diDelay_dx = (Dre * nHat) ** 2 * Drep

            return (
                dDre_dpar * diDelay_dDre
                + dDrep_dpar * diDelay_dDrep
                + dDrepp_dpar * diDelay_dDrepp
                + dx_dpar * diDelay_dx
                + dnhat_dpar * diDelay_dnhat
            )

    #################################################
    def delayS(self):
        """Binary shapiro delay

        T. Damour and N. Deruelle (1986) equation [26]
        """
        e = self.ecc()
        cE = np.cos(self.E())
        sE = np.sin(self.E())
        sOmega = np.sin(self.omega())
        cOmega = np.cos(self.omega())
        TM2 = self.M2.value * Tsun

        return (
            -2
            * TM2
            * np.log(
                1
                - e * cE
                - self.SINI * (sOmega * (cE - e) + (1 - e**2) ** 0.5 * cOmega * sE)
            )
        )

    def d_delayS_d_par(self, par):
        """Derivative.

        Computes::

           dsDelay/dPar = dsDelay/dTM2*dTM2/dPar+
                          dsDelay/decc*decc/dPar+
                          dsDelay/dE*dE/dPar+
                          dsDelay/domega*domega/dPar+
                          dsDelay/dSINI*dSINI/dPar
        """
        e = self.ecc()
        cE = np.cos(self.E())
        sE = np.sin(self.E())
        sOmega = np.sin(self.omega())
        cOmega = np.cos(self.omega())
        TM2 = self.M2.value * Tsun

        logNum = (
            1
            - e * cE
            - self.SINI * (sOmega * (cE - e) + (1 - e**2) ** 0.5 * cOmega * sE)
        )
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dTM2_dpar = self.prtl_der("TM2", par)
            dsDelay_dTM2 = -2 * np.log(logNum)
            decc_dpar = self.prtl_der("ecc", par)
            dsDelay_decc = (
                -2
                * TM2
                / logNum
                * (-cE - self.SINI * (-e * cOmega * sE / np.sqrt(1 - e**2) - sOmega))
            )
            dE_dpar = self.prtl_der("E", par)
            dsDelay_dE = (
                -2
                * TM2
                / logNum
                * (
                    e * sE
                    - self.SINI * (np.sqrt(1 - e**2) * cE * cOmega - sE * sOmega)
                )
            )
            domega_dpar = self.prtl_der("omega", par)
            dsDelay_domega = (
                2
                * TM2
                / logNum
                * self.SINI
                * ((cE - e) * cOmega - np.sqrt(1 - e**2) * sE * sOmega)
            )
            dSINI_dpar = self.prtl_der("SINI", par)
            dsDelay_dSINI = (
                -2
                * TM2
                / logNum
                * (-((1 - e**2) ** 0.5) * cOmega * sE - (cE - e) * sOmega)
            )
            return (
                dTM2_dpar * dsDelay_dTM2
                + decc_dpar * dsDelay_decc
                + dE_dpar * dsDelay_dE
                + domega_dpar * dsDelay_domega
                + dSINI_dpar * dsDelay_dSINI
            )

    #################################################
    def delayE(self):
        """Binary Einstein delay

        T. Damour and N. Deruelle (1986) equation [25]
        """

        return self.GAMMA * np.sin(self.E())

    def delayA(self):
        """Binary Aberration delay

        T. Damour and N. Deruelle (1986) equation [27]
        """
        omgPlusAe = self.omega() + self.nu()
        et = self.ecc()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())

        return self.A0 * (np.sin(omgPlusAe) + et * sinOmg) + self.B0 * (
            np.cos(omgPlusAe) + et * cosOmg
        )

    def d_delayA_d_par(self, par):
        """Derivative.

        Computes::

           aDelay = A0*(sin(omega+E)+e*sin(omega))+B0*(cos(omega+E)+e*cos(omega))
           daDelay/dpar = daDelay/dA0*dA0/dPar+     (1)
                          daDelay/dB0*dB0/dPar+     (2)
                          daDelay/domega*domega/dPar+    (3)
                          daDelay/dnu*dnu/dPar+        (4)
                          daDelay/decc*decc/dPar        (5)
        """
        e = self.ecc()
        sOmega = np.sin(self.omega())
        cOmega = np.cos(self.omega())
        snu = np.sin(self.nu())
        cnu = np.cos(self.nu())
        A0 = self.A0
        B0 = self.B0
        omgPlusAe = self.omega() + self.nu()
        if par == "A0":
            return e * sOmega + np.sin(omgPlusAe)
        elif par == "B0":
            return e * cOmega + np.cos(omgPlusAe)
        else:
            domega_dpar = self.prtl_der("omega", par)
            daDelay_domega = A0 * (np.cos(omgPlusAe) + e * cOmega) - B0 * (
                np.sin(omgPlusAe) + e * sOmega
            )

            dnu_dpar = self.prtl_der("nu", par)
            daDelay_dnu = A0 * np.cos(omgPlusAe) - B0 * np.sin(omgPlusAe)

            decc_dpar = self.prtl_der("ecc", par)
            daDelay_decc = A0 * sOmega + B0 * cOmega
            return (
                domega_dpar * daDelay_domega
                + dnu_dpar * daDelay_dnu
                + decc_dpar * daDelay_decc
            )

    #################################################

    def DDdelay(self):
        """Full DD model delay"""
        return self.delayInverse() + self.delayS() + self.delayA()

    def d_DDdelay_d_par(self, par):
        """Full DD model delay derivative"""
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return (
                self.d_delayI_d_par(par)
                + self.d_delayS_d_par(par)
                + self.d_delayA_d_par(par)
            )
