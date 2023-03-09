"""The ELL1+ model for approximately handling near-circular orbits but including second order eccentricity terms."""
import astropy.constants as c
import astropy.units as u
import numpy as np

from .ELL1_model import ELL1model


class ELL1plusmodel(ELL1model):
    """This is a class for base ELL1+ pulsar binary model.

    ELL1+ model is a generalization of ELL1 model to include terms
    up to second order in eccentricity

    References
    ----------
    - Zhu et al. (2019), MNRAS, 482 (3), 3249-3260 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.3249Z/abstract
    """

    def __init__(self):
        super().__init__()

        self.binary_name = "ELL1+"
        self.binary_delay_funcs = [self.ELL1plusdelay]
        self.d_binarydelay_d_par_funcs = [self.d_ELL1plusdelay_d_par]

        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values()  # Set parameters to default values.

        # self.orbits_func = self.orbits_ELL1

    @property
    def tt0(self):
        return self.ttasc()

    ###############################
    def d_delayR_da1(self):
        """ELL1+ Roemer delay in proper time divided by a1/c"""
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        return (
            np.sin(Phi) + 0.5 * (eps2 * np.sin(2 * Phi) - eps1 * np.cos(2 * Phi))
        ) - (1.0 / 8) * (
            5 * eps1**2 * np.sin(Phi)
            - 3 * eps1**2 * np.sin(3 * Phi)
            - 2 * eps1 * eps2 * np.cos(Phi)
            + 6 * eps1 * eps2 * np.cos(3 * Phi)
            + 3 * eps2**2 * np.sin(Phi)
            + 3 * eps2**2 * np.sin(3 * Phi)
        )

    def d_d_delayR_dPhi_da1(self):
        """d (ELL1+ Roemer delay)/dPhi in proper time divided by a1/c"""
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        return (
            np.cos(Phi)
            + eps1 * np.sin(2 * Phi)
            + eps2 * np.cos(2 * Phi)
            - (1.0 / 8)
            * (
                5 * eps1**2 * np.cos(Phi)
                - 9 * eps1**2 * np.cos(3 * Phi)
                + 2 * eps1 * eps2 * np.sin(Phi)
                - 18 * eps1 * eps2 * np.sin(3 * Phi)
                + 3 * eps2**2 * np.cos(Phi)
                + 9 * eps2**2 * np.cos(3 * Phi)
            )
        )

    def d_dd_delayR_dPhi_da1(self):
        """d^2 (ELL1+ Roemer delay)/dPhi^2 in proper time divided by a1/c"""
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        return (
            -np.sin(Phi)
            + 2 * eps1 * np.cos(2 * Phi)
            - 2 * eps2 * np.sin(2 * Phi)
            - (1.0 / 8)
            * (
                -5 * eps1**2 * np.sin(Phi)
                + 27 * eps1**2 * np.sin(3 * Phi)
                + 2 * eps1 * eps2 * np.cos(Phi)
                - 54 * eps1 * eps2 * np.cos(3 * Phi)
                - 3 * eps2**2 * np.sin(Phi)
                - 27 * eps2**2 * np.sin(3 * Phi)
            )
        )

    def delayR(self):
        """ELL1+ Roemer delay in proper time.
        Include terms up to second order in eccentricity
        Zhu et al. (2019), Eqn. 1
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
                    10 * eps1 * np.sin(Phi)
                    - 6 * eps1 * np.sin(3 * Phi)
                    - 2 * eps2 * np.cos(Phi)
                    + 6 * eps2 * np.cos(3 * Phi)
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
                    + 6 * eps2 * np.sin(Phi)
                    + 6 * eps2 * np.sin(3 * Phi)
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
        # Here we are using full d Dre/dPhi. But Tempo and Tempo2 ELL1+ model
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
                    10 * eps1 * np.cos(Phi)
                    - 18 * eps1 * np.cos(3 * Phi)
                    + 2 * eps2 * np.sin(Phi)
                    - 18 * eps2 * np.sin(3 * Phi)
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
                    + 6 * eps2 * np.cos(Phi)
                    + 18 * eps2 * np.cos(3 * Phi)
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
                    -5 * eps1**2 * np.cos(Phi)
                    + 81 * eps1**2 * np.cos(3 * Phi)
                    - 2 * eps1 * eps2 * np.sin(Phi)
                    + 162 * eps1 * eps2 * np.sin(3 * Phi)
                    - 3 * eps2**2 * np.cos(Phi)
                    - 81 * eps2**2 * np.cos(3 * Phi)
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
                    -10 * eps1 * np.sin(Phi)
                    + 54 * eps1 * np.sin(3 * Phi)
                    + 2 * eps2 * np.cos(Phi)
                    - 54 * eps2 * np.cos(3 * Phi)
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
                    - 6 * eps2 * np.sin(Phi)
                    - 54 * eps2 * np.sin(3 * Phi)
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

    def ELL1plusdelay(self):
        # TODO add aberration delay
        return self.delayI() + self.delayS()

    def d_ELL1plusdelay_d_par(self, par):
        return self.d_delayI_d_par(par) + self.d_delayS_d_par(par)
