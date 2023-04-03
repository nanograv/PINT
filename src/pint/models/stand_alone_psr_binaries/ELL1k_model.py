"""The ELL1k model for approximately handling near-circular orbits."""
import astropy.constants as c
import astropy.units as u
import numpy as np

from .ELL1_model import ELL1model


class ELL1kmodel(ELL1model):
    """This is a class for base ELL1k pulsar binary model.

    ELL1k model is a generalization of ELL1 model to handle systems with
    large advance of periastron.

    References
    ----------
    - Susobhanan et al. (2018), MNRAS, 480 (4), 5260-5271 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.5260S/abstract
    """

    def __init__(self):
        super().__init__()

        self.binary_name = "ELL1k"
        self.binary_delay_funcs = [self.ELL1kdelay]
        self.d_binarydelay_d_par_funcs = [self.d_ELL1kdelay_d_par]

        self.param_default_value.pop("EPS1DOT")
        self.param_default_value.pop("EPS2DOT")
        self.param_default_value.pop("EDOT")
        self.param_default_value.update(
            {"OMDOT": u.Quantity(0, "deg/year"), "LNEDOT": u.Quantity(0, "1/year")}
        )

        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values()  # Set parameters to default values.

        # self.orbits_func = self.orbits_ELL1

    @property
    def tt0(self):
        return self.ttasc()

    ###############################

    def eps1(self):
        """EPS1 as a function of time
        Susobhanan+ 2018 Eq. 15"""
        eps10 = self.EPS1
        eps20 = self.EPS2
        omdot = self.OMDOT
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return (1 + lnedot * dt) * (
            eps10 * np.cos(omdot * dt) + eps20 * np.sin(omdot * dt)
        )

    def d_eps1_d_EPS1(self):
        omdot = self.OMDOT
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return (1 + lnedot * dt) * np.cos(omdot * dt)

    def d_eps1_d_EPS2(self):
        omdot = self.OMDOT
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return (1 + lnedot * dt) * np.sin(omdot * dt)

    def d_eps1_d_OMDOT(self):
        dt = self.ttasc()
        return self.eps2() * dt

    def d_eps1_d_LNEDOT(self):
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return self.eps1() * dt / (1 + lnedot * dt)

    def d_eps1_d_TASC(self):
        omdot = self.OMDOT
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return -self.eps1() * lnedot / (1 + lnedot * dt) - self.eps2() * omdot

    def eps2(self):
        """EPS2 as a function of time
        Susobhanan+ 2018 Eq. 15"""
        eps10 = self.EPS1
        eps20 = self.EPS2
        omdot = self.OMDOT
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return (1 + lnedot * dt) * (
            eps20 * np.cos(omdot * dt) - eps10 * np.sin(omdot * dt)
        )

    def d_eps2_d_EPS1(self):
        return -self.d_eps1_d_EPS2()

    def d_eps2_d_EPS2(self):
        return -self.d_eps1_d_EPS1()

    def d_eps2_d_OMDOT(self):
        dt = self.ttasc()
        return -self.eps1() * dt

    def d_eps2_d_LNEDOT(self):
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return self.eps2() * dt / (1 + lnedot * dt)

    def d_eps2_d_TASC(self):
        omdot = self.OMDOT
        lnedot = self.LNEDOT
        dt = self.ttasc()
        return -self.eps2() * lnedot / (1 + lnedot * dt) + self.eps1() * omdot

    def delayR(self):
        """ELL1k Roemer delay in proper time.
        A Susobhanan et al 2018 Eq. 6
        There is an extra term (-3*a1*eps1)/(2*c) as compared to the ELL1 model."""

        Phi = self.Phi()
        return (
            self.a1()
            / c.c
            * (
                np.sin(Phi)
                + 0.5
                * (self.eps2() * np.sin(2 * Phi) - self.eps1() * (np.cos(2 * Phi) + 3))
            )
        ).decompose()

    def d_Dre_d_par(self, par):
        """Derivative computation.

        Computes::

            Dre = delayR = a1/c.c*(sin(phi) - 0.5* eps1*(cos(2*phi) + 3) +  0.5* eps2*sin(2*phi))
            d_Dre_d_par = d_a1_d_par/c.c * (sin(phi) - 0.5* eps1*(cos(2*phi) + 3) +  0.5* eps2*sin(2*phi))
                            + d_Dre_d_Phi * d_Phi_d_par
                            + d_Dre_d_eps1 * d_eps1_d_par
                            + d_Dre_d_eps2 * d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der("a1", par)
        d_Dre_d_Phi = self.Drep()
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Dre_d_eps1 = a1 / c.c * (-0.5 * (np.cos(2 * Phi) + 3))
        d_Dre_d_eps2 = a1 / c.c * (0.5 * np.sin(2 * Phi))

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Dre_d_par = (
                d_a1_d_par
                / c.c
                * (
                    np.sin(Phi)
                    - 0.5 * eps1 * (np.cos(2 * Phi) + 3)
                    + 0.5 * eps2 * np.sin(2 * Phi)
                )
                + d_Dre_d_Phi * d_Phi_d_par
                + d_Dre_d_eps1 * self.prtl_der("eps1", par)
                + d_Dre_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Dre_d_par

    def ELL1kdelay(self):
        # TODO add aberration delay
        return self.delayI() + self.delayS()

    def d_ELL1kdelay_d_par(self, par):
        return self.d_delayI_d_par(par) + self.d_delayS_d_par(par)
