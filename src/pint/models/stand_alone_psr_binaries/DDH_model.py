"""The DDS model - Damour and Deruelle with alternate Shapiro delay parametrization."""
import astropy.constants as c
import astropy.units as u
import numpy as np
from loguru import logger as log

from pint import Tsun

from .DD_model import DDmodel


class DDHmodel(DDmodel):
    """Damour and Deruelle model modified to use H3/STIGMA parameter for Shapiro delay.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

        H3
        STIGMA

    It also removes:

        SINI
        M2
            use ``H3``/``STIGMA`` instead


    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDH

    Note
    ----
    Based on Freire and Wex (2010) and Weisberg and Huang (2016)

    Notes
    -----
    This uses the full expression for the Shapiro delay, not the harmonic
    decomposition used in :class:`pint.models.stand_alone_psr_binaries.ELL1H_model.ELL1Hmodel`.

    References
    ----------
    - Freire and Wex (2010), MNRAS, 409, 199 [1]_
    - Weisberg & Huang (2016), ApH, 829 (1), 55 [2]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2016ApJ...829...55W/abstract

    """

    def __init__(self, t=None, input_params=None):
        super().__init__()
        self.binary_name = "DDH"
        self.param_default_value.update(
            {
                "H3": 0.0 * u.second,
                "STIGMA": 0.0 * u.Unit(""),
            }
        )
        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values()  # Set parameters to default values.

    @property
    def SINI(self):
        return 2 * self.STIGMA / (1 + self.STIGMA**2)

    @property
    def M2(self):
        return self.H3 / self.STIGMA**3 / Tsun.value

    @SINI.setter
    def SINI(self, val):
        log.debug(
            "DDH model uses H3/STIGMA as Shapiro delay parameter. SINI will not be used."
        )

    @M2.setter
    def M2(self, val):
        log.debug(
            "DDH model uses H3/STIGMA as Shapiro delay parameter. M2 will not be used."
        )

    def d_delayS_d_par(self, par):
        """Derivative.

        Computes::

            dsDelay/dPar = (dsDelay/dH3) * (dH3/dPar) +
                        (dsDelay/decc) * (decc/dPar) +
                        (dsDelay/dE) * (dE/dPar) +
                        (dsDelay/domega) * (domega/dPar) +
                        (dsDelay/dSTIGMA) * (dSTIGMA/dPar)
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
            dH3_dpar = self.prtl_der("H3", par)
            dsDelay_dH3 = -2 * np.log(logNum) / self.STIGMA**3
            dSTIGMA_dpar = self.prtl_der("STIGMA", par)
            dsDelay_dSTIGMA = 6 * self.H3 / self.STIGMA**4 / Tsun.value * np.log(
                1
                - e * cE
                - 2
                * self.STIGMA
                / (self.STIGMA**2 + 1)
                * (sOmega * (cE - e) + (-(e**2) + 1) ** 0.5e0 * cOmega * sE)
            ) - 2 * self.H3 / self.STIGMA**3 / Tsun.value * (
                -2
                / (self.STIGMA**2 + 1)
                * (sOmega * (cE - e) + (-(e**2) + 1) ** 0.5e0 * cOmega * sE)
                + 4
                * self.STIGMA**2
                / (self.STIGMA**2 + 1) ** 2
                * (sOmega * (cE - e) + (-(e**2) + 1) ** 0.5e0 * cOmega * sE)
            ) / (
                1
                - e * cE
                - 2
                * self.STIGMA
                / (self.STIGMA**2 + 1)
                * (sOmega * (cE - e) + (-(e**2) + 1) ** 0.5e0 * cOmega * sE)
            )

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
            return (
                dH3_dpar * dsDelay_dH3
                + decc_dpar * dsDelay_decc
                + dE_dpar * dsDelay_dE
                + domega_dpar * dsDelay_domega
                + dSTIGMA_dpar * dsDelay_dSTIGMA
            )
