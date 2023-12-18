"""Model parametrizing Shapiro delay differently."""

# This Python file uses the following encoding: utf-8
import astropy.units as u
import numpy as np
from loguru import logger as log

from .ELL1_model import ELL1BaseModel


class ELL1Hmodel(ELL1BaseModel):
    """ELL1H pulsar binary model using H3, H4 or STIGMA as shapiro delay parameters.

    Note
    ----
    Based on Freire and Wex (2010)

    The :class:`~pint.models.binary_ell1.BinaryELL1H` model parameterizes the Shapiro
    delay differently compare to the :class:`~pint.models.binary_ell1.BinaryELL1`
    model. A fourier series expansion is used for the Shapiro delay:

    .. math::

        \\Delta_S = -2r \\left( \\frac{a_0}{2} + \\Sum_k (a_k \\cos k\\phi + b_k \\sin k \phi) \\right)

    The first two harmonics are generlly absorbed by the ELL1 Roemer delay.
    Thus, :class:`~pint.models.binary_ell1.BinaryELL1H` uses the series from the third
    harmonic and higher.

    Notes
    -----
    Default value in `pint` for `NHARMS` is 7, while in `tempo2` it is 4.

    References
    ----------
    - Freire and Wex (2010), MNRAS, 409, 199 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract

    """

    def __init__(self):
        super().__init__()
        self.binary_name = "ELL1H"
        self.param_default_value.update(
            {
                "H3": 0.0 * u.second,
                "H4": 0.0 * u.second,
                "STIGMA": 0.0 * u.Unit(""),
                "NHARMS": 3 * u.Unit(""),
            }
        )
        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values()  # Set parameters to default values.
        self.ELL1H_interVars = ["stigma"]
        self.add_inter_vars(self.ELL1H_interVars)
        # NOTE since we are ELL1H is a inheritance of ELL1. We need to make sure
        # ELL1 delay and ELL1 derivative  function is not in the list.
        self.binary_delay_funcs = [self.ELL1Hdelay]
        self.d_binarydelay_d_par_funcs = [self.d_ELL1Hdelay_d_par]
        # Put those default value here.  It can be changed for a new model.
        self.fit_params = ["H3"]
        self.ds_func_list = [
            self.delayS3p_H3_STIGMA_approximate,
            self.delayS_H3_STIGMA_exact,
            self.delayS3p_H3_STIGMA_exact,
        ]
        self.ds_func = self.delayS3p_H3_STIGMA_approximate

    def delayS(self):
        if set(self.fit_params) == {"H3", "H4"}:
            if self.H3 == 0.0:
                if self.H4 == 0.0:
                    stigma = 0.0
                else:
                    raise ValueError("To use H4, H3 needs to be significant(H3 != 0).")
            else:
                stigma = self.H4 / self.H3
        elif set(self.fit_params) == {"H3", "STIGMA"}:
            stigma = self.STIGMA
        elif set(self.fit_params) == {"H3"}:
            stigma = 0.0
        else:
            raise NotImplementedError(
                f"ELL1H did not implement {str(self.fit_params)} parameter set yet."
            )
        return self.ds_func(self.H3, stigma, self.NHARMS)

    def ELL1Hdelay(self):
        # TODO need aberration
        return self.delayI() + self.delayS()

    def get_SINI_from_STIGMA(self):
        return 2 * self.STIGMA / (1 + self.STIGMA**2)

    def get_TM2_from_H3_STIGMA(self):
        return self.H3 / self.STIGMA**3

    def get_SINI_from_H3_H4(self):
        return 2 * self.H3 * self.H4 / (self.H3**2 + self.H4**2)

    def get_TM2_from_H3_H4(self):
        return (self.H3) ** 4 / (self.H4) ** 3

    def _ELL1H_fourier_basis(self, k, derivative=False):
        """Select the fourier basis and part of the coefficient depend on the parity of the harmonics k"""
        if k % 2 == 0:
            pwr = (k + 2) / 2
            basis_func = (lambda x: -1 * np.sin(x)) if derivative else np.cos
        else:
            pwr = (k + 1) / 2
            basis_func = np.cos if derivative else np.sin
        return pwr, basis_func

    def fourier_component(self, stigma, k, factor_out_power=0):
        """Freire and Wex (2010), Eq (13)

        Parameters
        ----------
        stigma: float
           Orthometric ratio
        k: positive integer
           Order of harmonics.
        factor_out_power: int
           The power factor out from the coefficient. For example, when
           factor_out_power = 1, for k = 3. we have:
           (-1) ^ ((k+1)/2) * 2 / k * stigma ^ 2, the one extra stigma is
           factored out to other terms.

        Returns
        -------
        float
            The coefficient of fourier component and the basis.
        """

        if k != 0:
            pwr, basis_func = self._ELL1H_fourier_basis(k)
            return (
                (-1) ** (pwr) * 2.0 / k * stigma ** (k - factor_out_power),
                basis_func,
            )
        else:
            basis_func = np.cos
            # a0 is -1 * np.log(1 + stigma ** 2)
            # But the in the Fourier series it is a0/2
            return (
                -1.0 * np.log(1 + stigma**2) * stigma ** (-factor_out_power),
                basis_func,
            )

    def d_fourier_component_d_stigma(self, stigma, k, factor_out_power=0):
        """This is a method to compute the derivative of a fourier component."""

        # prevent factor out zeros.
        if k == 0:
            # a0 is -1 * np.log(1 + stigma ** 2)
            # But the in the Fourier series it is a0/2
            return -2.0 / (1 + stigma**2.0) * stigma ** (1 - factor_out_power), np.cos
        pwr, basis_func = self._ELL1H_fourier_basis(k)

        if stigma == 0.0 and k == factor_out_power:
            return 0.0, basis_func
        else:
            return (
                (-1) ** (pwr)
                * 2.0
                * (k - factor_out_power)
                / k
                * stigma ** (k - factor_out_power - 1),
                basis_func,
            )

    def d_fourier_component_d_phi(self, stigma, k, factor_out_power=0):
        if k != 0:
            pwr, basis_func = self._ELL1H_fourier_basis(k, derivative=True)
            return (-1) ** (pwr) * 2.0 * stigma ** (k - factor_out_power), basis_func
        else:
            basis_func = -np.sin
            return 0, basis_func

    def d_STIGMA_d_H4(self):
        return 0.0 if self.H3 == 0.0 else 1.0 / self.H3

    def d_STIGMA_d_H3(self):
        if set(self.fit_params) == {"H3", "H4"}:
            d_stigma_d_H3 = 0.0 if self.H3 == 0.0 else -self.H4 / self.H3 / self.H3
        elif set(self.fit_params) == {"H3", "STIGMA"}:
            d_stigma_d_H3 = 0.0
        elif set(self.fit_params) == {"H3"}:
            d_stigma_d_H3 = 0.0
        else:
            raise NotImplementedError(
                f"ELL1H did not implement {str(self.fit_params)} parameter set yet."
            )
        return d_stigma_d_H3

    def ELL1H_shapiro_delay_fourier_harms(
        self, selected_harms, phi, stigma, factor_out_power=0
    ):
        """Fourier series harms of shapiro delay.

        One can select the start term and end term.
        Freire and Wex (2010), Eq. (10)

        Parameters
        ----------
        selected_harms: numpy.array or list of positive integers
           The selected harms for fourier harms
        phi: float
           Orbit phase in ELL1 model
        stigma: float

        Returns
        -------
        np.ndarray
            The summation of harmonics
        """
        harms = np.zeros((len(selected_harms), len(phi)))
        # To prevent factor out zeros
        if stigma == 0.0 and selected_harms.min() < factor_out_power:
            raise ValueError(
                "Can not factor_out_power can not bigger than" " the selected_harms."
            )
        for ii, k in enumerate(selected_harms):
            coeff, basis_func = self.fourier_component(
                stigma, k, factor_out_power=factor_out_power
            )
            harms[ii] = coeff * basis_func(k * phi)
        return np.sum(harms, axis=0)

    def d_ELL1H_fourier_harms_d_par(
        self, selected_harms, phi, stigma, par, factor_out_power=0
    ):
        """This is an overall derivative function."""
        # Find the right derivative  function for fourier components
        df_name = f"d_fourier_component_d_{par.lower()}"
        par_obj = getattr(self, par)
        try:
            df_func = getattr(self, df_name)
        except AttributeError:
            return 0.0 * u.Unit(None) / par_obj.Unit
        d_harms = np.zeros((len(selected_harms), len(phi)))
        # To prevent factor out zeros
        if stigma == 0.0 and selected_harms.min() < factor_out_power:
            raise ValueError(
                "Can not factor_out_power can not bigger than" " the selected_harms."
            )
        for ii, k in enumerate(selected_harms):
            coeff, basis_func = df_func(stigma, k, factor_out_power=factor_out_power)
            d_harms[ii] = coeff * basis_func(k * phi)
        return np.sum(d_harms, axis=0)

    def delayS3p_H3_STIGMA_approximate(self, H3, stigma, end_harm=6):
        """Shapiro delay using third or higher harmonics, appropriate for medium inclinations.

        defined in Freire and Wex (2010), Eq (19).
        """
        Phi = self.Phi()
        selected_harms = np.arange(3, end_harm + 1)
        sum_fharms = self.ELL1H_shapiro_delay_fourier_harms(
            selected_harms, Phi, stigma, factor_out_power=3
        )
        return -2.0 * H3 * sum_fharms

    def d_delayS3p_H3_STIGMA_approximate_d_H3(self, H3, stigma, end_harm=6):
        """derivative of delayS3p_H3_STIGMA with respect to H3"""
        Phi = self.Phi()
        selected_harms = np.arange(3, end_harm + 1)
        sum_fharms = self.ELL1H_shapiro_delay_fourier_harms(
            selected_harms, Phi, stigma, factor_out_power=3
        )
        return -2.0 * sum_fharms

    def d_delayS3p_H3_STIGMA_approximate_d_STIGMA(self, H3, stigma, end_harm=6):
        """derivative of delayS3p_H3_STIGMA with respect to STIGMA"""
        Phi = self.Phi()
        selected_harms = np.arange(3, end_harm + 1)
        sum_d_fharms = self.d_ELL1H_fourier_harms_d_par(
            selected_harms, Phi, stigma, "STIGMA", factor_out_power=3
        )
        return -2.0 * H3 * sum_d_fharms

    def d_delayS3p_H3_STIGMA_approximate_d_Phi(self, H3, stigma, end_harm=6):
        """derivative of delayS3p_H3_STIGMA with respect to Phi"""
        Phi = self.Phi()
        selected_harms = np.arange(3, end_harm + 1)
        sum_d_fharms = self.d_ELL1H_fourier_harms_d_par(
            selected_harms, Phi, stigma, "Phi", factor_out_power=3
        )
        return -2.0 * H3 * sum_d_fharms

    def delayS3p_H3_STIGMA_exact(self, H3, stigma, end_harm=None):
        """Shapiro delay (3rd hamonic and higher) using the exact form for very high inclinations.

        Defined in Freire and Wex (2010), Eq (28).
        """
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return (
            -2
            * H3
            / stigma**3
            * (
                np.log(lognum)
                + 2 * stigma * np.sin(Phi)
                - stigma * stigma * np.cos(2 * Phi)
            )
        )

    def d_delayS3p_H3_STIGMA_exact_d_H3(self, H3, stigma, end_harm=None):
        """derivative of exact Shapiro delay (3rd hamonic and higher) with respect to H3"""
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return (
            -2
            / stigma**3
            * (
                np.log(lognum)
                + 2 * stigma * np.sin(Phi)
                - stigma * stigma * np.cos(2 * Phi)
            )
        )

    def d_delayS3p_H3_STIGMA_exact_d_STIGMA(self, H3, stigma, end_harm=None):
        """derivative of exact Shapiro delay (3rd hamonic and higher) with respect to STIGMA"""
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return (
            -2
            * H3
            / stigma**4
            * (
                -3 * np.log(lognum)
                + 2 * stigma * (stigma - np.sin(Phi)) / lognum
                - 4 * stigma * np.sin(Phi)
                + stigma**2 * np.cos(Phi)
            )
        )

    def d_delayS3p_H3_STIGMA_exact_d_Phi(self, H3, stigma, end_harm=None):
        """derivative  of exact Shapiro delay (3rd hamonic and higher) with respect to phase"""
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return (
            -4
            * H3
            / stigma**2
            * (-np.cos(Phi) / lognum + np.cos(Phi) + stigma * np.sin(2 * Phi))
        )

    def delayS_H3_STIGMA_exact(self, H3, stigma, end_harm=None):
        """Shapiro delay (including all harmonics) using the exact form for very high inclinations.

        Defined in Freire and Wex (2010), Eq (29).
        """
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return -2 * H3 / stigma**3 * np.log(lognum)

    def d_delayS_H3_STIGMA_exact_d_H3(self, H3, stigma, end_harm=None):
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return -2 / stigma**3 * np.log(lognum)

    def d_delayS_H3_STIGMA_exact_d_STIGMA(self, H3, stigma, end_harm=None):
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return (
            -2
            * H3
            / stigma**4
            * (-3 * np.log(lognum) + 2 * stigma * (stigma - np.sin(Phi)) / lognum)
        )

    def d_delayS_H3_STIGMA_exact_d_Phi(self, H3, stigma, end_harm=None):
        Phi = self.Phi()
        lognum = 1 + stigma**2 - 2 * stigma * np.sin(Phi)
        return 4 * H3 / stigma**2 * (np.cos(Phi) / lognum)

    def d_delayS_d_par(self, par):
        if set(self.fit_params) == {"H3", "H4"}:
            if self.H3 == 0:
                if self.H4 == 0.0:
                    stigma = 0.0
                else:
                    ValueError("To use H4, H3 needs to be significant(H3 >= H4).")
            else:
                stigma = self.H4 / self.H3
        elif set(self.fit_params) == {"H3", "STIGMA"}:
            stigma = self.STIGMA
        elif set(self.fit_params) == {"H3"}:
            stigma = 0.0
        else:
            raise NotImplementedError(
                f"ELL1H fit not implemented for {self.fit_params} parameters"
            )

        d_ds_func_name_base = f"d_{self.ds_func.__name__}_d_"
        d_delayS_d_H3_func = getattr(self, f"{d_ds_func_name_base}H3")
        d_delayS_d_Phi_func = getattr(self, f"{d_ds_func_name_base}Phi")
        d_delayS_d_STIGMA_func = getattr(self, f"{d_ds_func_name_base}STIGMA")

        d_delayS_d_H3 = d_delayS_d_H3_func(self.H3, stigma, self.NHARMS)
        d_delayS_d_Phi = d_delayS_d_Phi_func(self.H3, stigma, self.NHARMS)
        d_delayS_d_STIGMA = d_delayS_d_STIGMA_func(self.H3, stigma, self.NHARMS)

        d_H3_d_par = self.prtl_der("H3", par)
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_STIGMA_d_par = self.prtl_der("STIGMA", par)

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_delayS_d_par = (
                d_delayS_d_H3 * d_H3_d_par
                + d_delayS_d_Phi * d_Phi_d_par
                + d_delayS_d_STIGMA * d_STIGMA_d_par
            )
        return d_delayS_d_par

    def d_ELL1Hdelay_d_par(self, par):
        return self.d_delayI_d_par(par) + self.d_delayS_d_par(par)
