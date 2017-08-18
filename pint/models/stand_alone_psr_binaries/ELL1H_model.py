from .ELL1_model import ELL1model
import numpy as np
import astropy.units as u
import astropy.constants as c
from pint import ls,GMsun,Tsun


class ELL1Hmodel(ELL1model):
    """This is a class for ELL1H pulsar binary model.
       ELL1H model is modified model from ELL1. It changes the parameterization
       of the binary shapiro delay.

       Note
       ----
       Ref : Paulo C. C. Freire and Norbert Wex, Mon. Not. R. Astron. Soc. 409,
       199–212 (2010)
       ELL1H model parameterize the shapiro delay differently in different
       inclination case.
       Low-inclination case, shapiro delay can not be separated from Roemer
       delay (P. Freire and N. Wex 2010 section 3.3)
       Medium-inclination case,
       Shapiro delay = −4 * H3 * (1/3 * sin(3 * Phi) - 1/4 * sigma * cos(4 * Phi) \
                       - 1/5 * sigma^2 * sin(5*Phi) + 1/6 * sigma^3 * cos(6 * Phi)+...)
       H3 is a measurable parameter, and sigma is defined in the Eq(12) of the
       paper.

       High-inclination case,
    """
    def __init__(self):
        super(ELL1Hmodel, self).__init__()
        self.binary_name = 'ELL1H'
        self.param_default_value.update({'H3': 0 * u.second})
        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values() # Set parameters to default values.
        self.ELL1H_interVars = ['var_sigma', 'c_bar']
        self.add_inter_vars(self.ELL1H_interVars)
        self.binary_delay_funcs += [self.ELL1Hdelay]
        self.d_binarydelay_d_par_funcs += [self.d_ELL1Hdelay_d_par]

    def c_bar(self):
        """Defined in the (P. Freire and N. Wex 2010) paper Eq (11)
           c_bar = |COS(i)|
        """
        return np.sqrt(1 - self.SINI ** 2.0)

    def var_sigma(self):
        """Defined in the (P. Freire and N. Wex 2010) paper Eq (3) (12)
        """
        cbar = self.c_bar()
        return self.SINI / (1 + cbar)

    def delayS_H3(self):
        """Defined in the (P. Freire and N. Wex 2010) paper Eq (19). This only
           gets to the third order of var sigma
        """
        sigma = self.var_sigma()
        Phi = self.Phi()
        ds = -4.0* self.H3 *(1.0 / 3.0 * np.sin(3.0 * Phi) \
                             + sigma * (- 1.0/4.0  * np.cos(4 * Phi) \
                             - 1.0/5.0 * sigma * np.sin(5*Phi) \
                             + 1.0/6.0 * sigma**2 * np.cos(6 * Phi)))
        return ds

    def ELL1Hdelay(self):
        # TODO need aberration
        return self.delayI() + self.delayS_H3()

    def get_TM2_from_H3(self):
        sigma = self.var_sigma
        return self.H3 / sigma **3

    def d_delayS_H3_d_H3(self):
        Phi = self.Phi()
        sigma = self.var_sigma()
        d_ds_d_h3 = -4.0 * (1.0 / 3.0 * np.sin(3.0 * Phi) \
                            + sigma * (- 1.0/4.0  * np.cos(4 * Phi) \
                            - 1.0/5.0 * sigma * np.sin(5*Phi) \
                            + 1.0/6.0 * sigma**2 * np.cos(6 * Phi)))
        return d_ds_d_h3

    def d_delayS_H3_d_var_sigma(self):
        sigma = self.var_sigma()
        Phi = self.Phi()
        d_ds_d_var_sigma = self.H3 * (np.cos(4 * Phi) \
                              + 8.0/5.0 * sigma * np.sin(5*Phi) \
                              -2 * sigma**2 * np.cos(6 * Phi))
        return d_ds_d_var_sigma

    def d_delayS_H3_d_Phi(self):
        sigma = self.var_sigma()
        Phi = self.Phi()
        ds = -4.0* self.H3 *(np.cos(3.0 * Phi) \
                             + sigma * (np.sin(4 * Phi) \
                             - sigma * np.cos(5*Phi) \
                             - sigma**2 * np.sin(6 * Phi)))
        return d_delayS_H3_d_Phi

    def d_delayS_H3_d_par(self, par):
        pass

    def d_ELL1Hdelay_d_par(self, par):
        pass
