from astropy.table import Table
from warnings import warn
import numpy as np
import astropy.units as u
from pint.models.timing_model import DelayComponent
from pint.models.parameter import floatParameter, prefixParameter, MJDParameter
from pint.utils import taylor_horner
from pint import DMconst

cmu = u.pc / u.cm**3 / u.MHz**2


class Chromatic(DelayComponent):
    """A base chromatic timing model."""

    def __init__(self):
        super().__init__()

        self.cm_value_funcs = []
        self.cm_deriv_funcs = {}

        self.alpha_value_funcs = []
        self.alpha_deriv_funcs = {}

    def chromatic_time_delay(self, cm, alpha, freq):
        """Return the chromatic time delay for a set of frequencies.

        delay_chrom = cm * DMconst * (freq / 1 MHz)**alpha
        """
        dmdelay = cm * DMconst * (freq / u.MHz) ** (-alpha)
        return dmdelay.to(u.s)

    def chromatic_type_delay(self, toas):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for chromatic delay!")
            bfreq = toas.table["freq"]

        cm = self.cm_value(toas)
        alpha = self.alpha_value(toas)
        return self.chromatic_time_delay(cm, alpha, bfreq)

    def cm_value(self, toas):
        """Compute modeled CM value at given TOAs.

        Parameters
        ----------
        toas : `TOAs` object or TOA table(TOAs.table)
            If given a TOAs object, it will use the whole TOA table in the
             `TOAs` object.

        Return
        ------
            CM values at given TOAs in the unit of CM.
        """
        toas_table = toas if isinstance(toas, Table) else toas.table
        cm = np.zeros(len(toas_table)) * self._parent.CM.units

        for cm_f in self.cm_value_funcs:
            cm += cm_f(toas)
        return cm

    def alpha_value(self, toas):
        """Compute modeled chromatic index value at given TOAs.

        Parameters
        ----------
        toas : `TOAs` object or TOA table(TOAs.table)
            If given a TOAs object, it will use the whole TOA table in the
             `TOAs` object.

        Return
        ------
            chromatic index values at given TOAs.
        """
        toas_table = toas if isinstance(toas, Table) else toas.table
        alpha = np.zeros(len(toas_table)) * u.dimensionless_unscaled

        for alpha_f in self.alpha_value_funcs:
            alpha += alpha_f(toas)
        return alpha

    def d_delay_d_cmparam(self, toas, param_name, acc_delay=None):
        """Derivative of delay wrt to DM parameter.

        Parameters
        ----------
        toas : `pint.TOAs` object.
            Input toas.
        param_name : str
            Derivative parameter name
        acc_delay : `astropy.quantity` or `numpy.ndarray`
            Accumulated delay values. This parameter is to keep the unified API,
            but not used in this function.
        """
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]

        param_unit = getattr(self, param_name).units
        d_cm_d_cmparam = np.zeros(toas.ntoas) * cmu / param_unit
        alpha = self.alpha_value(toas)

        for df in self.cm_deriv_funcs[param_name]:
            d_cm_d_cmparam += df(toas, param_name)

        return DMconst * d_cm_d_cmparam * (bfreq / u.MHz) ** (-alpha)

    def d_delay_d_alphaparam(self, toas, param_name, acc_delay=None):
        """Derivative of delay wrt to DM parameter.

        Parameters
        ----------
        toas : `pint.TOAs` object.
            Input toas.
        param_name : str
            Derivative parameter name
        acc_delay : `astropy.quantity` or `numpy.ndarray`
            Accumulated delay values. This parameter is to keep the unified API,
            but not used in this function.
        """
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]

        param_unit = getattr(self, param_name).units
        d_alpha_d_alphaparam = (
            np.zeros(toas.ntoas) * u.dimensionless_unscaled / param_unit
        )
        cm = self.cm_value(toas)
        alpha = self.alpha_value(toas)

        for df in self.alpha_deriv_funcs[param_name]:
            d_alpha_d_alphaparam += df(toas, param_name)

        return (
            DMconst
            * cm
            * (bfreq / u.MHz) ** (-alpha)
            * np.log((bfreq / u.MHz).to("").value)
            * d_alpha_d_alphaparam
        )
