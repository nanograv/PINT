"""The DDS model - Damour and Deruelle with alternate Shapiro delay parametrization."""
import astropy.constants as c
import astropy.units as u
import numpy as np
from loguru import logger as log

from pint import Tsun

from .DD_model import DDmodel


class DDSmodel(DDmodel):
    """Damour and Deruelle model with alternate Shapiro delay parameterization.

    This extends the :class:`pint.models.binary_dd.BinaryDD` model with
    :math:`SHAPMAX = -\log(1-s)` instead of just :math:`s=\sin i`, which behaves better
    for :math:`\sin i` near 1.  It does not (yet) implement the higher-order delays and lensing correction.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

       SHAPMAX
            :math:`-\log(1-\sin i)`

    It also removes:

       SINI
            use ``SHAPMAX`` instead

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDS

    References
    ----------
    - Kramer et al. (2006), Science, 314, 97 [klm+06]_
    - Rafikov and Lai (2006), PRD, 73, 063003 [rl06]_

    .. [klm+06] https://ui.adsabs.harvard.edu/abs/2006Sci...314...97K/abstract
    .. [rl06] https://ui.adsabs.harvard.edu/abs/2006PhRvD..73f3003R/abstract
    """

    def __init__(self, t=None, input_params=None):
        super().__init__()
        self.binary_name = "DDS"
        self.param_default_value.update(
            {
                "SHAPMAX": 0,
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
    def SINI(self):
        return 1 - np.exp(-self.SHAPMAX)

    @SINI.setter
    def SINI(self, val):
        log.debug(
            "DDS model uses SHAPMAX as inclination parameter. SINI will not be used."
        )

    def d_SINI_d_SHAPMAX(self):
        return np.exp(-self.SHAPMAX)

    def d_SINI_d_par(self, par):
        par_obj = getattr(self, par)
        try:
            ko_func = getattr(self, f"d_SINI_d_{par}")
        except AttributeError:
            ko_func = lambda: np.zeros(len(self.tt0)) * u.Unit("") / par_obj.unit
        return ko_func()
