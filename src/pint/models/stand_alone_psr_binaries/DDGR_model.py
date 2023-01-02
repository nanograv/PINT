"""The DDS model - Damour and Deruelle with alternate Shapiro delay paramterization."""
import astropy.constants as c
import astropy.units as u
import numpy as np
import warnings

from pint import Tsun
from pint import derived_quantities

from .DD_model import DDmodel


class DDGRmodel(DDmodel):
    """Damour and Deruelle model with alternate Shapiro delay parameterization.

    This extends the :class:`pint.models.binary_dd.BinaryDD` model with
    :math:`SHAPMAX = -\log(1-s)` instead of just :math:`s=\sin i`, which behaves better
    for :math:`\sin i` near 1.

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
    - Kramer et al. (2006), Science, 314, 97 [1]_
    - Rafikov and Lai (2006), PRD, 73, 063003 [2]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2006Sci...314...97K/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2006PhRvD..73f3003R/abstract
    """

    def __init__(self, t=None, input_params=None):
        super().__init__()
        self.binary_name = "DDS"
        self.param_default_value.update(
            {"MTOT": 2.8 * u.Msun, "XOMDOT": 0, "XPBDOT": 0}
        )

        # If any parameter has aliases, it should be updated
        # self.param_aliases.update({})
        self.binary_params = list(self.param_default_value.keys())
        # Remove unused parameter SINI
        for p in ["SINI", "PBDOT", "OMDOT", "GAMMA"]:
            del self.param_default_value[p]
        self.set_param_values()
        if input_params is not None:
            self.update_input(param_dict=input_params)

    def _update(self, ARTOL=1e-10):
        PB = self.pb()
        PB = PB.to("second")
        self._M1 = self.MTOT - self.M2
        self._n = 2 * np.pi / PB
        # initial value of semi-major axis before relativstic corrections
        arr0 = (c.G * self.MTOT / self._n**2) ** (1.0 / 3)
        arr = arr0
        arr_old = arr
        arr = arr0 * (
            1
            + (self._M1 * self.M2 / self.MTOT**2 - 9)
            * (c.G * self.Mtot / (2 * arr * c.c**2))
        ) ** (2.0 / 3)
        # iterate to get correct value
        while np.fabs((arr - arr_old) / arr) > ARTOL:
            arr_old = arr
            ar = arr0 * (
                1
                + (self._M1 * self.M2 / self.MTOT**2 - 9)
                * (c.G * self.Mtot / (2 * arr * c.c**2))
            ) ** (2.0 / 3)
        self._arr = arr
        # pulsar component
        self._ar = self._arr * (self.M2 / self.MTOT)
        self._SINI = (self.A1 / self._ar).decompose()
        self._GAMMA = (
            self.ecc()
            * c.G
            * self.M2
            * (self._M1 + 2 * self.M2)
            / (self._n * arr0 * self.MTOT)
        ).to(u.s)
        fe = (1 + (73.0 / 24) * self.ecc() ** 2 + (37.0 / 96) ** self.ecc() ** 4) * (
            1 - self.ecc() ** 2
        ) ** (-7.0 / 2)
        self._PBDOT = (
            (-192 * np.pi / (5 * c.c**5))
            * (c.G / self._n) ** (5.0 / 3)
            * self._M1
            * self.M2
            * self.MTOT ** (-1.0 / 3)
            * fe
        ).decompose()
        self._k = (3 * c.G * self.MTOT) / (arr0 * (1 - self.ecc() ** 2))
        self._DR = (c.G / (c.c**2 * self.MTOT * self._arr)) * (
            3 * self._M1**2 + 6 * self._M1 * self.M2 + 2 * self.M2**2
        )
        self._er = self.ecc() * (1 + self._DR)
        self._DTH = (c.G / (c.c**2 * self.MTOT * self._arr)) * (
            3.5 * self._M1**2 + 6 * self._M1 * self.M2 + 2 * self.M2**2
        )
        self._eth = self.ecc() * (1 + self._DTH)

    @property
    def k(self):
        self._update()
        return self._k

    @property
    def SINI(self):
        self._update()
        return self._SINI

    @property
    def GAMMA(self):
        self._update()
        return self._GAMMA

    @property
    def PBDOT(self):
        self._update()
        return self._PBDOT + self.XPBDOT

    @property
    def OMDOT(self):
        return self.XOMDOT

    @property
    def DR(self):
        self._update()
        return self._DR

    @property
    def DTH(self):
        self._update()
        return self._DTH

    def er(self):
        self._update()
        return self._er

    def eTheta(self):
        self._update()
        return self._eTheta

    def d_SINI_d_SHAPMAX(self):
        return np.exp(-self.SHAPMAX)

    def d_SINI_d_par(self, par):
        par_obj = getattr(self, par)
        try:
            ko_func = getattr(self, "d_SINI_d_" + par)
        except AttributeError:
            ko_func = lambda: np.zeros(len(self.tt0)) * u.Unit("") / par_obj.unit
        return ko_func()
