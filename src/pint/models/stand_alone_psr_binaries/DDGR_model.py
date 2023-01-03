"""The DDGR model - Damour and Deruelle with GR assumed"""
import astropy.constants as c
import astropy.units as u
import numpy as np
import warnings

from pint import Tsun
from pint import derived_quantities

from .DD_model import DDmodel


@u.quantity_input(M1=u.Msun, M2=u.Msun, n=1 / u.d)
def _solve_kepler(M1, M2, n, ARTOL=1e-10):
    """Relativistic version of Kepler's third law, solved by iteration

    Taylor & Weisberg (1989), Eqn. 15
    In tempo, implemented as ``mass2dd`` (https://sourceforge.net/p/tempo/tempo/ci/master/tree/src/mass2dd.f)


    Parameters
    ----------
    M1 : u.Quantity
        Mass of pulsar
    M2 : u.Quantity
        Mass of companion
    n : u.Quantity
        orbital angular frequency
    ARTOL : float
        fractional tolerance for solution

    Returns
    -------
    arr0 : u.Quantity
        non-relativistic semi-major axis
    arr : u.Quantity
        relativstic semi-major axis
    """
    MTOT = M1 + M2
    # initial NR value
    arr0 = (c.G * MTOT / n**2) ** (1.0 / 3)
    arr = arr0
    arr_old = arr
    arr = arr0 * (
        1 + (M1 * M2 / MTOT**2 - 9) * (c.G * MTOT / (2 * arr * c.c**2))
    ) ** (2.0 / 3)
    # iterate to get correct value
    while np.fabs((arr - arr_old) / arr) > ARTOL:
        arr_old = arr
        ar = arr0 * (
            1 + (M1 * M2 / MTOT**2 - 9) * (c.G * MTOT / (2 * arr * c.c**2))
        ) ** (2.0 / 3)

    return arr0, arr


class DDGRmodel(DDmodel):
    """Damour and Deruelle model assuming GR to be correct

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

        MTOT
            Total mass
        XPBDOT
            Excess PBDOT beyond what GR predicts
        XOMDOT
            Excess OMDOT beyond what GR predicts

    It also removes:

        SINI
        PBDOT
        OMDOT
        GAMMA

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDS

    References
    ----------
    - Taylor and Weisberg (1989), ApJ, 345, 434 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/1989ApJ...345..434T/abstract
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
        """Update measurable quantities from system parameters for DDGR model

        Taylor & Weisberg (1989), Eqn. 15-25
        In tempo, implemented as ``mass2dd`` (https://sourceforge.net/p/tempo/tempo/ci/master/tree/src/mass2dd.f)

        """
        PB = self.pb()
        PB = PB.to("second")
        self._M1 = self.MTOT - self.M2
        self._n = 2 * np.pi / PB
        arr0, arr = _solve_kepler(self._M1, self.M2, self._n)
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
            ko_func = getattr(self, f"d_SINI_d_{par}")
        except AttributeError:
            ko_func = lambda: np.zeros(len(self.tt0)) * u.Unit("") / par_obj.unit
        return ko_func()
