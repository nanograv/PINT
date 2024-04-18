"""Approximate binary model for small eccentricity."""

import astropy.units as u
import numpy as np
from astropy.time import Time

from loguru import logger as log

from pint.models.parameter import (
    MJDParameter,
    floatParameter,
    intParameter,
    funcParameter,
)
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries import binary_orbits as bo
from pint.models.stand_alone_psr_binaries.ELL1_model import ELL1model
from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
from pint.models.stand_alone_psr_binaries.ELL1k_model import ELL1kmodel
from pint.models.timing_model import MissingParameter
from pint.utils import taylor_horner_deriv
from pint import Tsun


def _eps_to_e(eps1, eps2):
    return np.sqrt(eps1**2 + eps2**2)


def _eps_to_om(eps1, eps2):
    OM = np.arctan2(eps1, eps2)
    if OM < 0:
        OM += 360 * u.deg
    return OM.to(u.deg)


def _epsdot_to_edot(eps1, eps2, eps1dot, eps2dot):
    # Eqn. A14,A15 in Lange et al. inverted
    ecc = np.sqrt(eps1**2 + eps2**2)
    return (eps1dot * eps1 + eps2dot * eps2) / ecc


def _epsdot_to_omdot(eps1, eps2, eps1dot, eps2dot):
    # Eqn. A14,A15 in Lange et al. inverted
    ecc = np.sqrt(eps1**2 + eps2**2)
    return ((eps1dot * eps2 - eps2dot * eps1) / ecc**2).to(
        u.deg / u.yr, equivalencies=u.dimensionless_angles()
    )


def _tasc_to_T0(TASC, PB, eps1, eps2):
    OM = np.arctan2(eps1, eps2)
    if OM < 0:
        OM += 360 * u.deg
    return TASC + ((PB / 2 / np.pi) * OM).to(
        u.d, equivalencies=u.dimensionless_angles()
    )


class BinaryELL1(PulsarBinary):
    """ELL1 binary model.

    This binary model uses a rectangular representation for the eccentricity of an orbit,
    resolving complexities that arise with periastron-based parameters in nearly-circular
    orbits. It also makes certain approximations (up to O(e^3)) that are invalid when the eccentricity
    is "large"; what qualifies as "large" depends on your data quality. A formula exists
    to determine when the approximations this model makes are sufficiently accurate.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.ELL1_model.ELL1model`.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    except that it removes ECC, OM, and T0:

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ell1.BinaryELL1

    References
    ----------
    - Lange et al. (2001), MNRAS, 326 (1), 274–282 [1]_
    - Zhu et al. (2019), MNRAS, 482 (3), 3249-3260 [2]_
    - Fiore et al. (2023), arXiv:2305.13624 [astro-ph.HE] [3]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.3249Z/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..274L/abstract
    .. [3] https://arxiv.org/abs/2305.13624

    Notes
    -----
    This includes o(e^2) expression for Roemer delay from Norbert Wex and Weiwei Zhu
    This is equation (1) of Zhu et al (2019) but with a corrected typo:
        In the first line of that equation, ex->e1 and ey->e2
        In the other lines, ex->e2 and ey->e1
    See Email from NW and WZ to David Nice on 2019-Aug-08
    The dre expression comes from NW and WZ; the derivatives
    were calculated by hand for PINT

    Also includes o(e^3) expression from equation (4) of Fiore et al. (2023)
    (derivatives also calculated by hand)

    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "ELL1"
        self.binary_model_class = ELL1model

        self.add_param(
            MJDParameter(
                name="TASC", description="Epoch of ascending node", time_scale="tdb"
            )
        )

        self.add_param(
            floatParameter(
                name="EPS1",
                units="",
                description="First Laplace-Lagrange parameter, ECC*sin(OM)",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="EPS2",
                units="",
                description="Second Laplace-Lagrange parameter, ECC*cos(OM)",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="EPS1DOT",
                units="1e-12/s",
                description="First derivative of first Laplace-Lagrange parameter",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="EPS2DOT",
                units="1e-12/s",
                description="Second derivative of first Laplace-Lagrange parameter",
                long_double=True,
            )
        )
        self.remove_param("ECC")
        self.remove_param("OM")
        self.remove_param("T0")

        self.add_param(
            funcParameter(
                name="ECC",
                units="",
                aliases=["E"],
                description="Eccentricity",
                params=("EPS1", "EPS2"),
                func=_eps_to_e,
            )
        )
        self.add_param(
            funcParameter(
                name="OM",
                units=u.deg,
                description="Longitude of periastron",
                long_double=True,
                params=("EPS1", "EPS2"),
                func=_eps_to_om,
            )
        )
        self.add_param(
            funcParameter(
                name="EDOT",
                units="1/s",
                description="Eccentricity derivative respect to time",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
                params=("EPS1", "EPS2", "EPS1DOT", "EPS2DOT"),
                func=_epsdot_to_edot,
            )
        )
        self.add_param(
            funcParameter(
                name="OMDOT",
                units="deg/year",
                description="Rate of advance of periastron",
                long_double=True,
                params=("EPS1", "EPS2", "EPS1DOT", "EPS2DOT"),
                func=_epsdot_to_omdot,
            )
        )
        # don't implement T0 yet since that is a MJDparameter at base
        # and our funcParameters don't support that yet
        # self.add_param(
        #     funcParameter(
        #         name="T0",
        #         description="Epoch of periastron passage",
        #         time_scale="tdb",
        #         params=("TASC", "PB", "EPS1", "EPS2"),
        #         func=_tasc_to_T0,
        #     )
        # )

        self.warn_default_params = []

    def validate(self):
        """Validate parameters."""
        super().validate()

        if self.TASC.value is None:
            raise MissingParameter("ELL1", "TASC", "TASC is required for ELL1 model.")
        for p in ["EPS1", "EPS2"]:
            pm = getattr(self, p)
            if pm.value is None:
                pm.value = 0

    def change_binary_epoch(self, new_epoch):
        """Change the epoch for this binary model.

        TASC will be changed to the epoch of the ascending node closest to the
        supplied epoch, and the Laplace parameters (EPS1, EPS2) and projected
        semi-major axis (A1 or X) will be updated according to the specified
        EPS1DOT, EPS2DOT, and A1DOT or XDOT, if present.

        Note that derivatives of binary orbital frequency higher than the first
        (FB2, FB3, etc.) are ignored in computing the new T0, even if present in
        the model. If high-precision results are necessary, especially for models
        containing higher derivatives of orbital frequency, consider re-fitting
        the model to a set of TOAs.

        Parameters
        ----------
        new_epoch: float MJD (in TDB) or `astropy.Time` object
            The new epoch value.
        """
        if isinstance(new_epoch, Time):
            new_epoch = Time(new_epoch, scale="tdb", precision=9)
        else:
            new_epoch = Time(new_epoch, scale="tdb", format="mjd", precision=9)

        # Get PB and PBDOT from model
        # make sure that the PB is the base parameter
        if self.PB.quantity is not None and not isinstance(self.PB, funcParameter):
            PB = self.PB.quantity
            if self.PBDOT.quantity is not None:
                PBDOT = self.PBDOT.quantity
            else:
                PBDOT = 0.0 * u.Unit("")
        else:
            PB = 1.0 / self.FB0.quantity
            try:
                PBDOT = -self.FB1.quantity / self.FB0.quantity**2
            except AttributeError:
                PBDOT = 0.0 * u.Unit("")

        # Find closest periapsis time and reassign T0
        tasc_ld = self.TASC.quantity.tdb.mjd_long
        dt = (new_epoch.tdb.mjd_long - tasc_ld) * u.day
        d_orbits = dt / PB - PBDOT * dt**2 / (2.0 * PB**2)
        n_orbits = np.round(d_orbits.to(u.Unit("")))
        if n_orbits == 0:
            return
        dt_integer_orbits = PB * n_orbits + PB * PBDOT * n_orbits**2 / 2.0
        self.TASC.quantity = self.TASC.quantity + dt_integer_orbits

        if hasattr(self, "FB2") and self.FB2.value is not None:
            log.warning(
                "Ignoring orbital frequency derivatives higher than FB1"
                "in computing new TASC; a model fit should resolve this."
            )

        # Update PB or FB0, FB1, etc.
        if isinstance(self.binary_instance.orbits_cls, bo.OrbitPB):
            dPB = PBDOT * dt_integer_orbits
            self.PB.quantity = self.PB.quantity + dPB
        else:
            fbterms = [0.0 * u.Unit("")] + self._parent.get_prefix_list("FB")

            for n in range(len(fbterms) - 1):
                cur_deriv = getattr(self, f"FB{n}")
                cur_deriv.value = taylor_horner_deriv(
                    dt_integer_orbits.to(u.s), fbterms, deriv_order=n + 1
                )

        # Update EPS1, EPS2, and A1
        if hasattr(self, "EPS1DOT") and self.EPS1DOT.quantity is not None:
            dEPS1 = self.EPS1DOT.quantity * dt_integer_orbits
            self.EPS1.quantity = self.EPS1.quantity + dEPS1
        if hasattr(self, "EPS2DOT") and self.EPS2DOT.quantity is not None:
            dEPS2 = self.EPS2DOT.quantity * dt_integer_orbits
            self.EPS2.quantity = self.EPS2.quantity + dEPS2
        if hasattr(self, "A1DOT") and self.A1DOT.quantity is not None:
            dA1 = self.A1DOT.quantity * dt_integer_orbits
            self.A1.quantity = self.A1.quantity + dA1

        return dt_integer_orbits


class BinaryELL1H(BinaryELL1):
    """ELL1 modified to use H3 parameter for Shapiro delay.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.ELL1H_model.ELL1Hmodel`.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ell1.BinaryELL1H

    Notes
    -----
    When `H3` only is supplied, `NHARMS` is ignored, and the approximate version is used (Eqn. 19) appropriate for medium inclinations.

    When `H3` and `H4` are supplied, `NHARMS` is taken to be `max(7,NHARMS)`, and the approximate version is used (Eqn. 19) appropriate for medium inclinations.
    Note that the default value in `pint` for `NHARMS` is 7, while in `tempo2` it is 4.

    When `H3` and `STIGMA` are supplied, `NHARMS` is ignored since the exact version is used (Eqn. 29) appropriate for very high inclinations.

    References
    ----------
    - Freire & Wex (2010), MNRAS, 409 (1), 199-212 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract
    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "ELL1H"
        self.binary_model_class = ELL1Hmodel

        self.add_param(
            floatParameter(
                name="H3",
                units="second",
                description="Shapiro delay parameter H3 as in Freire and Wex 2010 Eq(20)",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="H4",
                units="second",
                description="Shapiro delay parameter H4 as in Freire and Wex 2010 Eq(21)",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="STIGMA",
                units="",
                description="Shapiro delay parameter STIGMA as in Freire and Wex 2010 Eq(12)",
                long_double=True,
                aliases=["VARSIGMA", "STIG"],
            )
        )
        self.add_param(
            intParameter(
                name="NHARMS",
                units="",
                # value=7,
                description="Number of harmonics for ELL1H shapiro delay.",
            )
        )
        self.remove_param("M2")
        self.remove_param("SINI")

    @property
    def Shapiro_delay_funcs(self):
        return self.binary_instance.ds_func_list

    def setup(self):
        """Parameter setup."""
        super().setup()
        if self.H4.quantity is not None:
            self.binary_instance.fit_params = ["H3", "H4"]
            # If have H4 or STIGMA, choose 7th order harmonics
            if (self.NHARMS.value is not None) and (self.NHARMS.value < 7):
                log.warning(
                    f"Requested NHARMS={self.NHARMS.value}, but setting it to 7 since H4 is also specified"
                )
            self.NHARMS.value = (
                max(self.NHARMS.value, 7) if self.NHARMS.value is not None else 7
            )
            if self.STIGMA.quantity is not None:
                raise ValueError("ELL1H can use H4 or STIGMA but not both")

        if self.STIGMA.quantity is not None:
            self.binary_instance.fit_params = ["H3", "STIGMA"]
            if self.NHARMS.value is not None:
                log.warning(
                    f"Requested NHARMS={self.NHARMS.value} will be ignored, since will use exact parameterization with STIGMA specified"
                )
            self.binary_instance.ds_func = self.binary_instance.delayS_H3_STIGMA_exact
            if self.STIGMA.quantity <= 0:
                raise ValueError("STIGMA must be greater than zero.")
        self.update_binary_object(None)

    def validate(self):
        """Parameter validation."""
        super().validate()
        # if self.H3.quantity is None:
        #     raise MissingParameter("ELL1H", "H3", "'H3' is required for ELL1H model")


class BinaryELL1k(BinaryELL1):
    """ELL1k binary model.

    Modified version of the ELL1 model applicable to short-orbital period binaries where
    the periastron advance timescale is comparable to the data span. In such cases, the
    evolution of EPS1 and EPS2 should be described using OMDOT and LNEDOT rather than
    EPS1DOT and EPS2DOT. The (EPS1DOT, EPS2DOT) parametrization of the evolution of EPS1
    and EPS2 is a linear approximation of the (OMDOT, LNEDOT) parametrization which breaks
    down when the periastron advance timescale is comparable to the data span.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.ELL1k_model.ELL1kmodel`.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    except that it removes ECC, OM, and T0:

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ell1.BinaryELL1k

    References
    ----------
    - Susobhanan et al. (2018), MNRAS, 480 (4), 5260-5271 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.5260S/abstract
    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "ELL1k"
        self.binary_model_class = ELL1kmodel

        self.remove_param("OMDOT")
        self.remove_param("EDOT")
        self.remove_param("EPS1DOT")
        self.remove_param("EPS2DOT")

        self.add_param(
            floatParameter(
                name="OMDOT",
                units="deg/year",
                description="Rate of advance of periastron",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="LNEDOT",
                units="1/year",
                description="Log-derivative of the eccentricity EDOT/ECC",
                long_double=True,
            )
        )

    def validate(self):
        """Validate parameters."""
        super().validate()

    def change_binary_epoch(self, new_epoch):
        """Change the epoch for this binary model.

        EPS1 and EPS2 will be evolved in time according to OMDOT and LNEDOT.
        Everything else is the same as in the ELL1 model.

        Parameters
        ----------
        new_epoch: float MJD (in TDB) or `astropy.Time` object
            The new epoch value.
        """
        dt = super().change_binary_epoch(new_epoch)

        # Update EPS1, EPS2
        if self.OMDOT.quantity is not None and self.LNEDOT.quantity is not None:
            eps10 = self.EPS1.quantity
            eps20 = self.EPS1.quantity
            omdot = self.OMDOT.quantity
            lnedot = self.LNEDOT.quantity

            self.EPS1.quantity = (1 + lnedot * dt) * (
                eps10 * np.cos(omdot * dt) + eps20 * np.sin(omdot * dt)
            )
            self.EPS2.quantity = (1 + lnedot * dt) * (
                eps20 * np.cos(omdot * dt) - eps10 * np.sin(omdot * dt)
            )
