"""Damour and Deruelle binary model."""
import numpy as np
from astropy import units as u, constants as c

from pint import Tsun
from pint.models.parameter import floatParameter, funcParameter, intParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.DD_model import DDmodel
from pint.models.stand_alone_psr_binaries.DDS_model import DDSmodel
from pint.models.stand_alone_psr_binaries.DDGR_model import DDGRmodel
from pint.models.stand_alone_psr_binaries.DDH_model import DDHmodel
import pint.derived_quantities


# these would be doable with lambda functions
# but then the instances would not pickle
def _sini_from_shapmax(SHAPMAX):
    return 1 - np.exp(-SHAPMAX)


def _mp_from_mtot(MTOT, M2):
    return MTOT - M2


def _m2_from_h3_stigma(H3, STIGMA):
    return (H3 / Tsun / STIGMA**3) * u.Msun


def _sini_from_stigma(STIGMA):
    return 2 * STIGMA / (1 + STIGMA**2)


class BinaryDD(PulsarBinary):
    """Damour and Deruelle binary model.

    This binary model is described in
    `Damour and Deruelle 1986 <https://ui.adsabs.harvard.edu/abs/1986AIHS...44..263D/abstract>`_
    It is a parametrized post-Keplerian model that supports additional
    effects and post-Keplerian parameters compared to :class:`pint.models.binary_bt.BinaryBT`.
    It does not assume General Relativity in order to infer
    values for any post-Keplerian parameters.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DD_model.DDmodel`.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDD
    """

    register = True

    def __init__(
        self,
    ):
        super().__init__()
        self.binary_model_name = "DD"
        self.binary_model_class = DDmodel
        self.add_param(
            floatParameter(
                name="A0",
                value=0.0,
                units="s",
                description="DD model aberration parameter A0",
            )
        )

        self.add_param(
            floatParameter(
                name="B0",
                value=0.0,
                units="s",
                description="DD model aberration parameter B0",
            )
        )

        self.add_param(
            floatParameter(
                name="GAMMA",
                value=0.0,
                units="second",
                description="Time dilation & gravitational redshift",
            )
        )

        self.add_param(
            floatParameter(
                name="DR",
                value=0.0,
                units="",
                description="Relativistic deformation of the orbit",
            )
        )

        self.add_param(
            floatParameter(
                name="DTH",
                aliases=["DTHETA"],
                value=0.0,
                units="",
                description="Relativistic deformation of the orbit",
            )
        )

    def validate(self):
        """Validate the input parameters."""
        super().validate()
        self.check_required_params(["T0", "A1"])
        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if hasattr(self, p) and getattr(self, p).value is None:
                getattr(self, p).value = 0.0
                getattr(self, p).frozen = True

        if hasattr(self, "GAMMA") and self.GAMMA.value is None:
            self.GAMMA.value = 0.0
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("ECC", "OM", "OMDOT", "EDOT"):
                if hasattr(self, p):
                    getattr(self, p).value = 0.0
                    getattr(self, p).frozen = True


class BinaryDDS(BinaryDD):
    """Damour and Deruelle model with alternate Shapiro delay parameterization.

    This extends the :class:`pint.models.binary_dd.BinaryDD` model with
    :math:`SHAPMAX = -\log(1-s)` instead of just :math:`s=\sin i`, which behaves better
    for :math:`\sin i` near 1.  It does not (yet) implement the higher-order delays and lensing correction.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DDS_model.DDSmodel`.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

       SHAPMAX
            :math:`-\log(1-\sin i)`

    It also converts:

       SINI
            into a read-only parameter

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDS

    References
    ----------
    - Kramer et al. (2006), Science, 314, 97 [ksm+2006]_
    - Rafikov and Lai (2006), PRD, 73, 063003 [rl2006]_

    .. [ksm+2006] https://ui.adsabs.harvard.edu/abs/2006Sci...314...97K/abstract
    .. [rl2006] https://ui.adsabs.harvard.edu/abs/2006PhRvD..73f3003R/abstract

    """

    register = True

    def __init__(
        self,
    ):
        super().__init__()
        self.binary_model_name = "DDS"
        self.binary_model_class = DDSmodel

        self.add_param(
            floatParameter(
                name="SHAPMAX", value=0.0, description="Function of inclination angle"
            )
        )
        self.remove_param("SINI")
        self.add_param(
            funcParameter(
                name="SINI",
                units="",
                description="Sine of inclination angle",
                params=("SHAPMAX",),
                func=_sini_from_shapmax,
            )
        )

    def validate(self):
        """Validate parameters."""
        super().validate()
        if (
            hasattr(self, "SHAPMAX")
            and self.SHAPMAX.value is not None
            and not self.SHAPMAX.value > -np.log(2)
        ):
            raise ValueError(f"SHAPMAX must be > -log(2) ({self.SHAPMAX.quantity})")


class BinaryDDGR(BinaryDD):
    """Damour and Deruelle model assuming GR to be correct

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

        MTOT
            Total mass
        XPBDOT
            Excess PBDOT beyond what GR predicts
        XOMDOT
            Excess OMDOT beyond what GR predicts

    It also reads but converts:

        SINI
        PBDOT
        OMDOT
        GAMMA
        DR
        DTH

            into read-only parameters

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DDGR_model.DDGRmodel`.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDGR

    References
    ----------
    - Taylor and Weisberg (1989), ApJ, 345, 434 [tw89]_

    .. [tw89] https://ui.adsabs.harvard.edu/abs/1989ApJ...345..434T/abstract
    """

    register = True

    def __init__(
        self,
    ):
        super().__init__()
        self.binary_model_name = "DDGR"
        self.binary_model_class = DDGRmodel

        self.add_param(
            floatParameter(
                name="MTOT",
                units=u.M_sun,
                description="Total system mass in units of Solar mass",
            )
        )
        self.add_param(
            floatParameter(
                name="XOMDOT",
                units="deg/year",
                description="Excess longitude of periastron advance compared to GR",
                long_double=True,
            )
        )
        self.add_param(
            floatParameter(
                name="XPBDOT",
                units=u.day / u.day,
                description="Excess Orbital period derivative respect to time compared to GR",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
            )
        )
        for p in ["OMDOT", "PBDOT", "GAMMA", "SINI", "DR", "DTH"]:
            self.remove_param(p)

        self.add_param(
            funcParameter(
                name="MP",
                units=u.M_sun,
                description="Pulsar mass",
                params=("MTOT", "M2"),
                func=_mp_from_mtot,
            )
        )
        self.add_param(
            funcParameter(
                name="OMDOT",
                units="deg/year",
                description="Rate of advance of periastron",
                long_double=True,
                params=("MP", "M2", "PB", "ECC"),
                func=pint.derived_quantities.omdot,
            )
        )
        self.add_param(
            funcParameter(
                name="SINI",
                units="",
                description="Sine of inclination angle",
                params=("MP", "M2", "PB", "A1"),
                func=pint.derived_quantities.sini,
            )
        )
        self.add_param(
            funcParameter(
                name="PBDOT",
                units=u.day / u.day,
                description="Orbital period derivative respect to time",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
                params=("MP", "M2", "PB", "ECC"),
                func=pint.derived_quantities.pbdot,
            )
        )
        self.add_param(
            funcParameter(
                name="GAMMA",
                units="second",
                description="Time dilation & gravitational redshift",
                params=("MP", "M2", "PB", "ECC"),
                func=pint.derived_quantities.gamma,
            )
        )
        self.add_param(
            funcParameter(
                name="DR",
                units="",
                description="Relativistic deformation of the orbit",
                params=("MP", "M2", "PB"),
                func=pint.derived_quantities.dr,
            )
        )
        self.add_param(
            funcParameter(
                name="DTH",
                aliases=["DTHETA"],
                units="",
                description="Relativistic deformation of the orbit",
                params=("MP", "M2", "PB"),
                func=pint.derived_quantities.dth,
            )
        )

    def setup(self):
        """Parameter setup."""
        super().setup()

    def validate(self):
        """Validate parameters."""
        super().validate()
        aR = (c.G * self.MTOT.quantity * self.PB.quantity**2 / 4 / np.pi**2) ** (
            1.0 / 3
        )
        sini = (
            self.A1.quantity * self.MTOT.quantity / aR / self.M2.quantity
        ).decompose()
        if sini > 1:
            raise ValueError(
                f"Inferred SINI must be <= 1 for DDGR model (MTOT={self.MTOT.quantity}, PB={self.PB.quantity}, A1={self.A1.quantity}, M2={self.M2.quantity} imply SINI={sini})"
            )

    def update_binary_object(self, toas, acc_delay=None):
        super().update_binary_object(toas, acc_delay)
        self.binary_instance._updatePK()


class BinaryDDH(BinaryDD):
    """DD modified to use H3/STIGMA parameter for Shapiro delay.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DDH_model.DDHmodel`.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDH

    Notes
    -----
    This uses the full expression for the Shapiro delay, not the harmonic
    decomposition used in :class:`pint.models.stand_alone_psr_binaries.ELL1H_model.ELL1Hmodel`.

    References
    ----------
    - Freire & Wex (2010), MNRAS, 409 (1), 199-212 [1]_
    - Weisberg & Huang (2016), ApH, 829 (1), 55 [2]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2016ApJ...829...55W/abstract
    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "DDH"
        self.binary_model_class = DDHmodel

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
                name="STIGMA",
                units="",
                description="Shapiro delay parameter STIGMA as in Freire and Wex 2010 Eq(12)",
                long_double=True,
                aliases=["VARSIGMA", "STIG"],
            )
        )
        self.remove_param("M2")
        self.remove_param("SINI")
        self.add_param(
            funcParameter(
                name="SINI",
                units="",
                description="Sine of inclination angle",
                params=("STIGMA",),
                func=_sini_from_stigma,
            )
        )
        self.add_param(
            funcParameter(
                name="M2",
                units=u.Msun,
                description="Companion mass",
                params=("H3", "STIGMA"),
                func=_m2_from_h3_stigma,
            )
        )

    def setup(self):
        """Parameter setup."""
        super().setup()

        self.update_binary_object(None)

    def validate(self):
        """Parameter validation."""
        super().validate()
        # if self.H3.quantity is None:
        #     raise MissingParameter("ELL1H", "H3", "'H3' is required for ELL1H model")
