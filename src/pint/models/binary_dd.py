"""Damour and Deruelle binary model."""
import numpy as np
from astropy import units as u, constants as c

from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.DD_model import DDmodel
from pint.models.stand_alone_psr_binaries.DDS_model import DDSmodel
from pint.models.stand_alone_psr_binaries.DDGR_model import DDGRmodel
import pint.derived_quantities


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
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

        if hasattr(self, "GAMMA") and self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("ECC", "OM", "OMDOT", "EDOT"):
                if hasattr(self, p):
                    getattr(self, p).set("0")
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

    It also removes:

       SINI
            use ``SHAPMAX`` instead

    The inferred value of this can be accessed via the ``.sini`` property

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

    def validate(self):
        """Validate parameters."""
        super().validate()
        if (
            hasattr(self, "SHAPMAX")
            and self.SHAPMAX.value is not None
            and not self.SHAPMAX.value > -np.log(2)
        ):
            raise ValueError(f"SHAPMAX must be > -log(2) ({self.SHAPMAX.quantity})")

    @property
    def sini(self):
        return 1 - np.exp(-self.SHAPMAX.quantity)


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

    It also reads but ignores:

        SINI
        PBDOT
        OMDOT
        GAMMA
        DR
        DTH

    The values of these at any time can be obtained via
    ``.sini``, ``.pbdot``, ``.omdot``, ``.gamma``, ``.dr``, and ``.dth`` properties

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_dd.BinaryDDGR

    References
    ----------
    - Taylor and Weisberg (1989), ApJ, 345, 434 [1]_

    .. [1] https://ui.adsabs.harvard.edu/abs/1989ApJ...345..434T/abstract
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

    def setup(self):
        """Parameter setup."""
        super().setup()
        # don't remove these from the definition so they can be read
        # but remove them on setup so they will be ignored
        for p in ["OMDOT", "PBDOT", "GAMMA", "SINI", "DR", "DTH"]:
            super().remove_param(p)

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
        # for p in ["OMDOT", "PBDOT", "GAMMA", "SINI", "DR", "DTH"]:
        #     if hasattr(self, p) and not getattr(self, p).frozen:
        #         raise ValueError(f"PK parameter {p} cannot be unfrozen in DDGR model")

    def print_par(self, format="pint"):
        if self._parent is not None:
            # Check if the binary name are the same as BINARY parameter
            if self._parent.BINARY.value != self.binary_model_name:
                raise TimingModelError(
                    f"Parameter BINARY {self._parent.BINARY.value}"
                    f" does not match the binary"
                    f" component {self.binary_model_name}"
                )
            result = self._parent.BINARY.as_parfile_line(format=format)
        else:
            result = "BINARY {0}\n".format(self.binary_model_name)
        for p in self.params:
            par = getattr(self, p)
            if par.quantity is not None:
                result += par.as_parfile_line(format=format)
        # we will introduce the PK parameters here just for printing
        result += floatParameter(
            name="OMDOT",
            value=self.omdot.value,
            units="deg/year",
            description="Longitude of periastron",
            long_double=True,
        ).as_parfile_line(format=format)
        result += floatParameter(
            name="SINI",
            units="",
            description="Sine of inclination angle",
            value=self.sini.value,
        ).as_parfile_line(format=format)
        result += floatParameter(
            name="PBDOT",
            value=self.pbdot.value,
            units=u.day / u.day,
            description="Orbital period derivative respect to time",
            unit_scale=True,
            scale_factor=1e-12,
            scale_threshold=1e-7,
        ).as_parfile_line(format=format)
        result += floatParameter(
            name="GAMMA",
            value=self.gamma.value,
            units="second",
            description="Time dilation & gravitational redshift",
        ).as_parfile_line(format=format)
        result += floatParameter(
            name="DR",
            value=self.dr.value,
            units="",
            description="Relativistic deformation of the orbit",
        ).as_parfile_line(format=format)
        result += floatParameter(
            name="DTH",
            value=self.dth.value,
            units="",
            description="Relativistic deformation of the orbit",
        ).as_parfile_line(format=format)
        return result

    @property
    def pbdot(self):
        return pint.derived_quantities.pbdot(
            self.MTOT.quantity - self.M2.quantity,
            self.M2.quantity,
            self.PB.quantity,
            self.ECC.quantity,
        )

    @property
    def omdot(self):
        return pint.derived_quantities.omdot(
            self.MTOT.quantity - self.M2.quantity,
            self.M2.quantity,
            self.PB.quantity,
            self.ECC.quantity,
        )

    @property
    def gamma(self):
        return pint.derived_quantities.gamma(
            self.MTOT.quantity - self.M2.quantity,
            self.M2.quantity,
            self.PB.quantity,
            self.ECC.quantity,
        )

    @property
    def dr(self):
        return pint.derived_quantities.dr(
            self.MTOT.quantity - self.M2.quantity, self.M2.quantity, self.PB.quantity
        )

    @property
    def dth(self):
        return pint.derived_quantities.dth(
            self.MTOT.quantity - self.M2.quantity, self.M2.quantity, self.PB.quantity
        )

    @property
    def sini(self):
        return pint.derived_quantities.sini(
            self.MTOT.quantity - self.M2.quantity,
            self.M2.quantity,
            self.PB.quantity,
            self.A1.quantity,
        )
