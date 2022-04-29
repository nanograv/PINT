"""The BT (Blandford & Teukolsky) model."""
import astropy.units as u

from pint import GMsun, Tsun, ls
from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.BT_model import BTmodel
from pint.models.timing_model import MissingParameter, TimingModel


class BinaryBT(PulsarBinary):
    """Blandford and Teukolsky binary model.

    This binary model is described in Blandford and Teukolshy 1976. It is
    a relatively simple parametrized post-Keplerian model that does not
    support Shapiro delay calculations.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.BT_model.BTmodel`.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_bt.BinaryBT

    Notes
    -----
    Because PINT's binary models all support specification of multiple orbital
    frequency derivatives FBn, this is capable of behaving like the model called
    BTX in tempo2. The model called BTX in tempo instead supports multiple
    (non-interacting) companions, and that is not supported here. Neither can
    PINT accept "BTX" as an alias for this model.

    See Blandford & Teukolsky 1976, ApJ, 205, 580.
    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "BT"
        self.binary_model_class = BTmodel

        self.add_param(
            floatParameter(
                name="GAMMA",
                value=0.0,
                units="second",
                description="Time dilation & gravitational redshift",
            )
        )
        self.remove_param("M2")
        self.remove_param("SINI")

    def validate(self):
        """Validate BT model parameters"""
        super().validate()
        for p in ("T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p, "%s is required for BT" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).value = "0"
                getattr(self, p).frozen = True

        if self.GAMMA.value is None:
            self.GAMMA.value = "0"
            self.GAMMA.frozen = True
