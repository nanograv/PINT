"""Damour and Deruelle binary model."""
from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.DD_model import DDmodel
from pint.models.timing_model import MissingParameter


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
        super(BinaryDD, self).__init__()
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
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("ECC", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True
