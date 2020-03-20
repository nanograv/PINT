from __future__ import absolute_import, division, print_function

from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.DD_model import DDmodel
from pint.models.timing_model import MissingParameter


class BinaryDD(PulsarBinary):
    """This is a PINT pulsar binary dd model class a subclass of PulsarBinary.
    It is a wrapper for independent DDmodel class defined in
    ./stand_alone_psr_binary/DD_model.py
    All the detailed calculations are in the independent DDmodel.
    The aim for this class is to connect the independent binary model with PINT platform
    DDmodel special parameters:
    A0 Aberration
    B0 Aberration
    GAMMA Binary Einsten delay coeeficient
    DR Relativistic deformation of the orbit
    DTH Relativistic deformation of the orbit
    """

    register = True

    def __init__(self,):
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

    def setup(self):
        """setup.
        """
        super(BinaryDD, self).setup()

    def validate(self):
        """ Validate the input parameters.
        """
        super(BinaryDD, self).validate()
        self.check_required_params(["T0", "A1"])
        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True
            # TODO This steps seems duplicated.
            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("DD", "T0", "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("ECC", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True
