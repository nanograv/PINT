from __future__ import absolute_import, division, print_function

from warnings import warn

from pint.models.parameter import MJDParameter, floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.ELL1_model import ELL1model
from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
from pint.models.timing_model import MissingParameter


class BinaryELL1Base(PulsarBinary):
    """PINT wrapper around standalone ELL1 model.

    This is a PINT pulsar binary ELL1 model class a subclass of PulsarBinary.
    It is a wrapper for stand alone ELL1model class defined in ./pulsar_binary/ELL1_model.py
    All the detailed calculations are in the stand alone ELL1model.
    The aim for this class is to connect the stand alone binary model with PINT platform

    ELL1model special parameters:

        - TASC Epoch of ascending node
        - EPS1 First Laplace-Lagrange parameter, ECC x sin(OM) for ELL1 model
        - EPS2 Second Laplace-Lagrange parameter, ECC x cos(OM) for ELL1 model
        - EPS1DOT First derivative of first Laplace-Lagrange parameter
        - EPS2DOT Second derivative of second Laplace-Lagrange parameter

    """

    def __init__(self):
        super(BinaryELL1Base, self).__init__()
        self.add_param(
            MJDParameter(
                name="TASC", description="Epoch of ascending node", time_scale="tdb"
            )
        )

        self.add_param(
            floatParameter(
                name="EPS1",
                units="",
                description="First Laplace-Lagrange parameter, ECC x sin(OM) for ELL1 model",
                long_double=True,
            )
        )

        self.add_param(
            floatParameter(
                name="EPS2",
                units="",
                description="Second Laplace-Lagrange parameter, ECC x cos(OM) for ELL1 model",
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

        self.warn_default_params = []

    def setup(self):
        super(BinaryELL1Base, self).setup()

    def validate(self):
        """ Validate parameters
        """
        super(BinaryELL1Base, self).validate()

        for p in ["EPS1", "EPS2"]:
            if getattr(self, p).value is None:
                raise MissingParameter("ELL1", p, p + " is required for ELL1 model.")
        # Check TASC
        if self.TASC.value is None:
            if self.ECC.value == 0.0:
                warn("Since ECC is 0.0, using T0 as TASC.")
                if self.T0.value is not None:
                    self.TASC.value = self.T0.value
                else:
                    raise MissingParameter(
                        "ELL1", "T0", "T0 or TASC is required for ELL1 model."
                    )
            else:
                raise MissingParameter(
                    "ELL1", "TASC", "TASC is required for ELL1 model."
                )


class BinaryELL1(BinaryELL1Base):
    register = True

    def __init__(self):
        super(BinaryELL1, self).__init__()
        self.binary_model_name = "ELL1"
        self.binary_model_class = ELL1model

    def setup(self):
        super(BinaryELL1, self).setup()

    def validate(self):
        super(BinaryELL1, self).validate()


class BinaryELL1H(BinaryELL1Base):
    """Modified ELL1 to with H3 parameter.

    This is modified version of ELL1 model. a new parameter H3 is
    introduced to model the shapiro delay.

    Note
    ----
    Ref:  Freire and Wex 2010; Only the Medium-inclination case model is implemented.

    """

    register = True

    def __init__(self):
        super(BinaryELL1H, self).__init__()
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
            )
        )
        self.add_param(
            floatParameter(
                name="NHARMS",
                units="",
                value=3,
                description="Number of harmonics for ELL1H shapiro delay.",
            )
        )

    @property
    def Shapiro_delay_funcs(self):
        return self.binary_instance.ds_func_list

    def setup(self):
        """ Parameter setup.
        """
        super(BinaryELL1H, self).setup()

    def validate(self):
        """ Parameter validation.
        """
        super(BinaryELL1H, self).validate()
        if self.H3.quantity is None:
            raise MissingParameter("ELL1H", "H3", "'H3' is required for ELL1H model")
        if self.SINI.quantity is not None:
            warn("'SINI' will not be used in ELL1H model. ")
        if self.M2.quantity is not None:
            warn("'M2' will not be used in ELL1H model. ")
        if self.H4.quantity is not None:
            self.binary_instance.fit_params = ["H3", "H4"]
            # If have H4 or STIGMA, choose 7th order harmonics
            if self.NHARMS.value < 7:
                self.NHARMS.value = 7

        if self.STIGMA.quantity is not None:
            self.binary_instance.fit_params = ["H3", "STIGMA"]
            self.binary_instance.ds_func = self.binary_instance.delayS_H3_STIGMA_exact
