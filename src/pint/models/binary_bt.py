"""The BT (Blandford & Teukolsky) model.

See Blandford & Teukolsky 1976, ApJ, 205, 580.

"""
from __future__ import absolute_import, division, print_function

import astropy.units as u

from pint import GMsun, Tsun, ls
from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.BT_model import BTmodel
from pint.models.timing_model import MissingParameter, TimingModel


class BinaryBT(PulsarBinary):
    """Model implemenring the BT model.

    This is a PINT pulsar binary BT model class a subclass of PulsarBinary.
    It is a wrapper for stand alone BTmodel class defined in
    ./stand_alone_psr_binary/BT_model.py
    All the detailed calculations are in the stand alone BTmodel.
    The aim for this class is to connect the stand alone binary model with PINT platform
    BTmodel special parameters:
    GAMMA Binary Einsten delay coeeficient
    """

    register = True

    def __init__(self):
        super(BinaryBT, self).__init__()
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
        # remove unused parameter.
        self.remove_param("M2")
        self.remove_param("SINI")

    def setup(self):
        super(BinaryBT, self).setup()

    def validate(self):
        """ Validate BT model parameters
        """
        super(BinaryBT, self).validate()
        for p in ("T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p, "%s is required for BT" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("BT", "T0", "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True
