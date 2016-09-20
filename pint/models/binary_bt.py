
"""This model provides the BT (Blandford & Teukolsky 1976, ApJ, 205, 580) model.
    """
from pint import ls,GMsun,Tsun
from .stand_alone_psr_binaries.BT_model import BTmodel
from .pulsar_binary import PulsarBinary
import parameter as p
from .timing_model import Cache, TimingModel, MissingParameter
import astropy.units as u

class BinaryBT(PulsarBinary):
    """This is a PINT pulsar binary BT model class a subclass of PulsarBinary.
    It is a wrapper for stand alone BTmodel class defined in
    ./stand_alone_psr_binary/BT_model.py
    All the detailed calculations are in the stand alone BTmodel.
    The aim for this class is to connect the stand alone binary model with PINT platform
    BTmodel special parameters:
    GAMMA Binary Einsten delay coeeficient
    """
    def __init__(self):
        super(BinaryBT, self).__init__()
        self.binary_model_name = 'BT'
        self.binary_model_class = BTmodel

        self.add_param(p.floatParameter(name="GAMMA", value=0.0,
             units="second",
             description="Time dilation & gravitational redshift"),
             binary_param = True)

    def setup(self):
        super(BinaryBT, self).setup()
        # If any necessary parameter is missing, raise MissingParameter.
        # This will probably be updated after ELL1 model is added.
        for p in ("PB", "T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p,
                                       "%s is required for BT" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("BT", "T0",
                        "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("E", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True
