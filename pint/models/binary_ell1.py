import numpy as np
import time
from pint import ls,GMsun,Tsun
from pint import utils
from .stand_alone_psr_binaries.ELL1_model import ELL1model
from .pulsar_binary import PulsarBinary
import parameter as p
from .timing_model import Cache, TimingModel, MissingParameter
import astropy
from ..utils import time_from_mjd_string, time_to_longdouble
import astropy.units as u
from warnings import warn

class BinaryELL1(PulsarBinary):
    """This is a PINT pulsar binary ELL1 model class a subclass of PulsarBinary.
    It is a wrapper for stand alone ELL1model class defined in ./pulsar_binary/ELL1_model.py
    All the detailed calculations are in the stand alone ELL1model.
    The aim for this class is to connect the stand alone binary model with PINT platform
    ELL1model special parameters:
    TASC Epoch of ascending node
    EPS1 First Laplace-Lagrange parameter, ECC x sin(OM) for ELL1 model
    EPS2 Second Laplace-Lagrange parameter, ECC x cos(OM) for ELL1 model
    EPS1DOT First derivative of first Laplace-Lagrange parameter
    EPS2DOT Second derivative of second Laplace-Lagrange parameter
    """

    def __init__(self):
        super(BinaryELL1, self).__init__()
        self.binary_model_name = 'ELL1'
        self.binary_model_class = ELL1model

        self.add_param(p.MJDParameter(name="TASC",
                       description="Epoch of ascending node", time_scale='tdb'),
                       binary_param = True)

        self.add_param(p.floatParameter(name="EPS1", units="",
             description="First Laplace-Lagrange parameter, ECC x sin(OM) for ELL1 model",
             long_double = True), binary_param = True)

        self.add_param(p.floatParameter(name="EPS2", units="",
             description="Second Laplace-Lagrange parameter, ECC x cos(OM) for ELL1 model",
             long_double = True), binary_param = True)

        self.add_param(p.floatParameter(name="EPS1DOT", units="1e-12/s",
             description="First derivative of first Laplace-Lagrange parameter",
             long_double = True), binary_param = True)

        self.add_param(p.floatParameter(name="EPS2DOT", units="1e-12/s",
             description="Second derivative of first Laplace-Lagrange parameter",
             long_double = True), binary_param = True)

    def setup(self):
        """Check out parameters setup.
        """
        super(BinaryELL1, self).setup()
        for p in ['EPS1', 'EPS2']:
            if getattr(self, p).value is None:
                raise MissingParameter("ELL1", p, p + " is required for ELL1 model.")
        # Check TASC
        if self.TASC.value is None:
            if self.ECC.value == 0.0:
                warn("Since ECC is 0.0, using T0 as TASC.")
                if self.T0.value is not None:
                    self.TASC.value = self.T0.value
                else:
                    raise MissingParameter("ELL1", 'T0', "T0 or TASC is required for ELL1 model.")
            else:
                raise MissingParameter("ELL1", 'TASC', "TASC is required for ELL1 model.")
