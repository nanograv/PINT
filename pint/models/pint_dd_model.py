import numpy as np
import time
from pint import ls,GMsun,Tsun
from pint import utils
from .pulsar_binaries.DD_model import DDmodel
from .pint_pulsar_binary import PSRbinaryWapper
from .parameter import Parameter, MJDParameter
from .timing_model import Cache, TimingModel, MissingParameter
import astropy
from ..utils import time_from_mjd_string, time_to_longdouble
import astropy.units as u


class DDwrapper(PSRbinaryWapper):
    """This is a PINT pulsar binary dd model class a subclass of PSRbinaryWapper.
       It is a wrapper for independent DDmodel class defined in ./pulsar_binary/DD_model.py
       All the detailed calculations are in the independent DDmodel.
       The aim for this class is to connect the independent binary model with PINT platform
    """

    def __init__(self,):
        super(DDwrapper, self).__init__()
        self.binary_model_name = 'DD'
        self.binary_model_class = DDmodel
        self.add_param(Parameter(name="A0", units="s",
                       description="DD model aberration parameter A0"),
                       binary_param = True)
        self.add_param(Parameter(name="B0", units="s",
                       description="DD model aberration parameter B0"),
                       binary_param = True)
        self.add_param(Parameter(name="GAMMA", units="second",
                       description="Binary Einsten delay GAMMA term"),
                       binary_param = True)
        self.add_param(Parameter(name="DR", units="",
                       description="Relativistic deformation of the orbit"),
                       binary_param = True)
        self.add_param(Parameter(name="DTH", units="",
                       description="Relativistic deformation of the orbit"),
                       binary_param = True)
        self.add_param(Parameter(name="SINI", units="",
                       description="Sine of inclination angle"),
                       binary_param = True)

    def setup(self):
        super(DDwrapper,self).setup()
        for p in ("PB", "T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("DD", p,
                                       "%s is required for DD" % p)
        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("DD", "T0",
                        "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("ECC", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True


    
