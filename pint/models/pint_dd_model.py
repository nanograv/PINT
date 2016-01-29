import numpy as np
import time
from pint import ls,GMsun,Tsun
from pint import utils
from .DDindependent import PSRdd as DDmodel
from .psr_binary_wapper import PSRbinaryWapper
from .parameter import Parameter, MJDParameter
from .timing_model import Cache, TimingModel, MissingParameter
import astropy
from ..utils import time_from_mjd_string, time_to_longdouble
import astropy.units as u

class DDwrapper(PSRbinaryWapper):
    """A DD modle wapper
    """

    def __init__(self,):
        super(DDwrapper, self).__init__()
        self.BinaryModelName = 'DD'

        self.add_param(Parameter(name="A0",
             units="s",
             description="DD model aberration parameter A0",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="B0",
             units="s",
             description="DD model aberration parameter B0",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="GAMMA",
             units="second",
             description="Binary Einsten delay GAMMA term",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="DR",
             units="",
             description="Relativistic deformation of the orbit",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="DTH",
             units="",
             description="Relativistic deformation of the orbit",
             parse_value=np.double),binary_param = True)


        self.add_param(Parameter(name="SINI",
             units="",
             description="Sine of inclination angle",
             parse_value=np.double),binary_param = True)

        self.binary_delay_funcs += [self.DD_delay,]
        self.delay_funcs['L2'] += [self.DD_delay,]
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

        self.apply_units()

    @Cache.use_cache
    def get_dd_object(self, toas):
        """
        Obtain the DDmodel object for this set of parameters/toas
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        # Get barycnetric toa first
        self.barycentricTime = self.get_barycentric_toas(toas)

        ddobj = DDmodel(self.barycentricTime)

        pardict = {}
        for par in ddobj.parDefault.keys():     # Note, not called dd_params, but just params
            if hasattr(self, par):
                pardict[par] = getattr(self, par).num_value

        ddobj.set_par_values(pardict)    # This now does a lot of stuff that is in the PSRdd.__init__

        return ddobj

    @Cache.use_cache
    def DD_delay(self, toas):
        """Return the DD timing model delay"""
        ddob = self.get_dd_object(toas)

        return ddob.DDdelay()

    @Cache.use_cache
    def d_delay_d_par(self,par,toas):
        """Return the DD timing model delay derivtives"""
        ddob = self.get_dd_object(toas)

        return ddob.d_delay_d_par(par)
