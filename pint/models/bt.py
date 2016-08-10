
"""This model provides the BT (Blandford & Teukolsky 1976, ApJ, 205, 580) model.
    """
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
SECS_PER_JUL_YEAR = SECS_PER_DAY*365.25

import parameter as p
from .timing_model import Cache, TimingModel, MissingParameter
from ..phase import Phase
from ..utils import time_from_mjd_string, time_to_longdouble
from ..orbital.kepler import eccentric_from_mean
from .btmodel import BTmodel
import astropy.time
import numpy as np
import time
from astropy import log

class BT(TimingModel):
    """This class provides an implementation of the BT model
        by Blandford & Teukolsky 1976, ApJ, 205, 580.
    """
    def __init__(self):
        super(BT, self).__init__()

        # TO DO: add commonly used parameters such as T90, TASC with clear,
        # documented usage.

        # Parameters are mostly defined as numpy doubles.
        # Some might become long doubles in the future.
        self.binary_model_name = 'BT'
        self.add_param(p.floatParameter(name="PB",
            units="s",
            description="Orbital period"))

        self.add_param(p.floatParameter(name="A1",
            units="lt-s",
            description="Projected semi-major axis"))

        self.add_param(p.floatParameter(name="E",
            units="",
            aliases = ["ECC"],
            description="Eccentricity"))

        # Warning(paulr): This says the units on OM are deg (which is correct)
        # But then it converts the value to radians!
        # This may work OK in here, but when printing the value to a new
        # par file, it comes out wrong!
        self.add_param(p.AngleParameter(name="OM",
            units="deg",
            description="Longitude of periastron"))

        self.add_param(p.MJDParameter(name="T0",
            time_scale='tdb', description="Epoch of periastron passage"))

        self.add_param(p.floatParameter(name="PBDOT",
            units="s/s",
            description="First derivative of orbital period"))

        self.add_param(p.floatParameter(name="OMDOT",
            units="deg/yr",
            description="Periastron advance"))

        self.add_param(p.floatParameter(name="XDOT",
            units="s/s",
            description="Orbital spin-down rate"))

        self.add_param(p.floatParameter(name="EDOT",
            units="s^-1",
            description="Orbital spin-down rate"))

        self.add_param(p.floatParameter(name="GAMMA",
            units="s",
            description="Time dilation & gravitational redshift"))

        self.delay_funcs['L2'] += [self.BT_delay,]

    def setup(self):
        super(BT, self).setup()

        # If any necessary parameter is missing, raise MissingParameter.
        # This will probably be updated after ELL1 model is added.
        for p in ("PB", "T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p,
                                       "%s is required for BT" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "XDOT"):
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
        if self.E.value == 0 or self.E.value is None:
            for p in ("E", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

    @Cache.use_cache
    def get_bt_object(self, toas):
        """
        Obtain the BTmodel object for this set of parameters/toas
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        bt_params = ['PB', 'PBDOT', 'ECC', 'EDOT', 'OM', 'OMDOT', 'A1', \
                'A1DOT', 'T0', 'GAMMA']
        aliases = {'A1DOT':'XDOT', 'ECC':'E'}

        pardict = {}
        for par in bt_params:
            key = par if not par in aliases else aliases[par]

            # T0 needs to be converted to long double
            parobj = getattr(self, key)
            pardict[par] = parobj.value 

        # Apply all the delay terms, except for the binary model itself
        tt0 = toas['tdbld'] * SECS_PER_DAY
        for df in self.delay_funcs['L1']:
            if df != self.BT_delay:
                tt0 -= df(toas)

        # Return the BTmodel object
        return BTmodel(tt0/SECS_PER_DAY, **pardict)

    @Cache.use_cache
    def BT_delay(self, toas):
        """Return the BT timing model delay"""
        btob = self.get_bt_object(toas)

        return btob.delay()

    @Cache.use_cache
    def d_delay_d_A1(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_A1()

    @Cache.use_cache
    def d_delay_d_XDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_A1DOT()

    @Cache.use_cache
    def d_delay_d_OM(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_OM()

    @Cache.use_cache
    def d_delay_d_OMDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_OMDOT()

    @Cache.use_cache
    def d_delay_d_E(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_ECC()

    @Cache.use_cache
    def d_delay_d_EDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_EDOT()

    @Cache.use_cache
    def d_delay_d_PB(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_PB()

    @Cache.use_cache
    def d_delay_d_PBDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_PBDOT()

    @Cache.use_cache
    def d_delay_d_T0(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_T0()

    @Cache.use_cache
    def d_delay_d_GAMMA(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_GAMMA()
