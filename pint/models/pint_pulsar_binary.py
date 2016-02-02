# This is a wapper for independent binary model. It is a PINT timing model class
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
SECS_PER_JUL_YEAR = SECS_PER_DAY*365.25
import parameter as p
from .timing_model import Cache, TimingModel, MissingParameter
from ..phase import Phase
from ..utils import time_from_mjd_string, time_to_longdouble, \
    time_from_longdouble
from ..orbital.kepler import eccentric_from_mean
from .btmodel import BTmodel
import numpy as np
import time
from pint import ls,GMsun,Tsun
from pint import utils


class PSRbinaryWapper(TimingModel):
    """ A class for independent pulsar binary model wapper.
        Binary variables naming:
        Eccentric Anomaly               E (not parameter ECC)
        Mean Anomaly                    M
        True Anomaly                    nu
        Eccentric                       ecc
        Longitude of periastron         omega
        projected semi-major axis of orbit   a1
    """
    def __init__(self,):
        super(PSRbinaryWapper, self).__init__()
        self.BinaryModelName = None
        self.barycentricTime = None
        self.binary_delay_funcs= []
        self.binary_params = []
        self.add_param(p.floatParameter(name="PB",
            units=u.day,
            description="Orbital period"),
            binary_param = True)


        self.add_param(p.floatParameter(name="PBDOT",
            units=u.day/u.day,
            description="Orbital period derivitve respect to time"),
            binary_param = True)


        self.add_param(p.floatParameter(name="XPBDOT",
            units=u.s/u.s,
            description="Rate of change of orbital period minus GR prediction"),
            binary_param = True)


        self.add_param(p.floatParameter(name="A1",
            units=ls,
            description="Projected semi-major axis, a*sin(i)"),
            binary_param = True)

        self.add_param(p.floatParameter(name = "A1DOT",
            units=ls/u.s,
            description="Derivitve of projected semi-major axis, da*sin(i)/dt"),
            binary_param = True)

        self.add_param(p.floatParameter(name="ECC",
            units="",
            aliases = ["E"],
            description="Eccentricity"),
            binary_param = True)

        self.add_param(p.floatParameter(name="EDOT",
             units="1/s",
             description="Eccentricity derivitve respect to time"),
             binary_param = True)

        self.add_param(p.MJDParameter(name="T0",
            description="Epoch of periastron passage", time_scale='tdb'),
            binary_param = True)

        self.add_param(p.floatParameter(name="OM",
            units=u.deg,
            description="Longitude of periastron"),
            binary_param = True)

        self.add_param(p.floatParameter(name="OMDOT",
            units="deg/year",
            description="Longitude of periastron"),
            binary_param = True)

        self.add_param(p.floatParameter(name="M2",
             units=u.M_sun,
             description="Mass of companian in the unit Sun mass"),
             binary_param = True)


    def setup(self):
        super(PSRbinaryWapper, self).setup()



    def apply_units(self):
        for bpar in self.binary_params:
            bparObj = getattr(self,bpar)
            if bparObj.num_value is None or bparObj.num_unit is None:
                continue
            bparObj.value = bparObj.num_value*u.Unit(bparObj.num_unit)

    @Cache.use_cache
    def get_binary_object(self, toas, binaryObj):
        """
        Obtain the independent binary object for this set of parameters/toas
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        # Get barycnetric toa first
        self.barycentricTime = self.get_barycentric_toas(toas)

        binobj = binaryObj(self.barycentricTime)
        pardict = {}
        for par in binobj.binary_params:
            if par in binobj.param_aliases.keys():
                aliase = binobj.param_aliases[par]
            if hasattr(self, par) or \
                list(set(aliase).intersection(self.params)!=[]) :
                binObjpar = getattr(self, par)
                if binObjpar.num_value is None:
                    continue
                pardict[par] = binObjpar.num_value*binObjpar.num_unit

        binobj.set_param_values(pardict)
        return binobj


    def binary_delay(self,toas):
        """Returns total pulsar binary delay.
           Parameters
           ----------
           toas : PINT toas table
           Return
           ----------
           Pulsar binary delay in the units of second
        """
        bdelay = np.longdouble(np.zeros(len(toas)))*u.s
        for bdf in self.binary_delay_funcs:
            bdelay+= bdf(toas)
        return bdelay
