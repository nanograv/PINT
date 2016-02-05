# This is a wapper for independent binary model. It is a PINT timing model class
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
SECS_PER_JUL_YEAR = SECS_PER_DAY*365.25

from .parameter import Parameter, MJDParameter
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
        self.add_param(Parameter(name="PB",
            units=u.day,
            description="Orbital period",
            parse_value=utils.data2longdouble),binary_param = True)


        self.add_param(Parameter(name="PBDOT",
            units=u.day/u.day,
            description="Orbital period derivitve respect to time",
            parse_value=utils.data2longdouble),binary_param = True)


        self.add_param(Parameter(name="XPBDOT",
            units=u.s/u.s,
            description="Rate of change of orbital period minus GR prediction",
            parse_value=utils.data2longdouble),binary_param = True)


        self.add_param(Parameter(name="A1",
            units=ls,
            description="Projected semi-major axis, a*sin(i)",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name = "A1DOT",
            units=ls/u.s,
            description="Derivitve of projected semi-major axis, da*sin(i)/dt",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="ECC",
            units="",
            aliases = ["E"],
            description="Eccentricity",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="EDOT",
             units="1/s",
             description="Eccentricity derivitve respect to time",
             parse_value=np.double),binary_param = True)

        self.add_param(MJDParameter(name="T0",
            description="Epoch of periastron passage", time_scale='tdb'),
            binary_param = True)

        self.add_param(Parameter(name="OM",
            units=u.deg,
            description="Longitude of periastron",
            parse_value=lambda x: np.double(x)),binary_param = True)

        self.add_param(Parameter(name="OMDOT",
            units="deg/year",
            description="Longitude of periastron",
            parse_value=lambda x: np.double(x)),binary_param = True)

        self.add_param(Parameter(name="M2",
             units=u.M_sun,
             description="Mass of companian in the unit Sun mass",
             parse_value=np.double),binary_param = True)


    def setup(self):
        super(PSRbinaryWapper, self).setup()



    def apply_units(self):
        for bpar in self.binary_params:
            bparObj = getattr(self,bpar)
            if bparObj.num_value is None or bparObj.num_unit is None:
                continue
            bparObj.value = bparObj.num_value*u.Unit(bparObj.num_unit)


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
