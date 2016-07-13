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


class PulsarBinary(TimingModel):
<<<<<<< HEAD
<<<<<<< HEAD
    """ A wapper class for independent pulsar binary model interact with PINT
    platform. The calculations are done by the classes located at
=======
    """ A wapper class for independent pulsar binary model interact with PINT
    platform. The calculations are done by the class located at
>>>>>>> fix a bug and add docstrings
    pint/models/pulsar_binary

    Binary variables naming:
    Eccentric Anomaly               E (not parameter ECC)
    Mean Anomaly                    M
    True Anomaly                    nu
    Eccentric                       ecc
    Longitude of periastron         omega
    projected semi-major axis of orbit   a1
<<<<<<< HEAD

=======
    """ A class for independent pulsar binary model wapper.
        Binary variables naming:
        Eccentric Anomaly               E (not parameter ECC)
        Mean Anomaly                    M
        True Anomaly                    nu
        Eccentric                       ecc
        Longitude of periastron         omega
        projected semi-major axis of orbit   a1
>>>>>>> Change DDwrapper class name
=======
>>>>>>> fix a bug and add docstrings
    """
    def __init__(self,):
        super(PulsarBinary, self).__init__()
        self.binary_model_name = None
        self.barycentric_time = None
        self.binary_model_class = None
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


        self.add_param(p.floatParameter(name="XPBDOT", value = 0.0,
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

        # Set up delay function
        self.binary_delay_funcs += [self.binarymodel_delay,]
        self.delay_funcs['L2'] += [self.binarymodel_delay,]

    def setup(self):
        super(PulsarBinary, self).setup()
        for bpar in self.binary_params:
            self.make_delay_binary_deriv_funcs(bpar)
            self.delay_derivs += [getattr(self, 'd_delay_binary_d_' + bpar)]
        # Setup the model isinstance
        self.binary_instance = self.binary_model_class()

    # With new parameter class set up, do we need this?
    def apply_units(self):
        """Apply units to parameter value.
        """
        for bpar in self.binary_params:
            bparObj = getattr(self,bpar)
            if bparObj.value is None or bparObj.units is None:
                continue

            bparObj.value = bparObj.value * u.Unit(bparObj.units)

    @Cache.use_cache
    def update_binary_object(self, toas):
        """
        Update binary object instance for this set of parameters/toas
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        # Get barycnetric toa first
        self.barycentric_time = self.get_barycentric_toas(toas)
        pardict = {}
        for par in self.binary_instance.binary_params:
            if par in self.binary_instance.param_aliases.keys():
                aliase = self.binary_instance.param_aliases[par]
            if hasattr(self, par) or \
                list(set(aliase).intersection(self.params))!=[] :
                binObjpar = getattr(self, par)
                if binObjpar.value is None:
                    continue
                pardict[par] = binObjpar.value * binObjpar.units

        self.binary_instance.update_input(self.barycentric_time, pardict)

    @Cache.use_cache
    def binarymodel_delay(self, toas):
        """Return the binary model independent delay call"""
        self.update_binary_object(toas)
        return self.binary_instance.binary_delay()

    @Cache.use_cache
    def d_binary_delay_d_xxxx(self,param,toas):
        """Return the bianry model delay derivtives"""
        self.update_binary_object(toas)
        return self.binary_instance.d_binarydelay_d_par(param)

    def make_delay_binary_deriv_funcs(self, param):
        """This is a funcion to make binary derivative functions to the formate
        of d_binary_delay_d_paramName(toas)
        """
        # TODO make this function more generalized?
        def deriv_func(toas):
            return self.d_binary_delay_d_xxxx(param, toas)
        deriv_func.__name__ = 'd_delay_binary_d_' + param
        setattr(self, 'd_delay_binary_d_' + param, deriv_func)
