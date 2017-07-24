# This is a wapper for independent binary model. It is a PINT timing model class
import astropy.units as u
from . import parameter as p
from .timing_model import DelayComponent, MissingParameter
from pint import ls,GMsun,Tsun


class PulsarBinary(DelayComponent):
    """ A wapper class for independent pulsar binary model interact with PINT
    platform. The calculations are done by the classes located at
    pint/models/stand_alone_psr_binary
    Binary variables naming:
    Eccentric Anomaly               E (not parameter ECC)
    Mean Anomaly                    M
    True Anomaly                    nu
    Eccentric                       ecc
    Longitude of periastron         omega
    projected semi-major axis of orbit   a1

    """
    def __init__(self,):
        super(PulsarBinary, self).__init__()
        self.category = 'pulsar_system'
        self.binary_model_name = None
        self.barycentric_time = None
        self.binary_model_class = None
        self.add_param(p.floatParameter(name="PB",
            units=u.day,
            description="Orbital period", long_double=True))


        self.add_param(p.floatParameter(name="PBDOT",
            units = u.day/u.day,
            description="Orbital period derivitve respect to time", \
            unit_scale=True, scale_factor=1e-12, scale_threshold=1e-7))

        self.add_param(p.floatParameter(name="A1",
            units=ls,
            description="Projected semi-major axis, a*sin(i)"))
        # NOTE: the DOT here takes the value and times 1e-12, tempo/tempo2 can
        # take both.
        self.add_param(p.floatParameter(name = "A1DOT", aliases = ['XDOT'],
            units=ls/u.s,
            description="Derivitve of projected semi-major axis, da*sin(i)/dt", \
            unit_scale=True, scale_factor=1e-12, scale_threshold=1e-7))

        self.add_param(p.floatParameter(name="ECC",
            units="",
            aliases = ["E"],
            description="Eccentricity"))

        self.add_param(p.floatParameter(name="EDOT",
             units="1/s",
             description="Eccentricity derivitve respect to time", \
             unit_scale=True, scale_factor=1e-12, scale_threshold=1e-7))

        self.add_param(p.MJDParameter(name="T0",
            description="Epoch of periastron passage", time_scale='tdb'))

        self.add_param(p.floatParameter(name="OM",
            units=u.deg,
            description="Longitude of periastron",long_double=True))

        self.add_param(p.floatParameter(name="OMDOT",
            units="deg/year",
            description="Longitude of periastron", long_double=True))

        self.add_param(p.floatParameter(name="M2",
             units=u.M_sun,
             description="Mass of companian in the unit Sun mass"))

        self.add_param(p.floatParameter(name="SINI", value=0.0,
             units="",
             description="Sine of inclination angle"))

        # Set up delay function
        self.delay_funcs_component += [self.binarymodel_delay,]

    def setup(self):
        super(PulsarBinary, self).setup()
        for bpar in self.params:
            self.register_deriv_funcs(self.d_binary_delay_d_xxxx, bpar)
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

    def update_binary_object(self, toas, acc_delay=None):
        """
        Update binary object instance for this set of parameters/toas
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        # Get barycnetric toa first
        if acc_delay is None:
            # If the accumulate delay is not provided, it will try to get
            # the barycentric correction.
            acc_delay = self.delay(toas, self.__class__.__name__, False)
        self.barycentric_time = toas['tdbld'] * u.day - acc_delay 
        pardict = {}
        for par in self.binary_instance.binary_params:
            binary_par_names = [par,]
            if par in self.binary_instance.param_aliases.keys():
                aliase = self.binary_instance.param_aliases[par]
            else:
                aliase = []
            if hasattr(self, par) or \
                list(set(aliase).intersection(self.params))!=[]:
                binObjpar = getattr(self, par)
                if binObjpar.value is None:
                    continue
                pardict[par] = binObjpar.value * binObjpar.units
        #NOTE something is wrong here.
        self.binary_instance.update_input(self.barycentric_time, pardict)

    def binarymodel_delay(self, toas, acc_delay=None):
        """Return the binary model independent delay call"""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.binary_delay()

    def d_binary_delay_d_xxxx(self, toas, param, acc_delay):
        """Return the bianry model delay derivtives"""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.d_binarydelay_d_par(param)

    def print_par(self,):
        result = "BINARY {0}\n".format(self.binary_model_name)
        for p in self.params:
            par = getattr(self, p)
            if par.quantity is not None:
                result += par.as_parfile_line()
        return result
