# This is a wapper for independent binary model. It is a PINT timing model class
from __future__ import absolute_import, print_function, division
import astropy.units as u
import numpy as np
from . import parameter as p
from .timing_model import DelayComponent, MissingParameter
from astropy import log
from pint import ls,GMsun,Tsun
from .stand_alone_psr_binaries import binary_orbits as bo


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

        self.add_param(p.floatParameter(name="SINI", units="",
             description="Sine of inclination angle"))

        self.add_param(p.prefixParameter(name="FB0", value=None, units='1/s^1',
                       description="0th time derivative of frequency of orbit",
                       unit_template=self.FBX_unit, aliases = ['FB'],
                       description_template=self.FBX_description,
                       type_match='float', long_double=True))

        self.interal_params = []
        self.warn_default_params = ['ECC', 'OM']
        # Set up delay function
        self.delay_funcs_component += [self.binarymodel_delay,]

    def setup(self):
        super(PulsarBinary, self).setup()

        if self.T0.value is not None:
            # set the units
            self.T0.scale = self.UNITS.value.lower()

        for bpar in self.params:
            self.register_deriv_funcs(self.d_binary_delay_d_xxxx, bpar)
        # Setup the model isinstance
        self.binary_instance = self.binary_model_class()
        # Setup the FBX orbits if FB is set.
        FBX_mapping = self.get_prefix_mapping_component('FB')
        FBXs = {}
        for fbn in FBX_mapping.values():
            FBXs[fbn] = getattr(self, fbn).quantity
        if None not in list(FBXs.values()):
            for fb_name, fb_value in FBXs.items():
                if fb_value is None:
                    raise MissingParameter(self.binary_model_name, fb_name + \
                                           " is required for FB orbits.")
                self.binary_instance.add_binary_params(fb_name, fb_value)
            self.binary_instance.orbits_cls = bo.OrbitFBX(self.binary_instance,
                                                          list(FBXs.keys()))

    def check_required_params(self, required_params):
        # seach for all the possible to get the parameters.
        for p in required_params:
            par = getattr(self, p)
            if par.value is None:
                # try to search if there is any class method that computes it
                method_name = p.lower() + "_func"
                try:
                    par_method = getattr(self.binary_instance, method_name)
                    _ = par_method()
                except:
                    raise MissingParameter(self.binary_model_name, p + \
                                           " is required for '%s'." %
                                           self.binary_model_name)

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
        # Get barycentric toa first
        updates = {}
        tbl = toas.table
        if acc_delay is None:
            # If the accumulate delay is not provided, it will try to get
            # the barycentric correction.
            acc_delay = self.delay(toas, self.__class__.__name__, False)
        self.barycentric_time = tbl[self.UNITS.value.lower()+'ld'] * u.day - acc_delay
        updates['psr_pos'] = self.ssb_to_psb_xyz_ICRS(epoch=tbl[self.UNITS.value.lower()+'ld'].astype(np.float64))
        updates['barycentric_toa'] = self.barycentric_time
        updates['obs_pos'] = tbl['ssb_obs_pos'].quantity
        for par in self.binary_instance.binary_params:
            binary_par_names = [par,]
            if par in self.binary_instance.param_aliases.keys():
                aliase = self.binary_instance.param_aliases[par]
            else:
                aliase = []

            if hasattr(self, par) or \
                list(set(aliase).intersection(self.params))!=[]:
                pint_bin_name = self.match_param_aliases(par)
                if pint_bin_name == "" and par in self.interal_params:
                    pint_bin_name = par
                binObjpar = getattr(self, pint_bin_name)
                instance_par = getattr(self.binary_instance, par)
                if hasattr(instance_par, 'value'):
                    instance_par_val = instance_par.value
                else:
                    instance_par_val = instance_par
                if binObjpar.value is None:
                    if binObjpar.name in self.warn_default_params:
                        log.warn("'%s' is not set, using the default value %f "
                                 "instead." % (binObjpar.name, instance_par_val))
                    continue
                if binObjpar.units is not None:
                    updates[par] = binObjpar.value * binObjpar.units
                else:
                    updates[par] = binObjpar.value
        self.binary_instance.update_input(**updates)

    def binarymodel_delay(self, toas, acc_delay=None):
        """Return the binary model independent delay call"""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.binary_delay()

    def d_binary_delay_d_xxxx(self, toas, param, acc_delay):
        """Return the binary model delay derivtives"""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.d_binarydelay_d_par(param)

    def print_par(self,):
        result = "BINARY {0}\n".format(self.binary_model_name)
        for p in self.params:
            par = getattr(self, p)
            if par.quantity is not None:
                result += par.as_parfile_line()
        return result

    def FBX_unit(self, n):
        return "1/s^%d" % (n+1) if n else "1/s"

    def FBX_description(self, n):
        return "%dth time derivative of frequency of orbit" % n
