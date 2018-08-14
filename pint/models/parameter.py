# parameter.py
# Defines Parameter class for timing model parameters
from __future__ import absolute_import, print_function, division
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string,\
    time_to_longdouble, is_number, time_from_longdouble, str2longdouble, \
    longdouble2string, data2longdouble, split_prefixed_name
import numpy
import astropy.time as time
from astropy import log
from pint import pint_units
from pint import pulsar_mjd
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
import re
import numbers
from . import priors
from ..toa_select import TOASelect


class Parameter(object):
    """A base PINT class describing a single timing model parameter.
    PINT Parameter class will have

    A `Parameter` object can be created with one of the subclasses provided by
    `PINT` depending on the parameter usage.
    Current Parameter type:
    [`floatParameter`, `strParameter`, `boolParameter`, `MDJParameter`,
     `AngleParameter`, `prefixParameter`, `maskParameter`]

    Parameter Mechanism
    Parameter current value information will be stored at `.quantity` property
    which can be a flexible format, for example astropy.quantity in
    floatParameter and string in strParameter, (For more detail see Parameter
    subclasses docstrings). If applicable, Parameter default unit is
    stored at`.units` property which is an `astropy.unit` object. Property
    `.value` always returns a pure value associate with `.units` from
    `.quantity`. `.uncertainty` provides the storage for parameter uncertainty
    and `.uncertainty_value` for pure uncertainty value. Like `.value`,
    `.uncertainty_value` always associate with default unit.

    Parameters
    ----------
    name : str, optional
        The name of the parameter.
    value : number, str, `Astropy.units.Quantity` object, or other data type or
            object
        The input parameter value.
    units : str or Astropy.units, optional
        Parameter default unit. Parameter .value and .uncertainty_value attribute
        will associate with the default units.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.
    continuous : bool, optional
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    print_quantity : method, optional
        A function that converts the internal value to a string for output.
    set_quantity : method, optional
        A function that sets the quantity property
    get_value:
        A function that get purely value from quantity attribute

    Attributes
    ----------
    quantity: Type depends on the parameter subclass, it can be anything
        An internal storage for parameter value and units
    """

    def __init__(self, name=None, value=None, units=None, description=None,
                 uncertainty=None, frozen=True, aliases=None, continuous=True,
                 print_quantity=str, set_quantity=lambda x: x,
                 get_value=lambda x: x,
                 prior=priors.Prior(priors.UniformUnboundedRV()),
                 set_uncertainty=fortran_float):

        self.name = name  # name of the parameter
        self.units = units  # Default unit
        self.set_quantity = set_quantity
        # Method to get value
        self.get_value = get_value
        # method to convert quantity to a string.
        self.print_quantity = print_quantity
        # Method to get uncertainty from input
        self.set_uncertainty = set_uncertainty
        self.from_parfile_line = self.from_parfile_line_regular
        self.quantity = value  # The value of parameter, internal storage
        self.prior = prior

        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = [] if aliases is None else aliases
        self.is_prefix = False
        self.paramType = 'Not specified'  # Type of parameter. Here is general type
        self.valueType = None
        self.special_arg = []
    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self,p):
        if not isinstance(p,priors.Prior):
            log.error("prior must be an instance of Prior()")
        self._prior = p

    # Setup units property
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unt):
        # Check if this is the first time set units and check compatibility
        if hasattr(self, 'quantity'):
            if self.units is not None:
                if unt != self.units:
                    wmsg = 'Parameter '+self.name+' default units has been '
                    wmsg += ' reset to ' + str(unt) + ' from '+ str(self.units)
                    log.warning(wmsg)
                try:
                    if hasattr(self.quantity, 'unit'):
                        _ = self.quantity.to(unt)
                except:
                    log.warning('The value unit is not compatible with'
                                ' parameter units right now.')


        if unt is None:
            self._units = None

        # Always compare a string to pint_units.keys()
        # If search an astropy unit object with a sting list
        # If the string does not match astropy unit, astropy will guess what
        # does the string mean. It will take a lot of time.
        elif isinstance(unt, str) and unt in pint_units.keys():
            # These are special-case unit strings in in PINT
            self._units = pint_units[unt]

        else:
            # Try to use it as an astopy unit.  If this fails,
            # ValueError will be raised.
            self._units = u.Unit(unt)

        if hasattr(self, 'quantity') and hasattr(self.quantity, 'unit'):
            # Change quantity unit to new unit
            self.quantity = self.quantity.to(self._units)
        if hasattr(self, 'uncertainty') and hasattr(self.uncertainty, 'unit'):
            # Change uncertainty unit to new unit
            self.uncertainty = self.uncertainty.to(self._units)

    # Setup quantity property
    @property
    def quantity(self):
        """Return the internal stored parameter value and units.
        """
        return self._quantity

    @quantity.setter
    def quantity(self, val):
        """General wrapper method to set .quantity. For different type of
        parameters, the setter method is stored at .set_quantity attribute.
        """
        if val is None:
            if hasattr(self, 'quantity') and self.quantity is not None:
                raise ValueError('Setting an exist value to None is not'
                                 ' allowed.')
            else:
                self._quantity = val
                return
        self._quantity = self.set_quantity(val)

    def prior_pdf(self,value=None, logpdf=False):
        """Return the prior probability, evaluated at the current value of
        the parameter, or at a proposed value.

        Parameters
        ----------
        value : array_like or float_like

        Probabilities are evaluated using the value attribute
        """
        if value is None:
            return self.prior.pdf(self.value) if not logpdf else self.prior.logpdf(self.value)
        else:
            return self.prior.pdf(value) if not logpdf else self.prior.logpdf(value)

    # Setup .value property
    # .value will get pure number from ._quantity.
    # Setting .value property will change ._quantity.
    @property
    def value(self):
        """Return the pure value of a parameter. This value will associate with
        parameter default value, which is .units attribute.
        """
        if self._quantity is None:
            return None
        else:
            return self.get_value(self._quantity)

    @value.setter
    def value(self, val):
        """Method to set .value. Setting .value attribute will change the
        .quantity attribute other than .value attribute.
        """
        if val is None:
            if not isinstance(self.quantity, (str, bool)) and \
                self._quantity is not None:
                raise ValueError('This parameter value is number convertible. '
                                 'Setting .value to None will lost the '
                                 'parameter value.')
            else:
                self.value = val
        self._quantity = self.set_quantity(val)

    @property
    def uncertainty(self):
        """Return the internal stored parameter uncertainty value and units.
        """
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, val):
        """General wrapper setter for uncertainty. The setting method is stored
        at .set_uncertainty attribute
        """
        if val is None:
            if hasattr(self, 'uncertainty') and self.uncertainty is not None:
                raise ValueError('Setting an exist uncertainty to None is not'
                                 ' allowed.')
            else:
                self._uncertainty = val
                self._uncertainty_value = self._uncertainty
                return
        self._uncertainty = self.set_uncertainty(val)

        # This is avoiding negtive unvertainty input.
        if self._uncertainty is not None and self.uncertainty_value < 0:
            self.uncertainty_value = numpy.abs(self.uncertainty_value)

    @property
    def uncertainty_value(self):
        """Return a pure value from .uncertainty. The unit will associate
        with .units
        """
        if self._uncertainty is None:
            return None
        else:
            return self.get_value(self._uncertainty)

    @uncertainty_value.setter
    def uncertainty_value(self, val):
        """Setter for uncertainty_value. Setting .uncertainty_value will only change
        the .uncertainty attribute.
        """
        if val is None:
            if not isinstance(self.uncertainty, (str, bool)) and \
                self._uncertainty_value is not None:
                log.warning('This parameter has uncertainty value. '
                            'Change it to None will lost information.')
            else:
                self.uncertainty_value = val
        self._uncertainty = self.set_uncertainty(val)

    def print_uncertainty(self, uncertainty):
        return str(uncertainty.to(self.units).value)

    def __str__(self):
        out = self.name
        if self.units is not None:
            out += " (" + str(self.units) + ")"
        if self.quantity is not None:
            out += " " + self.print_quantity(self.quantity)
        else:
            out += " " + "UNSET"
            return out
        if self.uncertainty is not None and isinstance(self.value, numbers.Number):
            out += " +/- " + str(self.uncertainty.to(self.units))
        return out

    def set(self, value):
        """Parses a string 'value' into the appropriate internal representation
        of the parameter.
        """
        self.value = value

    def add_alias(self, alias):
        """Add a name to the list of aliases for this parameter."""
        self.aliases.append(alias)

    def help_line(self):
        """Return a help line containing parameter name, description and units."""
        out = "%-12s %s" % (self.name, self.description)
        if self.units is not None:
            out += ' (' + str(self.units) + ')'
        return out

    def as_parfile_line(self):
        """Return a parfile line giving the current state of the parameter."""
        # Don't print unset parameters
        if self.quantity is None:
            return ""
        line = "%-15s %25s" % (self.name, self.print_quantity(self.quantity))
        if self.uncertainty is not None:
            line += " %d %s" % (0 if self.frozen else 1, \
                                self.print_uncertainty(self.uncertainty))
        elif not self.frozen:
            line += " 1"
        return line + "\n"

    def from_parfile_line_regular(self, line):
        """
        Parse a parfile line into the current state of the parameter.
        Returns True if line was successfully parsed, False otherwise.
        Notes
        -----
        The accepted format:
            NAME value
            NAME value fit_flag
            NAME value fit_flag uncertainty
            NAME value uncertainty
        """
        try:
            k = line.split()
            name = k[0].upper()
        except IndexError:
            return False
        # Test that name matches
        if not self.name_matches(name):
            return False
        if len(k) < 2:
            return False
        if len(k) >= 2:
            self.set(k[1])
        if len(k) >= 3:
            try:
                fit_flag = int(k[2])
                if fit_flag == 0:
                    self.frozen = True
                    ucty = 0.0
                elif fit_flag == 1:
                    self.frozen = False
                    ucty = 0.0
                else:
                    ucty = fit_flag
            except:
                if is_number(k[2]):
                    ucty = k[2]
                else:
                    errmsg = 'Unidentified string ' + k[2] + ' in'
                    errmsg += ' parfile line ' + k
                    raise ValueError(errmsg)

            if len(k) >= 4:
                ucty = k[3]
            self.uncertainty = self.set_uncertainty(ucty)
        return True

    def name_matches(self, name):
        """Whether or not the parameter name matches the provided name
        """
        return (name == self.name.upper()) or (name in map(lambda x: x.upper(),
                                                           self.aliases))

class floatParameter(Parameter):
    """This is a Parameter type that is specific to the parameters has a float/
    float128 quantity as its value.

    `.quantity` stores current parameter value and its unit in an
    `astropy.units.quantity` class. The unit of `.quantity` can be any unit
    that convertible to default unit.

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : number, str, `Astropy.units.Quantity` object,
        The input parameter float value.
    units : str or Astropy.units
        Parameter default unit. Parameter .value and .uncertainty_value attribute
        will associate with the default units. If unit is dimensionless, use
        "''" as its unit.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.
    continuous : bool, optional, default True
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    long_double : bool, optional, default False
        A flag specifying whether value is float or float128/longdouble.

    Example::
        >>> from parameter import floatParameter
        >>> test = floatParameter(name='test1', value=100.0, units='second')
        >>> print test
        test1 (s) 100.0
    """
    def __init__(self, name=None, value=None, units=None, description=None,
                 uncertainty=None, frozen=True, aliases=None, continuous=True,
                 long_double=False, unit_scale=False, scale_factor=None,
                 scale_threshold=None, **kwargs):
        self.long_double = long_double
        self.scale_factor = scale_factor
        self.scale_threshold = scale_threshold
        set_quantity = self.set_quantity_float
        print_quantity = self.print_quantity_float
        get_value = self.get_value_float
        set_uncertainty = self.set_quantity_float
        self._unit_scale = False
        super(floatParameter, self).__init__(name=name, value=value,
                                             units=units, frozen=True,
                                             aliases=aliases,
                                             continuous=continuous,
                                             description=description,
                                             uncertainty=uncertainty,
                                             print_quantity=print_quantity,
                                             set_quantity=set_quantity,
                                             get_value=get_value,
                                             set_uncertainty=set_uncertainty)
        self.paramType = 'floatParameter'
        self.special_arg += ['long_double', 'unit_scale', 'scale_threshold',
                             'scale_factor']
        self.unit_scale = unit_scale
        self._original_units = self.units

    @property
    def long_double(self):
        return self._long_double

    @long_double.setter
    def long_double(self, val):
        """long double setter, if a floatParameter's longdouble flag has been
        changed, `.quantity` will get reset in order to get to the right data
        type.
        """
        if not isinstance(val, bool):
            raise ValueError("long_double property can only be set as boolean"
                             " type")
        if hasattr(self, 'long_double'):
            if self.long_double != val and hasattr(self, 'quantity'):
                if not val:
                    log.warning("Setting floatParameter from long double to float,"
                                " precision will be lost.")
                # Reset quantity to its asked type
                self._long_double = val
                self.quantity = self.quantity
        else:
            self._long_double = val

    @property
    def unit_scale(self):
        return self._unit_scale

    @unit_scale.setter
    def unit_scale(self, val):
        old_unit_scale = self._unit_scale
        self._unit_scale = val
        if self._unit_scale:
            if self.scale_factor is None:
                raise ValueError("The scale factor should be given if unit_scale"
                                 " is set to be True.")
            if self.scale_threshold is None:
                raise ValueError("The scale threshold should be given if unit_scale"
                                 " is set to be True.")
        else:
            if old_unit_scale: # This makes sure the unit_scale if from True to false
                self.units = self._original_units

    def set_quantity_float(self, val):
        """Set value method specific for float parameter
        accept format
        1. Astropy quantity
        2. float
        3. string
        """
        # Check long_double
        if not self._long_double:
            setfunc_with_unit = lambda x: x
            setfunc_no_unit = lambda x: fortran_float(x)
        else:
            setfunc_with_unit = lambda x: data2longdouble(x.value)*x.unit
            setfunc_no_unit = lambda x:  data2longdouble(x)

        # First try to use astropy unit conversion
        try:
            # If this fails, it will raise UnitConversionError
            _ = val.to(self.units)
            result = setfunc_with_unit(val)
        except AttributeError:
            # This will happen if the input value did not have units
            num_value = setfunc_no_unit(val)
            if self.unit_scale:
                if numpy.abs(num_value) > numpy.abs(self.scale_threshold):
                    log.warning("Parameter %s's unit will be scaled to %s %s" \
                             % (self.name, str(self.scale_factor), str(self._original_units)))
                    self.units = self.scale_factor * self._original_units
                else:
                    self.units = self._original_units
            result = (num_value) * self.units

        return result

    def print_quantity_float(self, quan):
        """A function gives print quantity string.
        """
        if not self._long_double:
            result = str(quan.to(self.units).value)
        else:
            result = longdouble2string(quan.to(self.units).value)
        return result

    def get_value_float(self, quan):
        if quan is None:
            return None
        else:
            return quan.to(self.units).value


class strParameter(Parameter):
    """This is a Parameter type that is specific to string values.
    `.quantity` stores current parameter information in a string. `.value`
    returns the same with `.quantity`. `.units` is not applicable.
    `strParameter` is not fitable.

    Parameter
    ---------
    name : str
        The name of the parameter.
    value : str
        The input parameter string value.
    description : str, optional
        A short description of what this parameter means.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example::
        >>> from parameter import strParameter
        >>> test = strParameter(name='test1', value='This is a test',)
        >>> print test
        test1 This is a test
    """
    def __init__(self, name=None, value=None, description=None,
                 aliases=None, **kwargs):
        print_quantity = str
        get_value = lambda x: x
        set_quantity = lambda x: str(x)
        set_uncertainty = lambda x: None

        super(strParameter, self).__init__(name=name, value=value,
                                           description=None, frozen=True,
                                           aliases=aliases,
                                           print_quantity=print_quantity,
                                           set_quantity=set_quantity,
                                           get_value=get_value,
                                           set_uncertainty=set_uncertainty)

        self.paramType = 'strParameter'
        self.value_type = str


class boolParameter(Parameter):
    """This is a Parameter type that is specific to boolean values.
    `.quantity` stores current parameter information in boolean type. `.value`
    returns the same with `.quantity`. `.units` is not applicable.
    `boolParameter` is not fitable.

    Parameter
    ---------
    name : str
        The name of the parameter.
    value : str, bool, [0,1]
        The input parameter boolean value.
    description : str, optional
        A short description of what this parameter means.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example::
        >>> from parameter import boolParameter
        >>> test = boolParameter(name='test1', value='N')
        >>> print test
        test1 N
    """
    def __init__(self, name=None, value=None, description=None, frozen=True,
                 aliases=None, **kwargs):
        print_quantity = lambda x: 'Y' if x else 'N'
        set_quantity = self.set_quantity_bool
        get_value = lambda x: x
        set_uncertainty = lambda x: None
        super(boolParameter, self).__init__(name=name, value=value,
                                            description=None, frozen=True,
                                            aliases=aliases,
                                            print_quantity=print_quantity,
                                            set_quantity=set_quantity,
                                            get_value=get_value,
                                            set_uncertainty=set_uncertainty)
        self.value_type = bool
        self.paramType = 'boolParameter'

    def set_quantity_bool(self, val):
        """ This function is to get boolean value for boolParameter class
        """
        # First try strings
        try:
            if val.upper() in ['Y','YES','T','TRUE','1']:
                return True
            else:
                return False
        except AttributeError:
            # Will get here on non-string types
            return bool(val)

class MJDParameter(Parameter):
    """This is a Parameter type that is specific to MJD values.
    `.quantity` stores current parameter information in an `astropy.Time` type
    in the format of MJD. `.value` returns the pure MJD value. `.units` is in
    day as default unit.

    Parameter
    ---------
    name : str
        The name of the parameter.
    value : astropy Time, str, float in mjd, str in mjd.
        The input parameter MJD value.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool, optional, default True
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.
    time_scale : str, optional, default 'utc'
        MJD parameter time scale.

    Example::
        >>> from parameter import MJDParameter
        >>> test = MJDParameter(name='test1', value='54000', time_scale='utc')
        >>> print test
        test1 (d) 54000.000000000000000
    """
    def __init__(self, name=None, value=None, description=None,
                 uncertainty=None, frozen=True, continuous=True, aliases=None,
                 time_scale='utc', **kwargs):
        self._time_scale = time_scale
        set_quantity = self.set_quantity_mjd
        print_quantity = time_to_mjd_string
        get_value = time_to_longdouble
        set_uncertainty = self.set_uncertainty_mjd
        super(MJDParameter, self).__init__(name=name, value=value, units="MJD",
                                           description=description,
                                           uncertainty=uncertainty,
                                           frozen=frozen,
                                           continuous=continuous,
                                           aliases=aliases,
                                           print_quantity=print_quantity,
                                           set_quantity=set_quantity,
                                           get_value=get_value,
                                           set_uncertainty=set_uncertainty)
        self.value_type = time.Time
        self.paramType = 'MJDParameter'
        self.special_arg += ['time_scale',]

    @property
    def time_scale(self):
        return self._time_scale

    @time_scale.setter
    def time_scale(self, val):
        self._time_scale = val
        mjd = self.value
        self.quantity = mjd

    @property
    def uncertainty_value(self):
        """Return a pure value from .uncertainty. The unit will associate
        with .units
        """
        if self._uncertainty is None:
            return None
        else:
            return self._uncertainty.value

    @uncertainty_value.setter
    def uncertainty_value(self, val):
        """Setter for uncertainty_value. Setting .uncertainty_value will only change
        the .uncertainty attribute.
        """
        if val is None:
            if not isinstance(self.uncertainty, (str, bool)) and \
                self._uncertainty_value is not None:
                log.warning('This parameter has uncertainty value. '
                            'Change it to None will lost information.')
            else:
                self.uncertainty_value = val
        self._uncertainty = self.set_uncertainty(val)

    def set_quantity_mjd(self, val):
        """Value setter for MJD parameter,
           Accepted format:
           Astropy time object
           mjd float
           mjd string
        """
        if isinstance(val, numbers.Number):
            val = numpy.longdouble(val)
            result = time_from_longdouble(val, self.time_scale)
        elif isinstance(val, str):
            try:
                result = time_from_mjd_string(val, self.time_scale)
            except:
                log.error('String ' + val + ' can not be converted to'
                                 'a time object.' )
                raise

        elif isinstance(val,time.Time):
            result = val
        else:
            raise ValueError('MJD parameter can not accept '
                             + type(val).__name__ + 'format.')
        return result

    def set_uncertainty_mjd(self, val):
        # First try to use astropy unit conversion
        try:
            # If this fails, it will raise UnitConversionError
            _ = val.to(self.units)
            result = data2longdouble(val.value) * self.units
        except AttributeError:
            # This will happen if the input value did not have units
            result = data2longdouble(val) * self.units
        return result

    def print_uncertainty(self, uncertainty):
        return longdouble2string(self.uncertainty_value)


class AngleParameter(Parameter):
    """This is a Parameter type that is specific to Angle values.
    `.quantity` stores current parameter information in an `astropy Angle` type.
    `.value` returns the pure angle value associate with default unit.
    `.units` currently can accept angle format  {'h:m:s': u.hourangle,
    'd:m:s': u.deg, 'rad': u.rad, 'deg': u.deg}

    Parameter
    ---------
    name : str
        The name of the parameter.
    value : angle string, float, astropy angle object
        The input parameter angle value.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool, optional, default True
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example::
        >>> from parameter import AngleParameter
        >>> test = AngleParameter(name='test1', value='12:20:10', units='H:M:S')
        >>> print test
        test1 (hourangle) 12:20:10.00000000
    """
    def __init__(self, name=None, value=None, description=None, units='rad',
             uncertainty=None, frozen=True, continuous=True, aliases=None,
             **kwargs):
        self._str_unit = units
        self.unit_identifier = {
            'h:m:s': (u.hourangle, 'h', pint_units['hourangle_second']),
            'd:m:s': (u.deg, 'd', u.arcsec),
            'rad': (u.rad, 'rad', u.rad),
            'deg': (u.deg, 'deg', u.deg),
        }
        # Check unit format
        if units.lower() not in self.unit_identifier.keys():
            raise ValueError('Unidentified unit ' + units)

        self.unitsuffix = self.unit_identifier[units.lower()][1]
        set_quantity = self.set_quantity_angle
        print_quantity = self.print_quantity_angle
        #get_value = lambda x: Angle(x * self.unit_identifier[units.lower()][0])
        get_value = lambda x: x.value
        set_uncertainty = self.set_uncertainty_angle
        self.value_type = Angle
        self.paramType = 'AngleParameter'

        super(AngleParameter, self).__init__(name=name, value=value,
                                             units=units,
                                             description=description,
                                             uncertainty=uncertainty,
                                             frozen=frozen,
                                             continuous=continuous,
                                             aliases=aliases,
                                             print_quantity=print_quantity,
                                             set_quantity=set_quantity,
                                             get_value=get_value,
                                             set_uncertainty=set_uncertainty)

    def set_quantity_angle(self, val):
        """ This function is to set value to angle parameters.
        Accepted format:
        1. Astropy angle object
        2. float
        3. number string
        """
        if isinstance(val, numbers.Number):
            result = Angle(data2longdouble(val) * self.units)
        elif isinstance(val, str):
            result = Angle(val + self.unitsuffix)
        elif hasattr(val, 'unit'):
            result = Angle(val.to(self.units))
        else:
            raise ValueError('Angle parameter can not accept '
                             + type(val).__name__ + 'format.')
        return result

    def set_uncertainty_angle(self, val):
        """This function is to set the uncertainty for an angle parameter.
        """
        if isinstance(val, numbers.Number):
            result =Angle(val * self.unit_identifier[self._str_unit.lower()][2])
        elif isinstance(val, str):

            result =Angle(str2longdouble(val) * \
                          self.unit_identifier[self._str_unit.lower()][2])
            #except:
            #    raise ValueError('Srting ' + val + ' can not be converted to'
            #                     ' astropy angle.')
        elif hasattr(val, 'unit'):
            result = Angle(val.to(self.unit_identifier[self._str_unit.lower()][2]))
        else:
            raise ValueError('Angle parameter can not accept '
                             + type(val).__name__ + 'format.')
        return result

    def print_quantity_angle(self, quan):
        """This is a function to print out the angle parameter.
        """
        if ':' in self._str_unit:
            return quan.to_string(sep=':', precision=8)
        else:
            return quan.to_string(decimal = True, precision=15)

    def print_uncertainty(self, unc):
        """This is a function for printing out the uncertainty
        """
        if ':' in self._str_unit:
            angle_arcsec = unc.to(u.arcsec)
            if self.units == u.hourangle:
                # Triditionaly hourangle uncertainty is in hourangle seconds
                angle_arcsec  /= 15.0
            return angle_arcsec.to_string(decimal = True, precision=20)
        else:
            return unc.to_string(decimal = True, precision=20)


class prefixParameter(object):
    """ This is a Parameter type for prefix parameters, for example DMX_
    Create a prefix parameter, is like create a normal parameter. But the
    name should be in the format of prefix and index. For example DMX_0001 or
    F22.
    To create a prefix parameter with the same prefix but different index, just
    use the `.new_param` method. It will return a new prefix parameter with the
    same setup but the index. Some parameters' unit and description will
    be changed once the index has been changed. In order to get the right units
    and description, `.unitTplt` and `.descriptionTplt` should be provided. If
    not the new prefix parameter will use the same units and description with
    the old one. A typical description and units template is like:
    >>> descritionTplt = lambda x: 'This is the description of parameter %d'%x
    >>> unitTplt = lambda x: 'second^%d'%x
    Parameter
    ---------
    name : str optional
        The name of the parameter. It has to be in the format of prefix + index.
    units :  str optional
        The unit of parameter
    unitTplt : lambda method
        The unit template for prefixed parameter
    description : str optional
        Description for the parameter
    descriptionTplt : lambda method optional
        Description template for prefixed parameters
    prefix_aliases : list of str optional
        Alias for the prefix
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool
    parameter_type : str, optional, default 'float'
        Example parameter class template for quantity and value setter
    long_double : bool, optional default 'double'
        Set float type quantity and value in numpy float128
    time_scale : str, optional default 'utc'
        Time scale for MJDParameter class.
    """
    def __init__(self, parameter_type='float',name=None, value=None, units=None,
                 unit_template=None, description=None, description_template=None,
                 uncertainty=None, frozen=True, continuous=True,
                 prefix_aliases=None, long_double=False, unit_scale=False, \
                 scale_factor=None, scale_threshold=None,  time_scale='utc',
                 **kwargs):
        # Split prefixed name, if the name is not in the prefixed format, error
        # will be raised
        self.name = name
        self.prefix, self.idxfmt, self.index = split_prefixed_name(name)
        # Type identifier
        self.type_mapping = {'float': floatParameter, 'str': strParameter,
                             'bool': boolParameter, 'mjd': MJDParameter,
                             'angle': AngleParameter}
        self.parameter_type = parameter_type
        try:
            self.param_class = self.type_mapping[self.parameter_type.lower()]
        except KeyError:
            raise ValueError("Unknow parameter type '"+ parameter_type + "' ")

        # Set up other attributes in the wrapper class
        self.unit_template = unit_template
        self.description_template = description_template
        input_units = units
        input_description = description
        self.prefix_aliases = [] if prefix_aliases is None else prefix_aliases
        # set templates, the templates should be a lambda function and input is
        # the index of prefix parameter.
        if self.unit_template is None:
            self.unit_template = lambda x: input_units
        if self.description_template is None:
            self.description_template = lambda x: input_description

        # Set the description and units for the parameter compostion.
        real_units = self.unit_template(self.index)
        real_description = self.description_template(self.index)
        aliases = []
        for pa in self.prefix_aliases:
            aliases.append(pa + self.idxfmt)
        self.long_double = long_double
        # initiate parameter class
        self.param_comp = self.param_class(name=self.name, value=value,
                                           units=real_units,
                                           description=real_description,
                                           uncertainty=uncertainty,
                                           frozen=frozen,
                                           continuous=continuous,
                                           aliases=aliases,
                                           long_double=long_double,
                                           time_scale=time_scale,
                                           unit_scale=unit_scale, \
                                           scale_factor=scale_factor,\
                                           scale_threshold=scale_threshold)
        self.is_prefix = True
    # Define prpoerties for access the parameter composition
    @property
    def units(self):
        return self.param_comp.units

    @units.setter
    def units(self, unt):
        self.param_comp.units = unt

    @property
    def quantity(self):
        return self.param_comp.quantity

    @quantity.setter
    def quantity(self, qnt):
        self.param_comp.quantity = qnt

    @property
    def value(self):
        return self.param_comp.value

    @value.setter
    def value(self, val):
        self.param_comp.value = val

    @property
    def uncertainty(self):
        return self.param_comp.uncertainty

    @uncertainty.setter
    def uncertainty(self, ucty):
        self.param_comp.uncertainty = ucty

    @property
    def uncertainty_value(self):
        return self.param_comp.uncertainty_value

    @uncertainty_value.setter
    def uncertainty_value(self, val):
        self.param_comp.uncertainty_value = val

    @property
    def prior(self):
        return self.param_comp.prior

    @prior.setter
    def prior(self,p):
        self.param_comp.prior = p

    @property
    def aliases(self):
        return self.param_comp.aliases

    @aliases.setter
    def aliases(self,a):
        self.param_comp.aliases = a

    @property
    def continuous(self):
        return self.param_comp.continuous

    @continuous.setter
    def continuous(self, val):
        self.param_comp.continuous = val

    @property
    def frozen(self):
        return self.param_comp.frozen

    @frozen.setter
    def frozen(self, val):
        self.param_comp.frozen = val

    @property
    def description(self):
        return self.param_comp.description

    @description.setter
    def description(self, val):
        self.param_comp.description = val

    @property
    def special_arg(self):
        return self.param_comp.special_arg

    def __str__(self):
        out = self.name
        if self.units is not None:
            out += " (" + str(self.units) + ")"
        out += " " + self.print_quantity(self.quantity)
        if self.uncertainty is not None:
            out += " +/- " + str(self.uncertainty.to(self.units))
        return out

    # Define the function to call functions inside of parameter composition.
    def __str__(self):
        return self.param_comp.__str__()

    def from_parfile_line(self, line):
        return self.param_comp.from_parfile_line(line)

    def prior_pdf(self,value=None, logpdf=False):
        return self.param_comp.prior_pdf(value, logpdf)

    def print_quantity(self, quantity):
        return self.param_comp.print_quantity(quantity)

    def name_matches(self, name):
        return self.param_comp.name_matches(name)

    def as_parfile_line(self):
        return self.param_comp.as_parfile_line()

    def help_line(self):
        return self.param_comp.help_line()

    def prefix_matches(self, prefix):
        return (prefix == self.perfix) or (prefix in self.prefix_aliases)

    def new_param(self, index):
        """Get one prefix parameter with the same type.
        Parameter
        ----------
        index : int
            index of prefixed parameter.
        Return
        ----------
        A prefixed parameter with the same type of instance.
        """

        new_name = self.prefix + format(index, '0'+ str(len(self.idxfmt)))
        kws = dict()
        for key in ['units', 'unit_template', 'description','description_template',
                    'frozen', 'continuous', 'prefix_aliases', 'long_double',
                    'time_scale', 'parameter_type']:
            if hasattr(self, key):
                kws[key] = getattr(self, key)

        newpfx = prefixParameter(name=new_name, **kws)
        return newpfx


class maskParameter(floatParameter):
    """ This is a Parameter type for mask parameters which is to select a
    certain subset of TOAs and apply changes on the subset of TOAs, for example
    JUMP. This type of parameter does not require index input. But eventrully
    an index part will be added, for the purpose of parsing the right value
    from the parfile. For example,
    >>> p = maskParameter(name='JUMP', index=2)
    >>> p.name
    'JUMP2'
    Parameter
    ---------
    name : str optional
        The name of the parameter.
    index : int optional [default 1]
        The index number for the prefixed parameter.
    key : str optional
        The key words/flag for the selecting TOAs
    key_value :  list/single value optional
        The value for key words/flags. Value can take one value as a flag value.
        or two value as a range.
        e.g. JUMP freq 430.0 1440.0. or JUMP -fe G430
    value : float or long_double optinal
        Toas/phase adjust value
    long_double : bool, optional default 'double'
        Set float type quantity and value in numpy float128
    units : str optional
        Unit for the offset value
    description : str optional
        Description for the parameter
    uncertainty: float/longdouble
        uncertainty of the parameter.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool optional
    aliases : list optional
        List of aliases for parameter name.

    TODO: Is mask parameter provide some other type of parameters other then
    floatParameter?
    """
    def __init__(self, name=None, index=1, key=None, key_value=None,
                 value=None, long_double=False, units= None, description=None,
                 uncertainty=None, frozen=True, continuous=False, aliases=[]):
        self.is_mask = True
        self.key_identifier = {'mjd': (lambda x: time.Time(x, format='mjd').mjd, 2),
                                'freq': (float, 2),
                                'name': (str, 1),
                                'tel': (str, 1)}
        if key_value is None:
            key_value = []
        elif not isinstance(key_value, list):
            key_value = [key_value]

        # Check key and key value
        if key is not None \
            and key.lower() in self.key_identifier.keys():
            key_info = self.key_identifier[key.lower()]
            if len(key_value) != key_info[1]:
                errmsg = "key " + key + " takes " + key_info[1] + \
                         " element."
                raise ValueError(errmsg)

        self.key = key
        self.key_value = key_value
        self.index = index
        name_param = name + str(index)
        self.origin_name = name
        self.prefix = self.origin_name
        # Make aliases with index.
        idx_aliases = []
        for al in aliases:
            idx_aliases.append(al + str(self.index))
        self.prefix_aliases = aliases
        super(maskParameter, self).__init__(name=name_param, value=value,
                                            units=units,
                                            description=description,
                                            uncertainty=uncertainty,
                                            frozen=frozen,
                                            continuous=continuous,
                                            aliases=idx_aliases,
                                            long_double=long_double)

        # For the first mask parameter, add name to aliases for the reading
        # first mask parameter from parfile.
        if index == 1:
            self.aliases.append(name)
        self.from_parfile_line = self.from_parfile_line_mask
        self.as_parfile_line = self.as_parfile_line_mask
        self.is_prefix = True

    def __str__(self):
        out = self.name
        if self.units is not None:
            out += " (" + str(self.units) + ")"

        out += " " + self.key
        for kv in self.key_value:
            out += " " + str(kv)
        if self.quantity is not None:
            out += " " + self.print_quantity(self.quantity)
        else:
            out += " " + "UNSET"
            return out
        if self.uncertainty is not None and isinstance(self.value, numbers.Number):
            out += " +/- " + str(self.uncertainty.to(self.units))
        return out

    def name_matches(self, name):
        if super(maskParameter, self).name_matches(name):
            return True
        else:
            name_idx = name + str(self.index)
            return super(maskParameter, self).name_matches(name_idx)

    def from_parfile_line_mask(self, line):
        """
        This is a method to read mask parameter line (e.g. JUMP)
        Notes
        -----
        The accepted format:
            NAME key key_value parameter_value
            NAME key key_value parameter_value fit_flag
            NAME key key_value parameter_value fit_flag uncertainty
            NAME key key_value parameter_value uncertainty
            NAME key key_value1 key_value2 parameter_value
            NAME key key_value1 key_value2 parameter_value fit_flag
            NAME key key_value1 key_value2 parameter_value fit_flag uncertainty
            NAME key key_value1 key_value2 parameter_value uncertainty
        """
        try:
            k = line.split()
            name = k[0].upper()
        except IndexError:
            return False
        # Test that name matches
        if not self.name_matches(name):
            return False

        try:
            self.key = k[1]
        except IndexError:
            return False


        key_value_info = self.key_identifier.get(self.key.lower(), (str, 1))
        len_key_v = key_value_info[1]
        if len(k) < 3 + len_key_v:
            return False

        for ii in range(len_key_v):
            if key_value_info[0] != str:
                try:
                    kval = float(k[2 + ii])
                except:
                    kval = k[2 + ii]
            else:
                kval = k[2 + ii]
            if ii > len(self.key_value)-1:
                self.key_value.append(key_value_info[0](kval))
            else:
                self.key_value[ii] = key_value_info[0](kval)
        if len(k) >= 3 + len_key_v:
            self.set(k[2 + len_key_v])
        if len(k) >= 4 + len_key_v:
            try:
                fit_flag =  int(k[3 + len_key_v])
                if fit_flag == 0:
                    self.frozen = True
                    ucty = 0.0
                elif fit_flag == 1:
                    self.frozen = False
                    ucty = 0.0
                else:
                    ucty = fit_flag
            except:
                if is_number(k[3 + len_key_v]):
                    ucty = k[3 + len_key_v]
                else:
                    errmsg = 'Unidentified string ' + k[3 + len_key_v] + ' in'
                    errmsg += ' parfile line ' + k
                    raise ValueError(errmsg)

            if len(k) >= 5 + len_key_v:
                ucty = k[4 + len_key_v]
            self.uncertainty = self.set_uncertainty(ucty)
        return True

    def as_parfile_line_mask(self):
        if self.quantity is None:
            return ""
        line = "%-15s %s " % (self.origin_name, self.key)
        for kv in self.key_value:
            if not isinstance(kv, time.Time):
                line += "%s " % kv
            else:
                line += "%s " % time_to_mjd_string(kv)
        line += "%25s" % self.print_quantity(self.quantity)
        if self.uncertainty is not None:
            line += " %d %s" % (0 if self.frozen else 1, str(self.uncertainty_value))
        elif not self.frozen:
            line += " 1"
        return line + "\n"

    def new_param(self, index):
        """Create a new but same style mask parameter
        """
        new_mask_param = maskParameter(name=self.origin_name, index=index,
                                       long_double=self.long_double,
                                       units= self.units,
                                       aliases=self.prefix_aliases)
        return new_mask_param

    def select_toa_mask(self, toas):
        """Select the toas.
        Parameter
        ---------
        toas : TOAs class
        Return
        ------
        A array of returned index.
        """
        column_match = {'mjd': 'mjd_float',
                        'freq': 'freq',
                        'tel': 'obs'}
        if len(self.key_value) == 1:
            if not hasattr(self, 'toa_selector'):
                self.toa_selector = TOASelect(is_range=False, use_hash=True)
            condition = {self.name: self.key_value[0]}
        elif len(self.key_value) == 2:
            if not hasattr(self, 'toa_selector'):
                self.toa_selector = TOASelect(is_range=True, use_hash=True)
            condition = {self.name: tuple(self.key_value)}
        else:
            raise ValueError('Parameter %s has more key values than '
                             'expected.(Expect 1 or 2 key values)' % self.name)
        # get the table columns
        # TODO Right now it is only supports mjd, freq, tel, and flagkeys,
        # We need to consider some more complicated situation
        key = self.key.replace('-', '')
        tbl = toas.table
        if key not in column_match.keys(): # This only works for the one with flags.
            section_name = key+'_section'
            if section_name not in tbl.keys():
                flag_col = [x.get(key, None) for x in tbl['flags']]
                tbl[section_name] = flag_col
            col = tbl[section_name]
        else:
            col = tbl[column_match[key]]
        select_idx = self.toa_selector.get_select_index(condition, col)

        return select_idx[self.name]
