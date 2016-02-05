# parameter.py
# Defines Parameter class for timing model parameters
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string,\
    time_to_longdouble, is_number
import numpy
import astropy.units as u
from astropy import log
from pint import pint_units
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
import numbers
import priors

class Parameter(object):
    """A PINT class describing a single timing model parameter. The parameter
    value will be stored at `value` property in a users speicified format. At
    the same time property `num_value` will store a num value from the value and
    `num_unit` will store the basic unit in the format of `Astropy.units`

    Parameters
    ----------
    name : str, optional
        The name of the parameter.
    value : number, str, `Astropy.units.Quantity` object, or other datatype or object
        The current value of parameter. It is the internal storage of
        parameter value
    units : str, optional
        String format for parameter unit
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
    parse_value : method, optional
        A function that converts string input into the appropriate internal
        representation of the parameter (typically floating-point but could be any datatype).
    print_value : method, optional
        A function that converts the internal value to a string for output.
    get_value : method, optional
        A function that converts num_value attribute and num_unit to value
        attribute
    get_num_value:
        A function that get purely value from value attribute
    """

    def __init__(self, name=None, value=None, units=None, description=None,
            uncertainty=None, frozen=True, aliases=None, continuous=True,
            prior=priors.Prior(),
            parse_value=fortran_float, print_value=str, get_value=lambda x: x,
            get_num_value=lambda x: x):
        self.name = name  # name of the parameter
        self.units = units # parameter unit in string format,or None
        # parameter num unit, in astropy.units object format.
        # Once it is speicified, num_unit will not be changed.

        self.get_num_value = get_num_value # Method to get num_value from value
        self.get_value = get_value # Method to update value from num_value
        self.value = value # The value of parameter, internal storage
        
        self.prior = prior

        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = [] if aliases is None else aliases
        self.parse_value = parse_value # method to read a value from string,
                                       # user can put the speicified format here
        self.print_value = print_value # method to convert value to a string.
        self.parse_uncertainty = fortran_float
        self.paramType = 'Parameter' # Type of parameter. Here is general type
    # Setup units property
    @property
    def units(self):
        return self._units
    @units.setter
    def units(self,unt):
        # Setup unit and num unit
        if isinstance(unt,(u.Unit,u.CompositeUnit)):
            self._units = unt.to_string()
            self._num_unit = unt
        elif isinstance(unt,(str)):
            if unt in pint_units.keys():
                self._units = unt
                self._num_unit = pint_units[unt]
            else:
                self._units = unt
                self._num_unit = u.Unit(self._units)
        elif unt is None:
            self._units = unt
            self._num_unit = unt

        else:
            raise ValueError('Units can only take string, astropy units or None')

        # Check if this is the first time set units
        if hasattr(self,'value'):
            wmsg = 'Parameter '+self.name+'units has been reset to '+unt
            log.warning(wmsg)
            try:
                if hasattr(self.value,'unit'):
                    temp = self.value.to(self.num_unit)
            except:
                log.warning('The value unit is not compatable with'\
                                 ' parameter units,right now.')
    # Setup value property
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self,val):
        self._value = val
        if hasattr(self._value,'unit'):  # If the new value is astropy angle or Quantity
            if self._num_unit is not None: # Check unit
                try:
                    value_num_unit = self._value.to(self._num_unit)
                except:
                    raise ValueError('The value unit is not compatable with'\
                                     ' parameter units.')
                self._num_value = value_num_unit.value
            else:
                self.unit = self._value.unit.to_string()
                self._num_value = self._value.value
        elif isinstance(self._value,(str,bool)) or self._value is None:
            self._num_value = None
        else:
            self._num_value = self.get_num_value(self._value)
            if not isinstance(self._num_value, numbers.Number):
                if not self._num_value is not None:
                    raise ValueError("The ._num_value has to be a pure number or None. "\
                                     "Please check your .get_num_value method. ")

    def prior_probability(self,value=None):
        """Return the prior probability, evaluated at the current value of
        the parameter, or at a proposed value.
        
        Probabilities are evaluated using the num_value attribute
        """
        if value is None:
            return self.prior.prior_probability(self.num_value)
        else:
            return self.prior.prior_probability(value)
            
    # Setup num_value property
    @property
    def num_value(self):
        return self._num_value
    @num_value.setter
    def num_value(self,val):
        if val is None:
            self._num_value = val
            if not isinstance(self.value,(str,bool)):
                raise ValueError('This parameter value is number convertable. '\
                                 'Setting ._num_value to None will lost the ' \
                                 'parameter value.' )
            else:
                self.value = None

        elif not isinstance(val,numbers.Number):
            raise ValueError('num_value has to be a pure number or None.')
        else:
            self._num_value = val
            # Update value
            if self.get_value is None:
                self.value = self._num_value*self.num_unit
            else:
                self.value = self.get_value(self._num_value)

    # Setup num_unit property
    @property
    def num_unit(self):
        return self._num_unit

    def __str__(self):
        out = self.name
        if self.units is not None:
            out += " (" + str(self.units) + ")"
        out += " " + self.print_value(self.value)
        if self.uncertainty is not None:
            out += " +/- " + str(self.uncertainty)
        return out

    def set(self, value):
        """Parses a string 'value' into the appropriate internal representation
        of the parameter.
        """
        self.value = self.parse_value(value)

    def add_alias(self, alias):
        """Add a name to the list of aliases for this parameter."""
        self.aliases.append(alias)

    def help_line(self):
        """Return a help line containing param name, description and units."""
        out = "%-12s %s" % (self.name, self.description)
        if self.units is not None:
            out += ' (' + str(self.units) + ')'
        return out

    def as_parfile_line(self):
        """Return a parfile line giving the current state of the parameter."""
        # Don't print unset parameters
        if self.value is None:
            return ""
        line = "%-15s %25s" % (self.name, self.print_value(self.value))
        if self.uncertainty is not None:
            line += " %d %s" % (0 if self.frozen else 1, str(self.uncertainty))
        elif not self.frozen:
            line += " 1"
        return line + "\n"

    def from_parfile_line(self, line):
        """
        Parse a parfile line into the current state of the parameter.
        Returns True if line was successfully parsed, False otherwise.
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
        if len(k) >= 3:  # Fixed a bug here. It can read parfile third column as uncertainty
            try:
                if int(k[2]) > 0:
                    self.frozen = False
                    ucty = '0.0'
            except:
                if is_number(k[2]):
                    ucty = k[2]
                else:
                    errmsg = 'The third column of parfile can only be fitting flag '
                    errmsg+= '(1/0) or uncertainty.'
                    raise ValueError(errmsg)
            if len(k) == 4:
                ucty = k[3]
            self.uncertainty = self.parse_uncertainty(ucty)
        return True

    def name_matches(self, name):
        """Whether or not the parameter name matches the provided name
        """
        return (name == self.name) or (name in self.aliases)


class MJDParameter(Parameter):
    """This is a Parameter type that is specific to MJD values."""
    def __init__(self, name=None, value=None, description=None,
                 uncertainty=None, frozen=True, continuous=True, aliases=None,
                 time_scale='utc'):
        super(MJDParameter, self).__init__(name=name, value=value,
                units="MJD", description=description,
                uncertainty=uncertainty, frozen=frozen,
                continuous=continuous,
                aliases=aliases)

        self.parse_value = lambda x: time_from_mjd_string(x, time_scale)
        self.print_value = time_to_mjd_string
        self.get_value = lambda x: longdouble_from_mjd_string(x, time_scale)
        self.get_num_value = time_to_longdouble
        self.paramType = 'MJDParameter'


class AngleParameter(Parameter):
    """This is a Parameter type that is specific to Angle values."""
    def __init__(self, name=None, value=None, description=None, units='rad',
                 uncertainty=None, frozen=True, continuous=True, aliases=None):
        super(AngleParameter, self).__init__(name=name, value=value,
                units=units, description=description, uncertainty=uncertainty,
                frozen=frozen, continuous=continuous, aliases=aliases)

        self.separator = {
            'h:m:s': (u.hourangle, 'h', '0:0:%.15fh'),
            'd:m:s': (u.deg, 'd', '0:0:%.15fd'),
            'rad': (u.rad, 'rad', '%.15frad'),
            'deg': (u.deg, 'deg', '%.15fdeg'),
        }
        # Check unit format
        if self.units.lower() not in self.separator.keys():
            raise ValueError('Unidentified unit ' + self.units)

        unitsuffix = self.separator[self.units.lower()][1]
        self.parse_value = lambda x: Angle(x+unitsuffix)
        self.print_value = lambda x: x.to_string(sep=':', precision=8) \
                           if x.unit != u.rad else x.to_string(decimal = True,
                           precision=8)
        self.get_value = lambda x: Angle(x * self.separator[units.lower])
        self.get_num_value = lambda x: x.value
        self.parse_uncertainty = lambda x: \
                                Angle(self.separator[self.units.lower()][2] \
                                      % fortran_float(x))
        self.paramType = 'AngleParameter'
