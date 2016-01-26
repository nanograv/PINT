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

class Parameter(object):
    """A PINT class describing a single timing model parameter. The parameter
    value will be stored at `value` property in a users speicified format. At
    the same time property `base_value` will store a base value from the value and
    `base_unit` will store the basic unit in the format of `Astropy.units`

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
        A function that converts base_value attribute and base_unit to value
        attribute
    get_base_value:
        A function that get purely value from value attribute
    """

    def __init__(self, name=None, value=None, units=None, description=None,
            uncertainty=None, frozen=True, aliases=None, continuous=True,
            parse_value=fortran_float, print_value=str,get_value = None,
            get_base_value=lambda x: x):
        self.name = name  # name of the parameter
        self.units = units # parameter unit in string format,or None
        # parameter base unit, in astropy.units object format.
        # Once it is speicified, base_unit will not be changed.

        self.get_base_value = get_base_value # Method to get base_value from value
        self.get_value = get_value # Method to update value from base_value
        self.value = value # The value of parameter, internal storage


        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = [] if aliases is None else aliases
        self.parse_value = parse_value # method to read a value from string,
                                       # user can put the speicified format here
        self.print_value = print_value # method to convert value to a string.
        self.paramType = 'Parameter' # Type of parameter. Here is general type
    # Setup units property
    @property
    def units(self):
        return self._units
    @units.setter
    def units(self,unt):
        # Setup unit and base unit
        if isinstance(unt,(u.Unit,u.CompositeUnit)):
            self._units = unt.to_string()
            self._base_unit = unt
        elif isinstance(unt,(str)):
            if unt in pint_units.keys():
                self._units = unt
                self._base_unit = pint_units[unt]
            else:
                self._units = unt
                self._base_unit = u.Unit(self._units)
        elif unt is None:
            self._units = unt
            self._base_unit = unt

        else:
            raise ValueError('Units can only take string, astropy units or None')

        # Check if this is the first time set units
        if hasattr(self,'value'):
            wmsg = 'Parameter '+self.name+'units has been reset to '+unt
            log.warning(wmsg)
            try:
                if hasattr(self.value,'unit'):
                    temp = self.value.to(self.base_unit)
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
        if self._value is None:
            self._base_value = None
            return

        # Check units
        if hasattr(self._value,'unit'):
            if self.units is None:
                self.units = self._value.unit.to_string()
                value_base_unit = self._value
            else:
                try:
                    value_base_unit = self._value.to(self._base_unit)
                except:
                    raise ValueError('The value unit is not compatable with'\
                                     ' parameter units.')

            self._base_value = value_base_unit.value
        else:
            self._base_value = self.get_base_value(self._value)


    # Setup base_value property
    @property
    def base_value(self):
        return self._base_value
    @base_value.setter
    def base_value(self,val):
        self._base_value = val
        if not isinstance(val, numbers.Number):
            return
        if self.get_value is None:
            self.value = self._base_value*self.base_unit
        else:
            self.value = self.get_value(self._base_value)

    # Setup base_unit property
    @property
    def base_unit(self):
        return self._base_unit

    def __str__(self):
        out = self.name
        if self.units is not None:
            out += " (" + str(self.units) + ")"
        out += " " + self.print_value(self.value)
        if self.uncertainty is not None:
            out += " +/- " + str(self.uncertainty)
        return out

    def set(self, value, with_unit = False):
        """Parses a string 'value' into the appropriate internal representation
        of the parameter.
        """
        self.value = self.parse_value(value)
        if with_unit is True:
            self.value = self.value*u.Unit(self.units)

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
            if name=="RAJ":
                self.uncertainty = Angle("0:0:%.15fh" % fortran_float(ucty))
            elif name=="DECJ":
                self.uncertainty = Angle("0:0:%.15fd" % fortran_float(ucty))
            else:
                self.uncertainty = fortran_float(ucty)
        return True
    def name_matches(self, name):
        """Whether or not the parameter name matches the provided name
        """
        return (name == self.name) or (name in self.aliases)

class MJDParameter(Parameter):
    """This is a Parameter type that is specific to MJD values."""
    def __init__(self, name=None, value=None, description=None,
            uncertainty=None, frozen=True, continuous=True, aliases=None,
            parse_value=time_from_mjd_string,
            print_value=time_to_mjd_string,
            get_value =lambda x: longdouble_from_mjd_string(x,'utc'),
            get_base_value = time_to_longdouble):
        super(MJDParameter, self).__init__(name=name, value=value,
                units="MJD", description=description,
                uncertainty=uncertainty, frozen=frozen,
                continuous=continuous,
                aliases=aliases,
                parse_value=parse_value,
                print_value=print_value,
                get_base_value = get_base_value)
        self.paramType = 'MJDParameter'
