# parameter.py
# Defines Parameter class for timing model parameters
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string,\
    time_to_longdouble, is_number, time_from_longdouble
import numpy
import astropy.units as u
from astropy import log
from pint import pint_units
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
import re
import numbers
import priors


class Parameter(object):
    """A PINT class describing a single timing model parameter. The parameter
    value will be stored at `value` property in a users speicified format. At
    the same time property `num_value` will store a num value from the value
    and `num_unit` will store the basic unit in the format of `Astropy.units`

    Parameters
    ----------
    name : str, optional
        The name of the parameter.
    value : number, str, `Astropy.units.Quantity` object, or other datatype or
            object
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
        representation of the parameter (typically floating-point but could be
        any datatype).
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
                 prior=priors.Prior(priors.UniformPrior()),
                 parse_value=fortran_float, print_value=str,
                 get_value=lambda x: x, get_num_value=lambda x: x):
        self.name = name  # name of the parameter
        self.units = units  # parameter unit in string format,or None
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

        self.is_prefix = False
        self.parse_value = parse_value  # method to read a value from string,
                                        # user can put the speicified format
                                        # here
        self.print_value = print_value  # method to convert value to a string.
        self.parse_uncertainty = fortran_float
        self.paramType = 'Parameter'  # Type of parameter. Here is general type

    # Setup units property
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unt):
        # Check if this is the first time set units
        if hasattr(self, 'value'):
            if self.units is None:
                return
            wmsg = 'Parameter '+self.name+' units has been reset to '+unt
            log.warning(wmsg)
            try:
                if hasattr(self.value, 'unit'):
                    temp = self.value.to(self.num_unit)
            except:
                log.warning('The value unit is not compatable with'
                            ' parameter units,right now.')
        # Setup unit and num unit
        if isinstance(unt, (u.Unit, u.CompositeUnit)):
            self._units = unt.to_string()
            self._num_unit = unt
        elif isinstance(unt, (str)):
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
            raise ValueError('Units can only take string, astropy units or'
                             ' None')

    # Setup value property
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        # If the new value is astropy angle or Quantity
        if hasattr(self._value, 'unit'):
            if self._num_unit is not None:  # Check unit
                try:
                    value_num_unit = self._value.to(self._num_unit)
                except:
                    raise ValueError('The value unit is not compatable with'
                                     ' parameter units.')
                self._num_value = value_num_unit.value
            else:
                self.unit = self._value.unit.to_string()
                self._num_value = self._value.value
        elif isinstance(self._value, (str, bool)) or self._value is None:
            self._num_value = None
        else:
            self._num_value = self.get_num_value(self._value)
            if not isinstance(self._num_value, numbers.Number):
                if not self._num_value is not None:
                    raise ValueError("The ._num_value has to be a pure number "
                                     "or None. Please check your .get_num_value"
                                     " method. ")

    def prior_pdf(self,value=None, logpdf=False):
        """Return the prior probability, evaluated at the current value of
        the parameter, or at a proposed value.
        
        Parameters
        ----------
        value : array_like or float_like
        
        Probabilities are evaluated using the num_value attribute
        """
        if value is None:
            return self.prior.pdf(self.num_value) if not logpdf else self.prior.logpdf(self.num_value)
        else:
            return self.prior.pdf(value) if not logpdf else self.prior.logpdf(value)
            
    # Setup num_value property
    @property
    def num_value(self):
        return self._num_value

    @num_value.setter
    def num_value(self, val):
        if val is None:
            self._num_value = val
            if not isinstance(self.value, (str, bool)):
                raise ValueError('This parameter value is number convertable. '
                                 'Setting ._num_value to None will lost the '
                                 'parameter value.')
            else:
                self.value = None

        elif not isinstance(val, numbers.Number):
            raise ValueError('num_value has to be a pure number or None. ({0} <- {1} ({2})'.format(self.name,val,type(val)))
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
        if len(k) >= 3:
            try:
                if int(k[2]) > 0:
                    self.frozen = False
                    ucty = '0.0'
            except:
                if is_number(k[2]):
                    ucty = k[2]
                else:
                    errmsg = 'The third column of parfile can only be fitting '
                    errmsg += 'flag (1/0) or uncertainty.'
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
        self.get_value = lambda x: time_from_longdouble(x, time_scale)
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
        self.get_value = lambda x: Angle(x * self.separator[units.lower()][0])
        self.get_num_value = lambda x: x.value
        self.parse_uncertainty = lambda x: \
                                Angle(self.separator[self.units.lower()][2] \
                                      % fortran_float(x))
        self.paramType = 'AngleParameter'


class prefixParameter(Parameter):
    """ This is a Parameter type for prefix parameters, for example DMX_

        Create a prefix parameter
        To create a prefix parameter, there are two ways:
        1. Create by name
            If optional agrument name with a prefixed format, such as DMX_001
            or F10, is given. prefixParameter class will figure out the prefix
            name, index and indexformat.
        2. Create by prefix and index
            This method allows you create a prefixParameter class using prefix
            name and index. The class name will be returned as prefix name plus
            index with the right index format. So the optional arguments
            prefix, indexformat and index are need. index default value is 1.
        If both of two methods are fillfulled, It will using the first method.
        Add descrition and units.
        1. Direct add
            A descrition and unit can be added directly by using the optional
            arguments, descrition and units. Both of them will return as a
            string attribution.
        2. descrition and units template.
            If the descrition and unit are changing with the prefix parameter
            index, optional argurment descritionTplt and unitTplt are need.
            These two attributions are lambda functions, for example
            >>> descritionTplt = lambda x: 'This is the descrition of parameter
                                            %d'%x
            The class will fill the descrition and unit automaticly.
        If both two methods are fillfulled, it prefer the first one.

        Parameter
        ---------
        name : str optional
            The name of the parameter. If it is not provided, the prefix and
            index format are needed.
        prefix : str optional
            Paremeter prefix, now it is only supporting 'prefix_' type and
            'prefix0' type.
        indexformat : str optional
            The format for parameter index
        index : int optional [default 1]
            The index number for the prefixed parameter.
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
    """

    def __init__(self, name=None, prefix=None, indexformat=None, index=1,
                 value=None, units=None, unitTplt=None,
                 description=None, descriptionTplt=None,
                 uncertainty=None, frozen=True, continuous=True,
                 prefix_aliases=[], parse_value=fortran_float,
                 print_value=str, get_value=None, get_num_value=lambda x: x):
        # Create prefix parameter by name
        if name is None:
            if prefix is None or indexformat is None:
                errorMsg = 'When prefix parameter name is not give, the prefix'
                errorMsg += 'and indexformat are both needed.'
                raise ValueError(errorMsg)
            else:
                # Get format fields
                digitLen = 0
                for i in range(len(indexformat)-1, -1, -1):
                    if indexformat[i].isdigit():
                        digitLen += 1
                self.indexformat_field = indexformat[0:len(indexformat)
                                                     - digitLen]
                self.indexformat_field += '{0:0' + str(digitLen) + 'd}'

                name = prefix+self.indexformat_field.format(index)
                self.prefix = prefix
                self.indexformat = self.indexformat_field.format(0)
                self.index = index
        else:  # Detect prefix and indexformat from name.
            namefield = re.split('(\d+)', name)
            if len(namefield) < 2 or namefield[-2].isdigit() is False\
               or namefield[-1] != '':
            #When Name has no index in the end or no prefix part.
                errorMsg = 'Prefix parameter name needs a perfix part'\
                           + ' and an index part in the end. '
                errorMsg += 'If you meant to set up with prefix, please use' \
                            + 'prefix and indexformat optional agruments.' \
                            + 'Leave name argument alone.'
                raise ValueError(errorMsg)
            else:  # When name has index in the end and prefix in front.
                indexPart = namefield[-2]
                prefixPart = namefield[0:-2]
                self.indexformat_field = '{0:0' + str(len(indexPart)) + 'd}'
                self.indexformat = self.indexformat_field.format(0)
                self.prefix = ''.join(prefixPart)
                self.index = int(indexPart)
        self.unit_template = unitTplt
        self.description_template = descriptionTplt
        # set templates
        if self.unit_template is None:
            self.unit_template = lambda x: self.units
        if self.description_template is None:
            self.description_template = lambda x: self.descrition

        super(prefixParameter, self).__init__(name=name, value=value,
                                              units=units,
                                              description=description,
                                              uncertainty=uncertainty,
                                              frozen=frozen,
                                              continuous=continuous,
                                              parse_value=parse_value,
                                              print_value=print_value,
                                              get_value=get_value,
                                              get_num_value=get_num_value)

        if units == 'MJD':
            self.parse_value = time_from_mjd_string
            self.print_value = time_to_mjd_string
            self.get_num_value = time_to_longdouble
        self.prefix_aliases = prefix_aliases
        self.is_prefix = True

    def prefix_matches(self, prefix):
        return (prefix == self.perfix) or (prefix in self.prefix_aliases)

    def apply_template(self):
        dsc = self.description_template(self.index)
        unt = self.unit_template(self.index)
        if self.description is None:
            self.description = dsc
        if self.units is None:
            self.units = unt

    def new_index_prefix_param(self, index):
        newpfx = prefixParameter(prefix=self.prefix,
                                 indexformat=self.indexformat, index=index,
                                 unitTplt=self.unit_template,
                                 descriptionTplt=self.description_template,
                                 frozen=self.frozen,
                                 continuous=self.continuous,
                                 parse_value=self.parse_value,
                                 print_value=self.print_value)
        newpfx.apply_template()
        return newpfx
