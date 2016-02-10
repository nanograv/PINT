# parameter.py
# Defines Parameter class for timing model parameters
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string,\
    time_to_longdouble, is_number, time_from_longdouble, str2longdouble, \
    longdouble2string, data2longdouble
import numpy
import astropy.units as u
import astropy.time as time
from astropy import log
from pint import pint_units
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
import re
import numbers
#from .timing_model import Cache

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
                 parse_value=fortran_float, print_value=str, set_value=lambda x: x,
                 get_value=lambda x: x, get_num_value=lambda x: x,
                 parse_uncertainty=None):
        self.name = name  # name of the parameter
        self.units = units  # parameter unit in string format,or None
        # parameter num unit, in astropy.units object format.
        # Once it is speicified, num_unit will not be changed.
        self.set_value = set_value
        # Method to get num_value from value
        self.get_num_value = get_num_value
        self.get_value = get_value  # Method to update value from num_value
        self.parse_value = parse_value  # method to read a value from string,
                                        # user can put the speicified format
                                        # here
        self.print_value = print_value  # method to convert value to a string.
        if parse_uncertainty is None:
            self.parse_uncertainty = fortran_float
        else:
            self.parse_uncertainty = parse_uncertainty
        self.value = value  # The value of parameter, internal storage
        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = [] if aliases is None else aliases
        self.is_prefix = False
        self.paramType = 'Parameter'  # Type of parameter. Here is general type
        self.valueType = None
    # Setup units property
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unt):
        # Check if this is the first time set units
        if hasattr(self, 'value'):
            if self.units is not None:
                wmsg = 'Parameter '+self.name+' units has been reset to '+unt
                wmsg += ' from '+self._units
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
        if val is None:
            if hasattr(self, 'value') and self.value is not None:
                raise ValueError('Setting an exist value to None is not'
                                 ' allowed.')
            else:
                self._value = val
                self._num_value = self._value
                return
        self._value = self.set_value(val)
        self._num_value = self.get_num_value(self._value)

    # Setup num_value property
    @property
    def num_value(self):
        return self._num_value

    @num_value.setter
    def num_value(self, val):
        if val is None:
            if not isinstance(self.value, (str, bool)):
                raise ValueError('This parameter value is number convertable. '
                                 'Setting ._num_value to None will lost the '
                                 'parameter value.')
        elif isinstance(val, numbers.Number):
            self._num_value = val
            self._value = self.get_value(val)
        else:
            raise ValueError('.num_value can only take a pure number.')
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
        self.value = value

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

class floatParameter(Parameter):
    """This is a Paraemeter type that is specific to astropy quantity values
    """
    def __init__(self, name=None, value=None, units=None, description=None,
                 uncertainty=None, frozen=True, aliases=None, continuous=True,
                 long_double=False):
        self.long_double = long_double
        if self.long_double:
            parse_value = self.set_value_float
            print_value = lambda x: longdouble2string(x.value)
            get_value = lambda x: data2longdouble(x)*self.num_unit

        else:
            parse_value = self.set_value_float
            print_value = lambda x: str(x.value)
            get_value = lambda x: x * self.num_unit

        get_num_value = self.get_num_value_float
        set_value = self.set_value_float
        super(floatParameter, self).__init__(name=name, value=value,
                                             units=units, frozen=True,
                                             aliases=aliases,
                                             continuous=continuous,
                                             description=description,
                                             uncertainty=uncertainty,
                                             parse_value=parse_value,
                                             print_value=print_value,
                                             set_value=set_value,
                                             get_value=get_value,
                                             get_num_value=get_num_value)
        self.value_type = u.quantity.Quantity
        self.paramType = 'floatParameter'


    def set_value_float(self, val):
        """Set value method specific for float parameter
        accept format
        1. Astropy quantity
        2. float
        3. string
        """
        if isinstance(val, u.Quantity):
            valu = val.unit
            if self.units is not None:
                try:
                    temp = val.to(self.num_unit)
                except:
                    emsg = 'Setting a uncompatible unit ' + valu.to_string
                    emsg += ' to value is not allowed'
                    raise ValueError(emsg)
                result = val
            else:
                result = val
        elif isinstance(val, numbers.Number):
            if self.num_unit is not None:
                result = val * self.num_unit
            else:
                result = val * u.Unit('')

        elif isinstance(val, str):
            try:
                if self.long_double:
                    v = str2longdouble(val)
                else:
                    v = fortran_float(val)
            except:
                raise ValueError('String ' + val + 'can not be converted to'
                                 ' float')
            result = self.set_value_float(v)
        else:
            raise ValueError('float parameter can not accept '
                             + type(val).__name__ + 'format.')
        return result

    def get_num_value_float(self, val):
        if val is None:
            return None
        else:
            return val.value

class strParameter(Parameter):
    """This is a Paraemeter type that is specific to string values
    """
    def __init__(self, name=None, value=None, description=None, frozen=True,
                 aliases=[]):
        parse_value = str
        print_value = str
        get_num_value = lambda x: None
        set_value = str
        get_value = self.get_value_str

        super(strParameter, self).__init__(name=name, value=value,
                                           description=None, frozen=True,
                                           aliases=aliases,
                                           parse_value=parse_value,
                                           print_value=print_value,
                                           set_value=set_value,
                                           get_value=get_value,
                                           get_num_value=get_num_value)

        self.paramType = 'strParameter'
        self.value_type = str

    def get_value_str(self, val):
        raise ValueError('Can not set a num value to a string type parameter.')

class boolParameter(Parameter):
    """This is a Paraemeter type that is specific to boolen values
    """
    def __init__(self, name=None, value=None, description=None, frozen=True,
                 aliases=[]):

        parse_value = lambda x: x.upper() == 'Y'
        print_value = lambda x: 'Y' if x else 'N'
        set_value = self.set_value_bool
        get_num_value = lambda x: None
        get_value = lambda x: log.warning('Can not set a pure value to a '
                                               'string boolen parameter.')
        super(boolParameter, self).__init__(name=name, value=value,
                                            description=None, frozen=True,
                                            aliases=aliases,
                                            parse_value=parse_value,
                                            print_value=print_value,
                                            set_value=set_value,
                                            get_value=get_value,
                                            get_num_value=get_num_value)
        self.value_type = bool
        self.paramType = 'boolParameter'

    def set_value_bool(self, val):
        """ This function is to get boolen value for boolParameter class
        """
        if isinstance(val, str):
            return val.upper() in ['Y', 'YES', 'SI']
        elif isinstance(val, bool):
            return val
        elif isinstance(val, numbers.Number):
            return val != 0


class MJDParameter(Parameter):
    """This is a Parameter type that is specific to MJD values."""
    def __init__(self, name=None, value=None, description=None,
                 uncertainty=None, frozen=True, continuous=True, aliases=None,
                 time_scale='utc'):
        self.time_scale = time_scale
        parse_value = self.set_value_mjd
        set_value = self.set_value_mjd
        print_value = time_to_mjd_string
        get_value = lambda x: time_from_longdouble(x, time_scale)
        get_num_value = time_to_longdouble
        super(MJDParameter, self).__init__(name=name, value=value, units="MJD",
                                           description=description,
                                           uncertainty=uncertainty,
                                           frozen=frozen,
                                           continuous=continuous,
                                           aliases=aliases,
                                           parse_value=parse_value,
                                           print_value=print_value,
                                           set_value=set_value,
                                           get_value=get_value,
                                           get_num_value=get_num_value)
        self.value_type = time.Time
        self.paramType = 'MJDParameter'

    def set_value_mjd(self, val):
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
                raise ValueError('String ' + val + 'can not be converted to'
                                 'a time object.' )

        elif isinstance(val,time.Time):
            result = val
        else:
            raise ValueError('MJD parameter can not accept '
                             + type(val).__name__ + 'format.')
        return result


class AngleParameter(Parameter):
    """This is a Parameter type that is specific to Angle values."""
    def __init__(self, name=None, value=None, description=None, units='rad',
             uncertainty=None, frozen=True, continuous=True, aliases=None):
        self.separator = {
            'h:m:s': (u.hourangle, 'h', '0:0:%.15fh'),
            'd:m:s': (u.deg, 'd', '0:0:%.15fd'),
            'rad': (u.rad, 'rad', '%.15frad'),
            'deg': (u.deg, 'deg', '%.15fdeg'),
        }
        # Check unit format
        if units.lower() not in self.separator.keys():
            raise ValueError('Unidentified unit ' + units)

        self.unitsuffix = self.separator[units.lower()][1]
        set_value = self.set_value_angle
        parse_value = lambda x: Angle(x + self.unitsuffix)
        print_value = lambda x: x.to_string(sep=':', precision=8) \
                        if x.unit != u.rad else x.to_string(decimal = True,
                        precision=8)
        get_value = lambda x: Angle(x * self.separator[units.lower()][0])
        get_num_value = lambda x: x.value
        parse_uncertainty = lambda x: \
                             Angle(self.separator[units.lower()][2] \
                                   % fortran_float(x))
        self.value_type = Angle
        self.paramType = 'AngleParameter'

        super(AngleParameter, self).__init__(name=name, value=value,
                                             units=units,
                                             description=description,
                                             uncertainty=uncertainty,
                                             frozen=frozen,
                                             continuous=continuous,
                                             aliases=aliases,
                                             parse_value=parse_value,
                                             print_value=print_value,
                                             set_value=set_value,
                                             get_value=get_value,
                                             get_num_value=get_num_value,
                                             parse_uncertainty=parse_uncertainty)

    def set_value_angle(self,val):
        """ This function is to set value to angle parameters.
        Accepted format:
        1. Astropy angle object
        2. float
        3. number string
        """
        if isinstance(val, numbers.Number):
            result = Angle(val * self.num_unit)
        elif isinstance(val, str):
            try:
                result = Angle(val + self.unitsuffix)
            except:
                raise ValueError('Srting ' + val + ' can not be converted to'
                                 ' astropy angle.')
        elif isinstance(val, Angle):
            result = val.to(self.num_unit)
        else:
            raise ValueError('Angle parameter can not accept '
                             + type(val).__name__ + 'format.')
        return result


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
                 prefix_aliases=[], type_match='float', long_double=False,
                 time_scale='utc'):
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

        # Set up other attributes
        self.unit_template = unitTplt
        self.description_template = descriptionTplt
        # set templates
        if self.unit_template is None:
            self.unit_template = lambda x: self.units
        if self.description_template is None:
            self.description_template = lambda x: self.descrition
        # Here is a bug example parameters should be initialize totally
        self.type_separator = {
                               'float': (floatParameter,
                                         ['units',' value', 'long_double',
                                          'num_value', 'uncertainty']),
                               'string': (strParameter, ['value']),
                               'bool': (boolParameter, ['value']),
                               'mjd': (MJDParameter, ['time_scale', 'value'
                                                     'num_value',
                                                     'uncertainty']),
                               'angle': (AngleParameter, ['units', 'value',
                                                         'uncertainty',
                                                         'num_value'])}
        if isinstance(type_match, str):
            self.type_match = type_match.lower()
        elif isinstance(value_type, type):
            self.type_match = type_match.__name__
        else:
            self.type_match = type(type_match).__name__

        if self.type_match not in self.type_separator.keys():
            raise ValueError('Unrecognized value type ' + self.type_match)

        print_value = self.print_value_prefix
        set_value = self.set_value_prefix
        get_value = self.get_value_prefix
        get_num_value = self.get_num_value_prefix
        parse_uncertainty = self.parse_uncertainty_prefix

        super(prefixParameter, self).__init__(name=name, value=value,
                                              units=units,
                                              description=description,
                                              uncertainty=uncertainty,
                                              frozen=frozen,
                                              continuous=continuous,
                                              parse_value=set_value,
                                              print_value=print_value,
                                              set_value=set_value,
                                              get_value=get_value,
                                              get_num_value=get_num_value,
                                              parse_uncertainty=parse_uncertainty)

        self.prefix_aliases = prefix_aliases
        self.is_prefix = True

    def prefix_matches(self, prefix):
        return (prefix == self.perfix) or (prefix in self.prefix_aliases)

    def apply_template(self):
        dsc = self.description_template(self.index)
        self.description = dsc
        unt = self.unit_template(self.index)
        self.units = unt

    #@Cache.cache_result
    def get_par_type_object(self):
        par_type_class = self.type_separator[self.type_match][0]
        obj = par_type_class('example')
        attr_dependency = self.type_separator[self.type_match][1]
        for dp in attr_dependency:
            if hasattr(self, dp):
                prefix_arg = getattr(self, dp)
                setattr(obj, dp, prefix_arg)
        return obj

    #@Cache.use_cache
    def set_value_prefix(self, val):
        obj = self.get_par_type_object()
        result = obj.set_value(val)
        return result

    #@Cache.use_cache
    def get_value_prefix(self, val):
        obj = self.get_par_type_object()
        result = obj.get_value(val)
        return result

    #@Cache.use_cache
    def get_num_value_prefix(self, val):
        obj = self.get_par_type_object()
        result = obj.get_num_value(val)
        return result

    #@Cache.use_cache
    def print_value_prefix(self, val):
        obj = self.get_par_type_object()
        result = obj.print_value(val)
        return result

    #@Cache.use_cache
    def parse_uncertainty_prefix(self, val):
        obj = self.get_par_type_object()
        result = obj.parse_uncertainty(val)
        return result

    def new_index_prefix_param(self, index):
        newpfx = prefixParameter(prefix=self.prefix, value=self.value,
                                 indexformat=self.indexformat, index=index,
                                 unitTplt=self.unit_template,
                                 descriptionTplt=self.description_template,
                                 frozen=self.frozen,
                                 continuous=self.continuous,
                                 type_match=self.type_match)
        newpfx.apply_template()
        return newpfx
