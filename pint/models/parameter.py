# parameter.py
# Defines Parameter class for timing model parameters
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string,\
                    time_to_longdouble
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
import re

class Parameter(object):
    """
    Parameter(name=None, value=None, units=None, description=None,
                uncertainty=None, frozen=True, continuous=True, aliases=[],
                parse_value=float, print_value=str)

        Class describing a single timing model parameter.  Takes the following
        inputs:

        name is the name of the parameter.

        value is the current value of the parameter.

        units is a string giving the units.

        description is a short description of what this parameter means.

        uncertainty is the current uncertainty of the value.

        frozen is a flag specifying whether "fitters" should adjust the
          value of this parameter or leave it fixed.

        continuous is flag specifying whether phase derivatives with
          respect to this parameter exist.

        aliases is an optional list of strings specifying alternate names
          that can also be accepted for this parameter.

        parse_value is a function that converts string input into the
          appropriate internal representation of the parameter (typically
          floating-point but could be any datatype).

        print_value is a function that converts the internal value to
          a string for output.

    """

    def __init__(self, name=None, value=None, units=None, description=None,
            uncertainty=None, frozen=True, aliases=None, continuous=True,
            parse_value=fortran_float, print_value=str):
        self.value = value
        self.name = name
        self.units = units
        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = [] if aliases is None else aliases
        self.parse_value = parse_value
        self.print_value = print_value
        self.is_prefix = False

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
            if int(k[2]) > 0:
                self.frozen = False
        if len(k) == 4:
            if name=="RAJ":
                self.uncertainty = Angle("0:0:%.15fh" % fortran_float(k[3]))
            elif name=="DECJ":
                self.uncertainty = Angle("0:0:%.15fd" % fortran_float(k[3]))
            else:
                self.uncertainty = fortran_float(k[3])
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
            print_value=time_to_mjd_string):
        super(MJDParameter, self).__init__(name=name, value=value,
                units="MJD", description=description,
                uncertainty=uncertainty, frozen=frozen,
                continuous=continuous,
                aliases=aliases,
                parse_value=parse_value,
                print_value=print_value)

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
            index with the right index format. So the optional arguments prefix,
            indexformat and index are need. index default value is 1.
        If both of two methods are fillfulled, It will using the first method.

        Add descrition and units.
        1. Direct add
            A descrition and unit can be added directly by using the optional
            arguments, descrition and units. Both of them will return as a string
            attribution.
        2. descrition and units template.
            If the descrition and unit are changing with the prefix parameter index,
            optional argurment descritionTplt and unitTplt are need. These two attributions
            are lambda functions, for example
            >>> descritionTplt = lambda x: 'This is the descrition of parameter %d'%x
            The class will fill the descrition and unit automaticly.
        If both two methods are fillfulled, it prefer the first one.

        Parameter
        ---------
        name : str optional
            The name of the parameter. If it is not provided, the prefix and index
            format are needed.
        prefix : str optional
            Paremeter prefix, now it is only supporting 'prefix_' type and 'prefix0' type.
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

    def __init__(self,name=None, prefix = None ,indexformat = None,index = 1,
            value=None,units = None, unitTplt = None,description=None,
            descriptionTplt = None,uncertainty=None, frozen=True,
            continuous=True,prefix_aliases = [],parse_value=fortran_float,
            print_value=None):
        # Create prefix parameter by name
        if name is None:
            if prefix is None or indexformat is None:
                errorMsg = 'When prefix parameter name is not give, the prefix'
                errorMsg += 'and indexformat are both needed.'
                raise ValueError(errorMsg)
            else:
                # Get format fields
                digitLen = 0
                for i in range(len(indexformat)-1,-1,-1):
                    if indexformat[i].isdigit():
                        digitLen+=1
                self.indexformat_field = indexformat[0:len(indexformat)-digitLen]
                self.indexformat_field += '{0:0' +str(digitLen)+'d}'

                name = prefix+self.indexformat_field.format(index)
                self.prefix = prefix
                self.indexformat = self.indexformat_field.format(0)
                self.index = index
        else: # Detect prefix and indexformat from name.
            namefield = re.split('(\d+)',name)
            if len(namefield)<2 or namefield[-2].isdigit() is False\
               or namefield[-1]!='':
            #When Name has no index in the end or no prefix part.
                errorMsg = 'Prefix parameter name needs a perfix part'\
                           +' and an index part in the end. '
                errorMsg += 'If you meant to set up with prefix, please use prefix '\
                           +'and indexformat optional agruments. Leave name argument alone.'
                raise ValueError(errorMsg)
            else: # When name has index in the end and prefix in front.
                indexPart = namefield[-2]
                prefixPart = namefield[0:-2]
                self.indexformat_field = '{0:0' +str(len(indexPart))+'d}'
                self.indexformat = self.indexformat_field.format(0)
                self.prefix = ''.join(prefixPart)
                self.index = int(indexPart)
        self.unit_template = unitTplt
        self.description_template = descriptionTplt

        super(prefixParameter, self).__init__(name=name, value=value,
                units=units, description=description,
                uncertainty=uncertainty, frozen=frozen,
                continuous=continuous,
                parse_value=parse_value,
                print_value=print_value)

        if units == 'MJD':
            self.parse_value = time_from_mjd_string
            self.print_value=time_to_mjd_string
        else:
            self.print_value = str

        self.prefix_aliases = prefix_aliases
        self.is_prefix = True

    def prefix_matches(self,prefix):
        return (prefix == self.perfix) or (prefix in self.prefix_aliases)

    def apply_template(self):
        dsc = self.description_template(self.index)
        unt = self.unit_template(self.index)
        if self.description is None:
            self.description = dsc
        if self.units is None:
            self.units = unt

    def new_index_prefix_param(self,index):
        newpfx = prefixParameter(prefix = self.prefix ,
                indexformat = self.indexformat,index = index,
                unitTplt = self.unit_template,
                descriptionTplt = self.description_template, frozen=self.frozen,
                continuous=self.continuous, parse_value=self.parse_value,
                print_value=self.print_value)
        newpfx.apply_template()
        return newpfx
