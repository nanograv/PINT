# parameter.py
# Defines Parameter class for timing model parameters
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string

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
            uncertainty=None, frozen=True, aliases=[], continuous=True,
            parse_value=fortran_float, print_value=str):
        self.value = value
        self.name = name
        self.units = units
        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = aliases
        self.parse_value=parse_value
        self.print_value=print_value

    def __str__(self):
        out = self.name
        if self.units is not None:
            out += " (" + str(self.units) + ")"
        out += " " + self.print_value(self.value)
        if self.uncertainty is not None:
            out += " +/- " + str(self.uncertainty)
        return out

    def set(self,value):
        """
        Parses a string 'value' into the appropriate internal representation
        of the parameter.
        """
        self.value = self.parse_value(value)

    def add_alias(self, alias):
        """
        Add a name to the list of aliases for this parameter.
        """
        self.aliases.append(alias)

    def help_line(self):
        """
        Return a help line containing param name, description and units.
        """
        out = "%-12s %s" % (self.name, self.description)
        if self.units is not None:
            out += ' (' + str(self.units) + ')'
        return out

    def as_parfile_line(self):
        """
        Return a parfile line giving the current state of the parameter.
        """
        # Don't print unset parameters
        if self.value is None: 
            return ""
        line = "%-15s %25s" % (self.name, self.print_value(self.value))
        if self.uncertainty is not None:
            line += " %d %s" % (0 if self.frozen else 1, str(self.uncertainty))
        elif not self.frozen:
            line += " 1" 
        return line + "\n"

    def from_parfile_line(self,line):
        """
        Parse a parfile line into the current state of the parameter.
        Returns True if line was successfully parsed, False otherwise.
        """
        try:
            k = line.split()
            name = k[0].upper()
        except:
            return False
        # Test that name matches
        if (name != self.name) and (name not in self.aliases):
            return False
        if len(k)<2:
            return False
        if len(k)>=2:
            self.set(k[1])
        if len(k)>=3:
            if int(k[2])>0: 
                self.frozen = False
        if len(k)==4:
            self.uncertainty = fortran_float(k[3])
        return True

class MJDParameter(Parameter):
    """
    MJDParameter(self, name=None, value=None, units=None, description=None, 
            uncertainty=None, frozen=True, aliases=[],
            parse_value=fortran_float, print_value=str):

    This is a Parameter type that is specific to MJD values.
    """

    def __init__(self, name=None, value=None, description=None, 
            uncertainty=None, frozen=True, continuous=True, aliases=[],
            parse_value=time_from_mjd_string,
            print_value=time_to_mjd_string):
        super(MJDParameter,self).__init__(name=name,value=value,
                units="MJD", description=description,
                uncertainty=uncertainty, frozen=frozen, 
                continuous=continuous,
                aliases=aliases,
                parse_value=parse_value,
                print_value=print_value)
