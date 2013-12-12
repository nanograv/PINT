# timing_model.py
# Defines the basic timing model interface classes
import string

class Parameter(object):
    """
    Parameter(name=None, value=None, units=None, description=None, 
                uncertainty=None, frozen=True, aliases=[],
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

        aliases is an optional list of strings specifying alternate names
          that can also be accepted for this parameter.

        parse_value is a function that converts string input into the
          appropriate internal representation of the parameter (typically
          floating-point but could be any datatype).

        print_value is a function that converts the internal value to
          a string for output.

    """

    def __init__(self, name=None, value=None, units=None, description=None, 
            uncertainty=None, frozen=True, aliases=[],
            parse_value=float, print_value=str):
        self.value = value
        self.name = name
        self.units = units
        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.aliases = aliases
        self.parse_value=parse_value
        self.print_value=print_value

    def __str__(self):
        out = self.name
        if self.units!=None:
            out += " (" + str(self.units) + ")"
        out += " " + self.print_value(self.value)
        if self.uncertainty!=None:
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

    def as_parfile_line(self):
        """
        Return a parfile line giving the current state of the parameter.
        """
        # Don't print unset parameters
        if self.value==None: 
            return ""
        line = "%-10s %25s" % (self.name, self.print_value(self.value))
        if self.uncertainty != None:
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
            self.uncertainty = float(k[3])
        return True

class TimingModel(object):

    def __init__(self):
        self.params = []  # List of model parameter names
        self.delay_funcs = [] # List of delay component functions
        self.phase_funcs = [] # List of phase component functions

        self.add_param(Parameter(name="PSR",
            units=None,
            description="Source name",
            aliases=["PSRJ","PSRB"],
            parse_value=str))

    def setup(self):
        print "TimingModel setup"

    def add_param(self, param):
        setattr(self, param.name, param)
        self.params += [param.name,]

    def compute_phase(self, toa):
        """
        Compute the model-predicted pulse phase for the given toa.
        """
        # First compute the delay to "pulsar time"
        delay = self.compute_delay(toa)
        phase = 0.0

        # Then compute the relevant pulse phase
        for pf in self.phase_funcs:
            phase += pf(toa - delay) # This is just a placeholder until we
                                     # define what datatype 'toa' has, and
                                     # how to add/subtract from it, etc.
        return phase

    def compute_delay(self, toa):
        """
        Compute the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.
        """
        delay = 0.0
        for df in self.delay_funcs:
            delay += getattr(self,df)(toa)
        return delay

    def __str__(self):
        result = ""
        for par in self.params:
            result += str(getattr(self,par)) + "\n"
        return result

    def as_parfile(self):
        """
        Returns a parfile representation of the entire model as a string.
        """
        result = ""
        for par in self.params:
            result += getattr(self,par).as_parfile_line()
        return result

    def read_parfile(self, filename):
        """
        Read values from the specified parfile into the model parameters.
        """
        pfile = open(filename,'r')
        for l in map(string.strip,pfile.readlines()):
            # Skip blank lines
            if not l: continue
            # Skip commented lines
            if l.startswith('#'): continue
            parsed = False
            for par in self.params:
                if getattr(self,par).from_parfile_line(l):
                    parsed = True
            if not parsed:
                print "warning: unrecognized parfile line '%s'" % l

        # The "setup" functions contain tests for required parameters or
        # combinations of parameters, etc, that can only be done
        # after the entire parfile is read
        self.setup()

def generate_timing_model(name,components):
    """
    Returns a timing model class generated from the specified 
    sub-components.  The return value is a class type, not an instance,
    so needs to be called to generate a usable instance.  For example:

    MyModel = generate_timing_model("MyModel",(Astrometry,Spindown))
    my_model = MyModel()
    my_model.read_parfile("J1234+1234.par")
    """
    # TODO could test here that all the components are derived from 
    # TimingModel?
    return type(name, components, {})

class TimingModelError(Exception):
    """
    Generic base class for timing model errors.
    """
    pass

class MissingParameter(TimingModelError):
    """
    This exception should be raised if a required model parameter was 
    not included.

    Attributes:
      module = name of the model class that raised the error
      param = name of the missing parameter
      msg = additional message
    """
    def __init__(self,module,param,msg=None):
        self.module = module
        self.param = param
        self.msg = msg

    def __str__(self):
        result = self.module + "." + self.param
        if self.msg != None:
            result += "\n  " + self.msg
        return result

