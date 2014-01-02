# timing_model.py
# Defines the basic timing model interface classes
import string
from warnings import warn
from .parameter import Parameter
from ..phase import Phase

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
        pass

    def add_param(self, param):
        setattr(self, param.name, param)
        self.params += [param.name,]

    def param_help(self):
        """
        Print help lines for all available parameters in model.
        """
        print "Available parameters for ", self.__class__
        for par in self.params:
            print getattr(self,par).help_line()

    def phase(self, toa):
        """
        Return the model-predicted pulse phase for the given toa.
        """
        # First compute the delay to "pulsar time"
        delay = self.delay(toa)
        phase = Phase(0,0.0)

        # Then compute the relevant pulse phase
        for pf in self.phase_funcs:
            phase += pf(toa,delay)  # This is just a placeholder until we
                                    # define what datatype 'toa' has, and
                                    # how to add/subtract from it, etc.
        return phase

    def delay(self, toa):
        """
        Return the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.
        """
        delay = 0.0
        for df in self.delay_funcs:
            delay += df(toa)
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
                warn("Unrecognized parfile line '%s'" % l)

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
        if self.msg is not None:
            result += "\n  " + self.msg
        return result

