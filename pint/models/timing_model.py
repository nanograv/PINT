# timing_model.py
# Defines the basic timing model interface classes
import string
import functools
from warnings import warn
from ..utils import fortran_float, time_from_mjd_string, time_to_mjd_string
from ..phase import Phase

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
            parse_value=fortran_float, print_value=str):
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
            uncertainty=None, frozen=True, aliases=[],
            parse_value=fortran_float, print_value=str):
        super(MJDParameter,self).__init__(name=name,value=value,
                units="MJD", description=description, 
                uncertainty=uncertainty, frozen=frozen, aliases=aliases,
                parse_value=time_from_mjd_string,
                print_value=time_to_mjd_string)

class Cache(object):
    """
    The Cache class is for temporarily caching timing model parameter
    results.  By itself it does not do anything.
    """
    pass

class TimingModel(object):

    def __init__(self):
        self.params = []  # List of model parameter names
        self.delay_funcs = [] # List of delay component functions
        self.phase_funcs = [] # List of phase component functions
        self.cache = None

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

    def enable_cache(self):
        """
        Run once to initialize a cache in which intermediate results
        can be stored.  To have a function cache its result, define
        it with @TimingModel.cache_result.
        """
        if self.cache is not None:
            # A cache exists already.. should this raise an exception,
            # use the existing cache, or clear the existing cache??
            # For now, use existing one.
            pass
        else:
            self.cache = Cache()

    def disable_cache(self):
        """
        Clear/disable the function result cache.
        """
        self.cache = None

    def phase(self, toa):
        """
        Return the model-predicted pulse phase for the given toa.
        """

        # Use the cache while in here
        self.enable_cache()

        # First compute the delay to "pulsar time"
        delay = self.delay(toa)
        phase = Phase(0,0.0)

        # Then compute the relevant pulse phase
        for pf in self.phase_funcs:
            phase += pf(toa,delay)  # This is just a placeholder until we
                                    # define what datatype 'toa' has, and
                                    # how to add/subtract from it, etc.

        # Delete the cache when we're done
        self.disable_cache()

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

    @classmethod
    def cache_result(cls,function):
        """
        This can be applied as a decorator to any timing model method
        for which it might be useful to store the value, once computed
        for a given TOA.  Note that the cache must be manually enabled
        and cleared when appropriate, so this functionality should be
        used with care.
        """
        the_func = function.__name__
        the_cache = "cache"
        @functools.wraps(function)
        def get_cached_result(*args,**kwargs):
            #print "Checking for cached value of", the_func
            # What to do about checking for a change of arguments?
            # args[0] should be a "self"
            if hasattr(args[0], the_cache):
                cache = getattr(args[0], the_cache)
                if isinstance(cache, Cache):
                    if hasattr(cache, the_func):
                        # Return the cached value
                        #print " ... using cached result"
                        return getattr(cache, the_func)
                    else:
                        # Evaluate the function and cache the results
                        #print " ... computing new result"
                        result = function(*args, **kwargs)
                        setattr(cache, the_func, result)
                        return result
            # Couldn't access the cache, just return the result
            # without caching it.
            #print " ... no cache found"
            return function(*args, **kwargs)
        return get_cached_result


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

