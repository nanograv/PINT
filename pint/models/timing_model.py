# timing_model.py
# Defines the basic timing model interface classes
import functools
from warnings import warn
from .parameter import Parameter
from ..phase import Phase
from astropy import log
import numpy as np
class Cache(object):
    """Temporarily cache timing model internal computation results.

    The Cache class defines two decorators, use_cache and cache_result.
    """

    # The name of the cache attribute
    the_cache = "cache"

    @classmethod
    def cache_result(cls, function):
        """Caching decorator for functions.

        This can be applied as a decorator to any timing model method
        for which it might be useful to store the value, once computed
        for a given TOA.  Note that the cache must be manually enabled
        and cleared when appropriate, so this functionality should be
        used with care.
        """
        the_func = function.__name__
        @functools.wraps(function)
        def get_cached_result(*args, **kwargs):
            log.debug("Checking for cached value of %s" % the_func)
            # What to do about checking for a change of arguments?
            # args[0] should be a "self"
            if hasattr(args[0], cls.the_cache):
                cache = getattr(args[0], cls.the_cache)
                if isinstance(cache, cls):
                    if hasattr(cache, the_func):
                        # Return the cached value
                        log.debug(" ... using cached result")
                        return getattr(cache, the_func)
                    else:
                        # Evaluate the function and cache the results
                        log.debug(" ... computing new result")
                        result = function(*args, **kwargs)
                        setattr(cache, the_func, result)
                        return result
            # Couldn't access the cache, just return the result
            # without caching it.
            log.debug(" ... no cache found")
            return function(*args, **kwargs)
        return get_cached_result

    @classmethod
    def use_cache(cls, function):
        """Caching decorator for functions.

        This can be applied as a decorator to a function that should
        internally use caching of function return values.  The cache
        will be deleted when the function exits.  If the top-level function
        calls other functions that have caching enabled they will share
        the cache, and it will only be deleted when the top-level function
        exits.
        """
        @functools.wraps(function)
        def use_cached_results(*args, **kwargs):
            # args[0] should be a "self"
            # Test whether a cache attribute is present
            if hasattr(args[0], cls.the_cache):
                cache = getattr(args[0], cls.the_cache)
                # Test whether caching is already enabled
                if isinstance(cache, cls):
                    # Yes, just execute the function
                    return function(*args, **kwargs)
                else:
                    # Init the cache, excute the function, then delete cache
                    setattr(args[0], cls.the_cache, cls())
                    result = function(*args, **kwargs)
                    setattr(args[0], cls.the_cache, None)
                    return result
            else:
                # no "self.cache" attrib is found.  Could raise an error, or
                # just execute the function normally.
                return function(*args, **kwargs)
        return use_cached_results



class TimingModel(object):

    def __init__(self):
        self.params = []  # List of model parameter names
        self.delay_funcs = [] # List of delay component functions
        self.delay_funcs_ld = [] # List of delay component function long double
        self.phase_funcs = [] # List of phase component functions
        self.phase_funcs_ld = [] # List of phase function long double
        self.cache = None
      
        self.add_param(Parameter(name="PSR",
            units=None,
            description="Source name",
            aliases=["PSRJ", "PSRB"],
            parse_value=str))

    def setup(self):
        pass

    def add_param(self, param):
        setattr(self, param.name, param)
        self.params += [param.name,]

    def param_help(self):
        """Print help lines for all available parameters in model.
        """
        s = "Available parameters for %s\n" % self.__class__
        for par in self.params:
            s += "%s\n" % getattr(self, par).help_line()

    @Cache.use_cache
    def phase(self, toa):
        """Return the model-predicted pulse phase for the given toa.
        """
        # First compute the delay to "pulsar time"
        delay = self.delay(toa)
        phase = Phase(0, 0.0)

        # Then compute the relevant pulse phase
        for pf in self.phase_funcs:
            phase += pf(toa, delay)  # This is just a placeholder until we
                                     # define what datatype 'toa' has, and
                                     # how to add/subtract from it, etc.
        return phase

    @Cache.use_cache
    def delay(self, toa):
        """Total delay for a given TOA.

        Return the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.
        """
        delay = 0.0
        for df in self.delay_funcs:
            delay += df(toa)
        return delay
    
    def delay_ld_array(self,TOAs):
        """
        Total delay for a given TOA long double numpy array
        """
        delay = np.zeros_like(TOAs.tdbld)
        for df in self.delay_funcs_ld:
            delay += df(TOAs) 
        return delay
        
            
    def d_phase_d_tpulsar(self, toa):
        """
        Return the derivative of phase wrt time at the pulsar.
        NOT Implemented
        """
        pass

    def d_phase_d_toa(self, toa):
        """
        Return the derivative of phase wrt TOA (ie the current apparent
        spin freq of the pulsar at the observatory).
        NOT Implemented yet.
        """
        pass

    def d_phase_d_param(self, toa, param):
        """
        Return the derivative of phase with respect to the parameter.
        NOTE, not implemented yet
        """
        result = 0.0
        # TODO need to do correct chain rule stuff wrt delay derivs, etc
        # Is it safe to assume that any param affecting delay only affects
        # phase indirectly (and vice-versa)??
        return result

    def d_delay_d_param(self, toa, param):
        """
        Return the derivative of delay with respect to the parameter.
        """
        result = 0.0
        for f in self.delay_derivs[param]:
            result += f(toa)
        return result

    def __str__(self):
        result = ""
        for par in self.params:
            result += str(getattr(self, par)) + "\n"
        return result

    def as_parfile(self):
        """Returns a parfile representation of the entire model as a string.
        """
        result = ""
        for par in self.params:
            result += getattr(self, par).as_parfile_line()
        return result

    def read_parfile(self, filename):
        """Read values from the specified parfile into the model parameters.
        """
        pfile = open(filename, 'r')
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue
            # Skip commented lines
            if l.startswith('#'):
                continue
            parsed = False
            for par in self.params:
                if getattr(self, par).from_parfile_line(l):
                    parsed = True
            if not parsed:
                warn("Unrecognized parfile line '%s'" % l)

        # The "setup" functions contain tests for required parameters or
        # combinations of parameters, etc, that can only be done
        # after the entire parfile is read
        self.setup()

def generate_timing_model(name, components,longdouble = False):
    """Build a timing model from components.

    Returns a timing model class generated from the specifiied
    sub-components.  The return value is a class type, not an instance,
    so needs to be called to generate a usable instance.  For example:

    MyModel = generate_timing_model("MyModel",(Astrometry,Spindown))
    my_model = MyModel()
    my_model.read_parfile("J1234+1234.par")
    """
    # TODO could test here that all the components are derived from
    # TimingModel?
    if longdouble == False:  #  space for a numpy longdouble flag
        return type(name, components, {})
    else:
        return type(name, components, {})
class TimingModelError(Exception):
    """Generic base class for timing model errors.
    """
    pass

class MissingParameter(TimingModelError):
    """A required model parameter was not included.

    Attributes:
      module = name of the model class that raised the error
      param = name of the missing parameter
      msg = additional message
    """
    def __init__(self, module, param, msg=None):
        super(MissingParameter, self).__init__()
        self.module = module
        self.param = param
        self.msg = msg

    def __str__(self):
        result = self.module + "." + self.param
        if self.msg is not None:
            result += "\n  " + self.msg
        return result

class DuplicateParameter(TimingModelError):
    """
    This exception is raised if a model parameter is defined (added)
    multiple times.
    """
    def __init__(self, param, msg=None):
        self.param = param
        self.msg = msg

    def __str__(self):
        result = self.param
        if self.msg is not None:
            result += "\n  " + self.msg
        return result

