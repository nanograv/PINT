# timing_model.py
# Defines the basic timing model interface classes
import functools
from .parameter import Parameter
from ..phase import Phase
from astropy import log
import numpy as np
import pint.toa as toa 
import pint.utils as utils

# parameters or lines in parfiles to ignore (for now?), or at
# least not to complain about
ignore_params = ['START', 'FINISH', 'SOLARN0', 'EPHEM', 'CLK', 'UNITS',
                 'TIMEEPH', 'T2CMETHOD', 'CORRECT_TROPOSPHERE', 'DILATEFREQ',
                 'NTOA', 'CLOCK', 'TRES', 'TZRMJD', 'TZRFRQ', 'TZRSITE',
                 'NITS', 'IBOOT']

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
        self.params = []  # List of model parameter names
        self.delay_funcs = [] # List of delay component functions
        self.phase_funcs = [] # List of phase component functions
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
        return s

    @Cache.use_cache
    def phase(self, toas):
        """Return the model-predicted pulse phase for the given TOAs."""
        # First compute the delays to "pulsar time"
        delay = self.delay(toas)
        phase = Phase(np.zeros(len(toas)), np.zeros(len(toas)))
        # Then compute the relevant pulse phases
        for pf in self.phase_funcs:
            phase += Phase(pf(toas, delay))
        return phase

    @Cache.use_cache
    def delay(self, toas):
        """Total delay for the TOAs.

        Return the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.
        """
        delay = np.zeros(len(toas))
        for df in self.delay_funcs:
            delay += df(toas)
        return delay

    def d_phase_d_tpulsar(self, toas):
        """Return the derivative of phase wrt time at the pulsar.

        NOT implemented yet.
        """
        pass

    def d_phase_d_toa(self, toas, maxStep = 3):
        """Return the derivative of phase wrt TOA
        Time sample
        --------------------------------------------------
        Toa0-n*dt, Toa0-(n-1)*dt, ... Toa0, ..., Toa0+n*dt
                                      Toa1,
                                        .
                                        .
                                        .    
        --------------------------------------------------    
        x = [Toa0-n*dt, Toa0-(n-1)*dt, ... Toa0, ..., Toa0+n*dt]
        y = phaes([Toa0-n*dt, Toa0-(n-1)*dt, ... Toa0, ..., Toa0+n*dt])
        
        dy/dx will be calculated using numpy gradient. 
        d_phase_d_toa = dy/dx(toa) 

        """
    
        # Using finite difference, resample near the toas
        dt = np.longdouble(1.0)/self.F0.value*10000
        sampleArray = np.linspace(-maxStep*dt,maxStep*dt,2*maxStep+1)
        toas_Temp = toas # Add a temp toa table
        # matrix for new time sample and phase value
        time_tdb = np.longdouble(np.zeros((len(toas),len(sampleArray))))
        p = np.zeros_like(time_tdb)
        # Calculate the phase for each time sample
        
        for i,timeStep in enumerate(sampleArray):
            toas_Temp['tdbld'] = toas['tdbld']+timeStep/86400.0
            ph = self.phase(toas_Temp)
            p[:,i] = ph.frac+ph.int
            time_tdb[:,i] = toas_Temp['tdbld']

        dy = np.zeros_like(p)
        # Do derivative on time samples near toas. 
        for i in range(len(toas)):
            dx = np.gradient(time_tdb[i,:]*86400.0)
            y = p[i,:]
            dy[i,:] = np.gradient(y-np.mean(y),dx)
        # Return derivative value on toa.
        return dy[:,maxStep]

    def d_phase_d_toa_chebyshev(self,toas,n = 12):
        """Return the derivative of phase wrt TOA.

           The derivation will be performed on the scale of one hour. 
        """

        step = np.longdouble(1.0)/self.F0.value*100
        d_p_d_toa = []
        for time,obs in zip(toas['mjd'],toas['obs']):
            toa_utc_ld = utils.time_to_longdouble(time)
            domain = (toa_utc_ld - np.longdouble(0.5)*3600.0/86400.0, 
                      toa_utc_ld + np.longdouble(0.5)*3600.0/86400.0)
            numData = np.longdouble(3600.0)/step
            resmpl_time = np.linspace(domain[0],domain[1],numData)
            toa_filelike = toa.TOA_file_like((np.modf(resmpl_time)[1],np.modf(resmpl_time)[0]),
                            obs = obs)
            resmpl_toas = toa.get_TOAs_filelike(toa_filelike)
            ph = self.phase(resmpl_toas.table)
            p = ph.int-ph.int.min()+ph.frac
            p = np.array(p,dtype='float64')
            resmpl_time = np.array((resmpl_time-resmpl_time.min())*86400.0,dtype = 'float64')
            print resmpl_time.dtype
            coeff = np.polynomial.chebyshev.chebfit(resmpl_time, p,n)
            dcoeff = np.polynomial.chebyshev.chebder(coeff)
            dy =  np.polynomial.chebyshev.chebval(resmpl_time,dcoeff)
            d_p_d_toa.append(dy[len(resmpl_time)/2+1])
        return np.array(d_p_d_toa)

    def d_phase_d_param(self, toas, param):
        """ Return the derivative of phase with respect to the parameter.

        NOT implemented yet.
        """
        result = 0.0
        # TODO need to do correct chain rule stuff wrt delay derivs, etc
        # Is it safe to assume that any param affecting delay only affects
        # phase indirectly (and vice-versa)??
        return result

    def d_delay_d_param(self, toas, param):
        """
        Return the derivative of delay with respect to the parameter.
        """
        result = numpy.zeros(len(toas))
        for f in self.delay_derivs[param]:
            result += f(toas)
        return result

    def __str__(self):
        result = ""
        for par in self.params:
            result += str(getattr(self, par)) + "\n"
        return result

    def as_parfile(self):
        """Returns a parfile representation of the entire model as a string."""
        result = ""
        for par in self.params:
            result += getattr(self, par).as_parfile_line()
        return result

    def read_parfile(self, filename):
        """Read values from the specified parfile into the model parameters."""
        pfile = open(filename, 'r')
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue
            # Skip commented lines
            if l.startswith('#') or l[:2]=="C ":
                continue
            parsed = False
            for par in self.params:
                if getattr(self, par).from_parfile_line(l):
                    parsed = True
            if not parsed and l.split()[0] not in ignore_params:
                log.warn("Unrecognized parfile line '%s'" % l)

        # The "setup" functions contain tests for required parameters or
        # combinations of parameters, etc, that can only be done
        # after the entire parfile is read
        self.setup()

def generate_timing_model(name, components):
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
    return type(name, components, {})

class TimingModelError(Exception):
    """Generic base class for timing model errors."""
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
