# timing_model.py
# Defines the basic timing model interface classes
import functools
from .parameter import Parameter
from ..phase import Phase
from astropy import log
import astropy.time as time
import numpy as np
import pint.toa as toa
import pint.utils as utils
import astropy.units as u
import copy
# parameters or lines in parfiles to ignore (for now?), or at
# least not to complain about
ignore_params = ['START', 'FINISH', 'SOLARN0', 'EPHEM', 'CLK', 'UNITS',
                 'TIMEEPH', 'T2CMETHOD', 'CORRECT_TROPOSPHERE', 'DILATEFREQ',
                 'NTOA', 'CLOCK', 'TRES', 'TZRMJD', 'TZRFRQ', 'TZRSITE',
                 'NITS', 'IBOOT','BINARY']
ignore_prefix = ['DMXF1_','DMXF2_','DMXEP_'] # DMXEP_ for now.

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
        self.prefix_params = []  # List of model parameter names
        self.num_prefix_params = {}
        self.params = []  # List of model parameter names
        self.delay_funcs = {'L1':[],'L2':[]} # List of delay component functions
        # L1 is the first level of delays. L1 delay does not need barycentric toas
        # After L1 delay, the toas have been corrected to solar system barycenter.
        # L2 is the second level of delays. L2 delay need barycentric toas

        self.phase_funcs = [] # List of phase component functions
        self.cache = None
        self.add_param(Parameter(name="PSR",
            units=None,
            description="Source name",
            aliases=["PSRJ", "PSRB"],
            parse_value=str))
        self.model_type = None

    def setup(self):
        pass


    def add_param(self, param,binary_param = False):
        """Add a parameter to the timing model. If it is a prefixe parameter,
           it will add prefix information to the prefix information attributes.
        """
        setattr(self, param.name, param)
        self.params += [param.name,]
        if param.is_prefix is True:
            if param.prefix not in self.prefix_params:
                self.prefix_params.append(param.prefix)
            if self.num_prefix_params.has_key(param.prefix):
                self.num_prefix_params[param.prefix]+=1
            else:
                self.num_prefix_params[param.prefix]=1
        if binary_param is True:
            self.binary_params +=[param.name,]


    def add_more_prefix_params(self,prefixParamExp,maxPrefixIndex):
        """Add more same type of prefixed parameters into the timing model.
           Parameter
           ----------
           prefixParamExp : PINT prefixParam class
               The eample prefixed parameter need to be added
           maxPrefixIndex : int
               The maximum number of prefixed parameter can be add.
        """
        try:
            isp = prefixParamExp.is_prefix
            if isp is not True:
                raise ValueError('prefixParamExp needs to be a prefixParameter class.')
        except:
            raise ValueError('prefixParamExp needs to be a Parameter class.')

        pfx = prefixParamExp.prefix
        idxfmt = prefixParamExp.indexformat
        unitTplt = prefixParamExp.unit_template
        descriptionTplt = prefixParamExp.description_template
        frozen=prefixParamExp.frozen
        continuous=prefixParamExp.continuous
        parse_value=prefixParamExp.parse_value
        print_value=prefixParamExp.print_value
        for ii in range(1,maxPrefixIndex+1):

            pp = prefixParameter(prefix = prefix ,indexformat = idxfmt,
                    index = ii, unitTplt = unitTplt,
                    descriptionTplt = descriptionTplt, frozen=frozen,
                    continuous=continuous, parse_value=parse_value,
                    print_value=print_value)
            pp.apply_template()
            if pp.name not in self.params:
                self.add_param(pp)



    def param_help(self):
        """Print help lines for all available parameters in model.
        """
        s = "Available parameters for %s\n" % self.__class__
        for par in self.params:
            s += "%s\n" % getattr(self, par).help_line()
        return s


    @Cache.use_cache
    def get_prefix_mapping(self,prefix):
        """Get the index mapping for the prefix parameters.
           Parameter
           ----------
           prefix : str
               Name of prefix.
           Return
           ----------
           A dictionary with prefix pararameter real index as key and parameter
           name as value.
        """
        parnames = [x for x in self.params if x.startswith(prefix)]
        mapping = dict()
        for parname in parnames:
            par = getattr(self,parname)
            if par.is_prefix == True:
                mapping[par.index] = parname

        setattr(self,prefix+'mapping',mapping)


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
        for dlevel in self.delay_funcs.keys():
            for df in self.delay_funcs[dlevel]:
                delay += df(toas)

        return delay

    @Cache.use_cache
    def get_barycentric_toas(self,toas):
        toasObs = toas['tdbld']
        delay = np.zeros(len(toas))
        for df in self.delay_funcs['L1']:
            delay += df(toas)
        toasBary = toasObs*u.day - delay*u.second
        return toasBary

    def d_phase_d_tpulsar(self, toas):
        """Return the derivative of phase wrt time at the pulsar.

        NOT implemented yet.
        """
        pass

    def d_phase_d_toa(self, toas, obs=None, sample_step=None):
        """Return the derivative of phase wrt TOA
        Parameter
        ---------
        toas : PINT TOAs class
            The toas where the derivative of phase will be evaluated at.
        obs : str optional
            The location where the derivative of phase will be evaluated at.
        sample_step : float optional
            Finite difference steps. If not specified, it will take 1/10 of the
            spin period. 
        """
        copy_toas = copy.deepcopy(toas)
        ttable = copy_toas.table
        if sample_step is None:
            pulse_period = 1.0/self.F0.num_value
            sample_step = pulse_period/10.0
        sample_step_sec = sample_step*u.second
        sample_step_day = sample_step_sec.to(u.day).value
        sample_dt = [-sample_step_day,2 * sample_step_day]
        table = toas.table
        time_scale = table['mjd'][0].scale
        if time_scale == 'utc':
            time_scale = 'ut1'


        if obs is None:
            TOAobj = copy_toas
            delay_fs = [self.delay,]
            phase_fs = self.phase_funcs

        # Get read for barycentric d_phase_d_toa
        elif obs.lower() in ['ssb','bary','barycenter']:
            time_scale = 'tdb'
            bTOA = self.get_barycentric_toas(ttable)
            TOAobj = toa.read_fake_TOAs(bTOA.value, obs='Barycenter',
                                        scale=time_scale)
            delay_fs = self.delay_funcs['L2'] # Only the second level delay
                                              # Solar system delay will not count
            phase_fs = self.phase_funcs

        else:
            raise ValueError('Unknown position to calculate d_phase_d_toa')

        sample_phase = []
        for dt in sample_dt:

            deltaT = time.TimeDelta([dt]*len(table),
                                    format='jd',scale=time_scale)
            TOAobj.adjust_TOAs(deltaT)
            delay = np.zeros(len(ttable))
            for df in delay_fs:
                delay += df(TOAobj.table)
            # Compute phase
            phase = Phase(np.zeros(len(ttable)), np.zeros(len(ttable)))
            for pf in phase_fs:
                phase += Phase(pf(TOAobj.table, delay))
            sample_phase.append(phase)
        # Use finite difference method.
        # phase'(t) = (phase(t+h)-phase(t-h))/2+ 1/6*F2*h^2 + ..
        # The error should be near 1/6*F2*h^2
        dp = (sample_phase[1]-sample_phase[0])
        d_phase_d_toa = dp.int/(2*sample_step)+dp.frac/(2*sample_step)
        del copy_toas

        return d_phase_d_toa

    @Cache.use_cache
    def d_phase_d_param(self, toas, param):
        """ Return the derivative of phase with respect to the parameter.

        Either analytically, or numerically
        """
        result = 0.0

        an_funcname = "d_phase_d_" + param
        if hasattr(self, an_funcname):
            # Have an analytic function for this parameter
            result = getattr(self, an_funcname)(toas)
        else:
            result = self.d_phase_d_param_num(toas, param)

        return result

    @Cache.use_cache
    def d_phase_d_param_num(self, toas, param):
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

    @Cache.use_cache
    def designmatrix(self, toas, incfrozen=False, incoffset=True):
        """
        Return the design matrix: the matrix with columns of d_phase_d_param/F0
        or d_toa_d_param
        """
        params = ['Offset',] if incoffset else []
        params += [par for par in self.params if incfrozen or
                not getattr(self, par).frozen]

        F0 = self.F0.value / u.s        # 1/sec
        ntoas = len(toas)
        nparams = len(params)
        delay = self.delay(toas)
        units = []

        # Apply all delays ?
        #tt = toas['tdbld']
        #for df in self.delay_funcs:
        #    tt -= df(toas)

        M = np.zeros((ntoas, nparams))
        for ii, param in enumerate(params):
            dpdp = "d_phase_d_" + param
            dddp = "d_delay_d_" + param
            if param == 'Offset':
                M[:,ii] = 1.0
                units.append(u.s/u.s)
            elif hasattr(self, dpdp):
                q = getattr(self, dpdp)(toas) / F0
                #q = self.d_phase_d_param(toas, param) / F0
                M[:,ii] = q
                units.append(q.unit)
            elif hasattr(self, dddp):
                q = getattr(self, dddp)(toas)
                M[:,ii] = q
                units.append(q.unit)

        return M, params, units

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
        # Always include UNITS in par file. For now, PINT only supports TDB
        result += "UNITS TDB"
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
            if not parsed:
                try:
                    prefix,f,v = utils.split_prefixed_name(l.split()[0])
                    if prefix not in ignore_prefix:
                        log.warn("Unrecognized parfile line '%s'" % l)
                except:
                    if l.split()[0] not in ignore_params:
                        log.warn("Unrecognized parfile line '%s'" % l)

        # The "setup" functions contain tests for required parameters or
        # combinations of parameters, etc, that can only be done
        # after the entire parfile is read
        self.setup()


    def is_in_parfile(self,para_dict):
        """ Check if this subclass inclulded in parfile.
            Parameters
            ------------
            para_dict : dictionary
                A dictionary contain all the parameters with values in string
                from one parfile
            Return
            ------------
            True : bool
                The subclass is inculded in the parfile.
            False : bool
                The subclass is not inculded in the parfile.
        """
        pNames_inpar = para_dict.keys()

        pNames_inModel = self.params

        prefix_inModel = self.prefix_params

        # Remove the common parameter PSR
        try:
            del pNames_inModel[pNames_inModel.index('PSR')]
        except:
            pass

        # For solar system shapiro delay component
        if hasattr(self,'PLANET_SHAPIRO'):
            if "NO_SS_SHAPIRO" in pNames_inpar:
                return False
            else:
                return True


        # For Binary model component
        try:
            if getattr(self,'BinaryModelName') == para_dict['BINARY'][0]:
                return True
            else:
                return False
        except:
            pass

        # Compare the componets parameter names with par file parameters
        compr = list(set(pNames_inpar).intersection(pNames_inModel))

        if compr==[]:
            # Check aliases
            for p in pNames_inModel:
                al = getattr(self,p).aliases
                # No aliase in parameters
                if al == []:
                    continue
                # Find alise check if match any of parameter name in parfile
                if list(set(pNames_inpar).intersection(al)):
                    return True
                else:
                    continue

            # TODO Check prefix parameter

            return False

        return True

def generate_timing_model(name, components, attributes={}):

    """Build a timing model from components.

    Returns a timing model class generated from the specifiied
    sub-components.  The return value is a class type, not an instance,
    so needs to be called to generate a usable instance.  For example:

    MyModel = generate_timing_model("MyModel",(Astrometry,Spindown),{})
    my_model = MyModel()
    my_model.read_parfile("J1234+1234.par")
    """
    # Test that all the components are derived from
    # TimingModel
    try:
        numComp = len(components)
    except:
        components = (components,)

    for c in components:
        try:
            if not issubclass(c,TimingModel):
                raise(TypeError("Class "+c.__name__+
                                 " is not a subclass of TimingModel"))
        except:
            raise(TypeError("generate_timing_model() Arg 2"+
                            "has to be a tuple of classes"))

    return type(name, components, attributes)

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
