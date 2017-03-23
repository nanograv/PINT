# timing_model.py
# Defines the basic timing model interface classes
import functools
from .parameter import strParameter
from ..phase import Phase
from astropy import log
import astropy.time as time
import numpy as np
import pint.utils as utils
import astropy.units as u
from astropy.table import Table
import copy
import abc
from six import add_metaclass

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


class ModelMeta(abc.ABCMeta):
    """
    This is a Meta class for timing model registeration. In order ot get a
    timing model registered, a member called 'register' has to be set true in the
    TimingModel subclass.
    """
    def __init__(cls, name, bases, dct):
        regname = '_model_list'
        if not hasattr(cls,regname):
            setattr(cls,regname,{})
        if 'register' in dct:
            if cls.register:
                getattr(cls,regname)[name] = cls
        super(ModelMeta, cls).__init__(name, bases, dct)


@add_metaclass(ModelMeta)
class TimingModel(object):
    """
    Base-level object provides an interface for implementing pulsar timing
    models. It contains several over all wrapper methods.

    Notes
    -----
    PINT models pulsar pulse time of arrival at observer from its emission process and
    propagation to observer. Emission generally modeled as pulse 'Phase' and propagation.
    'time delay'. In pulsar timing different astrophysics phenomenons are separated to
    time model components for handling a specific emission or propagation effect.

    All timing model component classes should subclass this timing model base class.
    Each timing model component generally requires the following parts:
        Timing Parameters
        Delay/Phase functions which implements the time delay and phase.
        Derivatives of delay and phase respect to parameter for fitting toas.
    Each timing parameters are stored as TimingModel attribute in the type of `pint.model.parameter`
    delay or phase and its derivatives are implemented as TimingModel Methods.

    Attributes
    ----------
    params : list
        A list of all the parameter names.
    prefix_params : list
        A list of prefixed parameter names.
    delay_funcs : dict
        All the delay functions implemented in timing model. The delays do not
        need barycentric toas are placed under the 'L1' keys as a list of methods,
        the ones needs barycentric toas are under the 'L2' delay. This will be improved
        in the future. One a delay method is defined in model component, it should
        get registered in this dictionary.
    phase_funcs : list
        All the phase functions implemented in timing model. Once a phase method is defined
        in model component, it should get registered in this list.
    delay_derivs : list
        All the delay derivatives respect to timing parameters.
        Once a delay derivative method is defined in model component, it should get registered in this list.
    phase_derivs : list
        All the phase derivatives respect to timing parameters.
        Once a phase derivative method is defined in model component, it should get registered in this list.
    phase_derivs_wrt_delay : list
        All the phase derivatives respect to delay.
    """

    def __init__(self):
        self.params = []  # List of model parameter names
        self.prefix_params = []  # List of model parameter names
        self.delay_funcs = {'L1':[],'L2':[]} # List of delay component functions
        # L1 is the first level of delays. L1 delay does not need barycentric toas
        # After L1 delay, the toas have been corrected to solar system barycenter.
        # L2 is the second level of delays. L2 delay need barycentric toas

        self.phase_funcs = [] # List of phase component functions
        self.cache = None
        self.add_param(strParameter(name="PSR",
            description="Source name",
            aliases=["PSRJ", "PSRB"]))
        self.model_type = None
        self.delay_derivs = {}
        self.phase_derivs = {}
        self.phase_derivs_wrt_delay = []
        self.order_number = None
        self.print_par_func = ''
    def setup(self):
        """This is a abstract class for setting up timing model class. It is designed for
        reading .par file and check parameters.
        """
        pass


    def add_param(self, param,binary_param = False):
        """Add a parameter to the timing model. If it is a prefixe parameter,
           it will add prefix information to the prefix information attributes.
        """
        setattr(self, param.name, param)
        self.params += [param.name,]

        if binary_param is True:
            self.binary_params +=[param.name,]
        if hasattr(self, '_param_table'):
            param_row = [param.name, ]

    def remove_param(self, param):
        delattr(self, param)
        self.params.remove(param)
        if param in self.binary_params:
            self.binary_params.remove(param)

    def set_special_params(self, spcl_params):
        als = []
        for p in spcl_params:
            als += getattr(self, p).aliases
        spcl_params += als
        self.model_special_params = spcl_params


    def param_help(self):
        """Print help lines for all available parameters in model.
        """
        s = "Available parameters for %s\n" % self.__class__
        for par in self.params:
            s += "%s\n" % getattr(self, par).help_line()
        return s

    def get_params_of_type(self, param_type):
        """ Get all the parameters in timing model for one specific type
        """
        result = []
        for p in self.params:
            par = getattr(self, p)
            par_type = type(par).__name__
            par_prefix = par_type[:-9]
            if param_type.upper() == par_type.upper() or \
                param_type.upper() == par_prefix.upper():
                result.append(par.name)
        return result

    #@Cache.use_cache
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
            par = getattr(self, parname)
            if par.is_prefix == True and par.prefix == prefix:
                mapping[par.index] = parname
        return mapping

    def match_param_aliases(self, alias):
        p_aliases = {}
        # if alias is a parameter name, return itself
        if alias in self.params:
            return alias
        # get all the aliases
        for p in self.params:
            par = getattr(self, p)
            if par.aliases !=[]:
                p_aliases[p] = par.aliases
        # match alias
        for pa, pav in zip(p_aliases.keys(), p_aliases.values()):
            if alias in pav:
                return pa
        # if not found any thing.
        return ''

    def sort_model_components(self):
        # initiate the sorted_components
        sorted_list = ['']*len(list(self.components.keys()))
        in_placed = []
        not_in_placed = []
        for cp, cpv in self.components.items():
            if cpv.order_number is not None:
                sorted_list[cpv.order_number] = cp
                in_placed.append(cp)
            else:
                not_in_placed.append(cp)
        for nicp in not_in_placed:
            idx = sorted_list.index('')
            sorted_list[idx] = nicp
        return sorted_list

    #@Cache.use_cache
    def phase(self, toas):
        """Return the model-predicted pulse phase for the given TOAs."""
        # First compute the delays to "pulsar time"
        delay = self.delay(toas)
        phase = Phase(np.zeros(len(toas)), np.zeros(len(toas)))
        # Then compute the relevant pulse phases
        for pf in self.phase_funcs:
            phase += Phase(pf(toas, delay))
        return phase

    #@Cache.use_cache
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

    #@Cache.use_cache
    def get_barycentric_toas(self,toas):
        toasObs = toas['tdbld']
        delay = np.zeros(len(toas))
        for df in self.delay_funcs['L1']:
            delay += df(toas)
        toasBary = toasObs*u.day - delay*u.second
        return toasBary

    def _make_delay_derivative_funcs(self, param, function, name_tplt):
        """This function is a method to help make the delay derivatives wrt timing
        model parameters use the parameter specific name and only have the toas
        table as input.
        Parameter
        ----------
        param: str
            Name of parameter
        function: method
            The method to compute the delay derivatives. It is generally in the
            formate of function(toas, parameter)
        name_tplt: str
            The name template of the new function. The parameter name will be
            added in the end. For example: 'd_delay_d_' is a name template.
        Return
        ---------
        A delay derivative function wrt to input parameter name with the input
        name template and parameter name.
        """
        def deriv_func(toas):
            return function(toas, param)
        deriv_func.__name__ = name_tplt + param
        deriv_func.__doc__ = "Delay derivative wrt " + param + " \n"
        deriv_func.__doc__ += "Parameter\n----------\ntoas: TOA table\n    "
        deriv_func.__doc__ += "TOA point where the derivative is evaluated at.\n"
        deriv_func.__doc__ += "Return\n---------\n Delay derivatives wrt " + param
        deriv_func.__doc__ += " at toa."
        setattr(self, deriv_func.__name__, deriv_func)

    def _make_phase_derivative_funcs(self, param, function, name_tplt):
        """This function is a method to help make the phase derivatives wrt timing
        model parameters use the parameter specific name and only have the toas
        table as input.
        Parameter
        ----------
        param: str
            Name of parameter
        function: method
            The method to compute the phase derivatives. It is generally in the
            formate of 'function(toas, parameter, delay)'
        name_tplt: str
            The name template of the new function. The parameter name will be
            added in the end. For example: 'd_phase_d_' is a name template.
        Return
        ---------
        A phase derivative function wrt to input parameter name with the input
        name template and parameter name.
        """
        def deriv_func(toas, delay):
            return function(toas, param, delay)
        deriv_func.__name__ = name_tplt + param
        deriv_func.__doc__ = "Phase derivative wrt " + param + " \n"
        deriv_func.__doc__ += "Parameter\n----------\ntoas: TOA table\n    "
        deriv_func.__doc__ += "TOA point where the derivative is evaluated at.\n"
        deriv_func.__doc__ += "delay: numpy array\n    Time delay for phase calculation.\n"
        deriv_func.__doc__ += "Return\n---------\n Phase derivatives wrt " + param
        deriv_func.__doc__ += " at toa."
        setattr(self, deriv_func.__name__, deriv_func)

    def register_deriv_funcs(self, func, deriv_type, param=''):
        """
        This is a function to register the derivative function in to the
        deriv_func dictionaries.
        Parameter
        ---------
        func: method
            The method calculates the derivative
        deriv_type: str ['delay', 'phase', 'd_phase_d_delay']
            Flag for different type of derivatives. It only accepts the three
            above.
        param: str, if for d_phase_d_delay it is optional
            Name of parameter the derivative respect to
        """
        if deriv_type == 'd_phase_d_delay':
            self.phase_derivs_wrt_delay += [func,]
        elif deriv_type == 'delay':
            pn = self.match_param_aliases(param)
            if pn == '':
                raise ValueError("Parameter '%s' in not in the model." % param)
            if pn not in self.delay_derivs.keys():
                self.delay_derivs[pn] = [func,]
            else:
                self.delay_derivs[pn] += [func,]
        elif deriv_type == 'phase':
            pn = self.match_param_aliases(param)
            if pn == '':
                raise ValueError("Parameter '%s' in not in the model." % param)
            if pn not in self.phase_derivs.keys():
                self.phase_derivs[pn] = [func,]
            else:
                self.phase_derivs[pn] += [func,]

    def d_phase_d_tpulsar(self, toas):
        """Return the derivative of phase wrt time at the pulsar.

        NOT implemented yet.
        """
        pass

    def d_phase_d_toa(self, toas, sample_step=None):
        """Return the derivative of phase wrt TOA
        Parameter
        ---------
        toas : PINT TOAs class
            The toas when the derivative of phase will be evaluated at.
        sample_step : float optional
            Finite difference steps. If not specified, it will take 1/10 of the
            spin period.
        """
        copy_toas = copy.deepcopy(toas)
        if sample_step is None:
            pulse_period = 1.0 / self.F0.value
            sample_step = pulse_period * 1000
        sample_dt = [-sample_step, 2 * sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = ([dt] * copy_toas.ntoas) * u.s
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT)
            phase = self.phase(copy_toas.table)
            sample_phase.append(phase)
        # Use finite difference method.
        # phase'(t) = (phase(t+h)-phase(t-h))/2+ 1/6*F2*h^2 + ..
        # The error should be near 1/6*F2*h^2
        dp = (sample_phase[1] - sample_phase[0])
        d_phase_d_toa = dp.int / (2*sample_step) + dp.frac / (2*sample_step)
        del copy_toas
        return d_phase_d_toa


    #@Cache.use_cache
    def d_phase_d_param(self, toas, delay, param):
        """ Return the derivative of phase with respect to the parameter.
        """
        result = 0.0
        par = getattr(self, param)
        # TODO need to do correct chain rule stuff wrt delay derivs, etc
        # Is it safe to assume that any param affecting delay only affects
        # phase indirectly (and vice-versa)??
        result = np.longdouble(np.zeros(len(toas))) * u.Unit('')/par.units
        param_phase_derivs = []
        if param in self.phase_derivs.keys():
            for df in self.phase_derivs[param]:
                if df.__name__.endswith(param):
                    result += df(toas, delay).to(result.unit,
                                         equivalencies=u.dimensionless_angles())
                else: # Then this is a general derivative function.
                    result += df(toas, param, delay).to(result.unit,
                                         equivalencies=u.dimensionless_angles())
        else: # Apply chain rule for the parameters in the delay.
            d_delay_d_p = self.d_delay_d_param(toas, param)
            for dpddf in self.phase_derivs_wrt_delay:
                result += (dpddf(toas, delay) * d_delay_d_p).to(result.unit,
                                         equivalencies=u.dimensionless_angles())
        return result

    #@Cache.use_cache
    def d_phase_d_param_num(self, toas, param, step=1e-2):
        """ Return the derivative of phase with respect to the parameter.
        """
        # TODO : We need to know the range of parameter.
        par = getattr(self, param)
        ori_value = par.value
        unit = par.units
        if ori_value == 0:
            h = 1.0 * step
        else:
            h = ori_value * step
        parv = [par.value-h, par.value+h]

        phaseI = np.zeros((len(toas),2))
        phaseF = np.zeros((len(toas),2))
        for ii, val in enumerate(parv):
            par.value = val
            ph = self.phase(toas)
            phaseI[:,ii] = ph.int
            phaseF[:,ii] = ph.frac
        resI = (- phaseI[:,0] + phaseI[:,1])
        resF = (- phaseF[:,0] + phaseF[:,1])
        result = (resI + resF)/(2.0 * h)
        # shift value back to the original value
        par.value = ori_value
        return result * u.Unit("")/unit

    def d_delay_d_param_num(self, toas, param, step=1e-2):
        """ Return the derivative of phase with respect to the parameter.
        """
        # TODO : We need to know the range of parameter.
        par = getattr(self, param)
        ori_value = par.value
        if ori_value is None:
             # A parameter did not get to use in the model
            log.warn("Parameter '%s' is not used by timing model." % param)
            return np.zeros(len(toas)) * (u.second/par.units)
        unit = par.units
        if ori_value == 0:
            h = 1.0 * step
        else:
            h = ori_value * step
        parv = [par.value-h, par.value+h]
        delay = np.zeros((len(toas),2))
        for ii, val in enumerate(parv):
            par.value = val
            try:
                delay[:,ii] = self.delay(toas)
            except:
                par.value = ori_value
                raise
        d_delay = (-delay[:,0] + delay[:,1])/2.0/h
        par.value = ori_value
        return d_delay * (u.second/unit)

    def d_delay_d_param(self, toas, param):
        """
        Return the derivative of delay with respect to the parameter.
        """
        par = getattr(self, param)
        result = np.longdouble(np.zeros(len(toas)) * u.s/par.units)
        if param not in self.delay_derivs.keys():
            raise AttributeError("Derivative function for '%s' is not provided"
                                 " or not registred. "%param)
        for df in self.delay_derivs[param]:
            # The derivative function is for a specific parameter.
            if df.__name__.endswith(param):
                result += df(toas).to(result.unit, equivalencies=u.dimensionless_angles())
            else: # Then this is a general derivative function.
                result += df(toas, param).to(result.unit, equivalencies=u.dimensionless_angles())
        return result

    #@Cache.use_cache
    def designmatrix(self, toas, scale_by_F0=True, incfrozen=False, incoffset=True):
        """
        Return the design matrix: the matrix with columns of d_phase_d_param/F0
        or d_toa_d_param
        """
        params = ['Offset',] if incoffset else []
        params += [par for par in self.params if incfrozen or
                not getattr(self, par).frozen]

        F0 = self.F0.quantity        # 1/sec
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
            if param == 'Offset':
                M[:,ii] = 1.0
                units.append(u.s/u.s)
            else:
                # NOTE Here we have negative sign here. Since in pulsar timing
                # the residuals are calculated as (Phase - int(Phase)), which is different
                # from the conventional defination of least square definetion (Data - model)
                # We decide to add minus sign here in the design matrix, so the fitter
                # keeps the conventional way.
                q = - self.d_phase_d_param(toas, delay,param)
                M[:,ii] = q
                units.append(u.Unit("")/ getattr(self, param).units)

        if scale_by_F0:
            mask = []
            for ii, un in enumerate(units):
                if params[ii] == 'Offset':
                    continue
                units[ii] = un * u.second
                mask.append(ii)
            M[:, mask] /= F0.value
        return M, params, units, scale_by_F0

    def __str__(self):
        result = ""
        for par in self.params:
            result += str(getattr(self, par)) + "\n"
        return result

    def print_param_control(self, control_info={'UNITS': 'TDB', 'TIMEEPH':'FB90'},
                          order=['UNITS', 'TIMEEPH']):
        result = ""
        for pc in order:
            if pc not in control_info.keys():
                continue
            result += pc + ' ' + control_info[pc] + '\n'
        return result

    def print_param_component(self, component_name):
        result = ''
        if component_name not in self.components:
            return result
        else:
            if hasattr(self, self.components[component_name].param_print_func):
                result += getattr(self, self.components[component_name].param_print_func)()
            else:
                for p in self.components[component_name].params:
                    par = getattr(self, p)
                    if par.quantity is not None:
                        result += par.as_parfile_line()
        return result

    def as_parfile(self):
        """Returns a parfile representation of the entire model as a string."""
        result = ""
        result += self.PSR.as_parfile_line()
        sort_comps = self.sort_model_components()
        for scp in sort_comps:
            result += self.print_param_component(scp)
        result += self.print_param_control()
        return result

    def read_parfile(self, filename):
        """Read values from the specified parfile into the model parameters."""
        checked_param = []
        repeat_param = {}
        pfile = open(filename, 'r')
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue
            # Skip commented lines
            if l.startswith('#') or l[:2]=="C ":
                continue

            k = l.split()
            name = k[0].upper()

            if name in checked_param:
                if name in repeat_param.keys():
                    repeat_param[name] += 1
                else:
                    repeat_param[name] = 2
                k[0] = k[0] + str(repeat_param[name])
                l = ' '.join(k)

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

            checked_param.append(name)
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
        pNames_inpar = list(para_dict.keys())

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
            if getattr(self,'binary_model_name') == para_dict['BINARY'][0]:
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

class ComponentInfo(object):
    """
    This is a class to save the components information before it get combined
    to a specific timing model
    Parameter
    ---------
    comp : TimingModel component object
        The timing model component object
    """
    def __init__(self, comp):
        self.name = comp.__class__.__name__
        comp.params.remove('PSR')
        self.params = comp.params
        #NOTE This should be changed in the future
        self.param_print_func = comp.print_par_func
        self.order_number = comp.order_number
    def is_param_in_component(self, param):
        if param in self.params:
            return True
        else:
            match = False
            try:
                p, i, iv = utils.split_prefixed_name(param)
                for cp in self.params:
                    if cp.startswith(p):
                        try:
                            cpp, cpi, cpiv = utils.split_prefixed_name(param)
                            if cpp == p:
                                match = True
                                break
                        except:
                            continue
                return match
            except:
                return match

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
    cps = {}
    for c in components:
        try:
            if not issubclass(c,TimingModel):
                raise(TypeError("Class "+c.__name__+
                                 " is not a subclass of TimingModel"))
        except:
            raise(TypeError("generate_timing_model() Arg 2"+
                            "has to be a tuple of classes"))
        cp = c()
        cpi = ComponentInfo(cp)
        cps[cpi.name] = cpi
    attributes['components'] = cps

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
