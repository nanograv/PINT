# This file is a prototype of independent psr binary model class
import numpy as np
import functools
import collections
from astropy import log
from pint.models.timing_model import Cache
from pint import utils as ut
import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
SECS_PER_JUL_YEAR = SECS_PER_DAY*365.25
from pint import ls,GMsun,Tsun


class PSR_BINARY(object):
    """A base (generic) object for psr binary models. In this class, a set of
    generally used binary paramters and several commonly used calculations are
    defined. For each binary model, the specific parameters and calculations
    are defined in the subclass.

    A binary model takes the solar system barycentric time (ssb time) as input.
    When a binary model is instantiated, the parameters are set to the default
    values and input time is not initialized. The update those values, method
    `.updata_input()` should be used.

    Example of build a sepcific binary model class
    -------
    >>> from pint.models.stand_alone_psr_binaries.pulsar_binary import PSR_BINARY
    >>> import numpy
    >>> class foo(PSR_BINARY):
            def __init__(self):
                # This is to initialize the mother class attributes.
                super(foo, self).__init__()
                self.binary_name = 'foo'
                # Add parameter that specific for my_binary, with default value and units
                self.param_default_value.update({'A0':0*u.second,'B0':0*u.second,
                                       'DR':0*u.Unit(''),'DTH':0*u.Unit(''),
                                       'GAMMA':0*u.second,})
                self.set_param_values() # This is to set all the parameters to attributes
                self.binary_delay_funcs += [self.foo_delay]
                self.d_binarydelay_d_par_funcs += [self.d_foo_delay_d_par]
                # If you have intermedia value in the calculation
                self.dd_interVars = ['er','eTheta','beta','alpha','Dre','Drep','Drepp',
                                     'nhat', 'TM2']
                self.add_inter_vars(self.dd_interVars)

            def foo_delay(self):
                pass

            def d_foo_delay_d_par(self):
                pass
    >>> # To build a model instance
    >>> binary_foo = foo()
    >>> # binary_foo class has the defualt parameter value without toa input.
    >>> # Update the toa input and parameters
    >>> t = numpy.linspace(54200.0,55000.0,800)
    >>> paramters_dict = {'A0':0.5,'ECC':0.01}
    >>> binary_foo.update_input(t, paramters_dict)
    >>> # Now the binary delay and derivatives can be computed.

    To acess the binary model class from pint platform, a pint pulsar binary
    wrapper is needed. See docstrings in the source code of pint/models/pul
    sar_binary class `PulsarBinary`.

    Included general parameters:
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec)
    @param A1DOT:       Time-derivative of A1 (lt-sec/sec)
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param XPBDOT:      Rate of change of orbital period minus GR prediction
    @param OMDOT:       Time-derivative of OMEGA [0.0]

    Intermedia variables calculation method are given here:
    Eccentric Anomaly               E (not parameter ECC)
    Mean Anomaly                    M
    True Anomaly                    nu
    Eccentric                       ecc
    Longitude of periastron         omega
    projected semi-major axis of orbit   a1
    TM2

    Generic methon provided:
    binary_delay()  Binary total delay
    d_binarydelay_d_par()   Derivatives respect to one parameter
    prtl_der()    partial derivatives respect to some variable
    """
    def __init__(self,):
        # Necessary parameters for all binary model
        self.binary_name = None
        self.param_default_value = {'PB':np.longdouble(10.0)*u.day,
                           'PBDOT':0.0*u.day/u.day,
                           'ECC': 0.9*u.Unit('') ,
                           'EDOT':0.0/u.second ,
                           'A1':10.0*ls,'A1DOT':0.0*ls/u.second,
                           'T0':np.longdouble(54000.0)*u.day,
                           'OM':10.0*u.deg,
                           'OMDOT':0.0*u.deg/u.year,
                           'XPBDOT':0.0*u.day/u.day,
                           'M2':0.0*u.M_sun,
                           'SINI':0*u.Unit('') }
        self.param_aliases = {'ECC':['E'],'EDOT':['ECCDOT'],
                              'A1DOT':['XDOT']}
        self.binary_params = self.param_default_value.keys()
        self.inter_vars = ['E','M','nu','ecc','omega','a1','TM2']
        self.binary_delay_funcs = []
        self.d_binarydelay_d_par_funcs = []

    @property
    def t(self):
        return self._t
    @t.setter
    def t(self,val):
        self._t = val
        if hasattr(self,'T0'):
            self._tt0 = self.get_tt0(self._t)

    @property
    def T0(self):
        return self._T0
    @T0.setter
    def T0(self,val):
        self._T0 = val
        if hasattr(self, '_t'):
            self._tt0 = self.get_tt0(self._t)
    @property
    def tt0(self):
        return self._tt0

    def update_input(self, barycentric_toa=None, param_dict=None):
        """ A function updates the toas and parameters
        """
        # Update toas
        if barycentric_toa is not None:
            if not isinstance(barycentric_toa, np.ndarray) and \
               not isinstance(barycentric_toa, list):
                self.t = np.array([barycentric_toa,])
            else:
                self.t = barycentric_toa
        # Update parameters
        if param_dict is not None:
            self.set_param_values(param_dict)

    def set_param_values(self, valDict = None):
        """A function that sets the parameters and assign values
        If the valDict is not provided, it will set parameter as default value
        """
        if valDict is None:
            for par in self.param_default_value.keys():
                setattr(self,par.upper(),self.param_default_value[par])
        else:
            for par in valDict.keys():
                if par not in self.binary_params: # search for aliases
                    par = self.search_alias(par)
                    if par is None:
                        raise AttributeError('Can not find parameter '+par+' in '\
                                              + self.binary_name+'model')
                setattr(self,par,valDict[par])

    def add_binary_params(self,parameter,defaultValue):
        """Add one parameter to the binary class
        """
        if parameter not in self.binary_params:
            self.binary_params.append(p)
            if not hasattr(defaultValue,unit):
                log.warning("Binary paramters' value generally has unit."\
                            " Treat paramter "+paramter+" as a diemension less unit.")
                self.param_default_value[p] = defaultValue*u.Unit("")
            else:
                self.param_default_value[p] = defaultValue
            setattr(self, parameter, self.param_default_value[p])


    def add_inter_vars(self,interVars):
        if not isinstance(interVars,list):
            interVars = [interVars,]
        for v in interVars:
            if v not in self.inter_vars:
                self.inter_vars.append(v)

    def search_alias(self,parname):
        for pn in self.param_aliases.keys():
            if parname in self.param_aliases[pn]:
                return pn
            else:
                return None

    def binary_delay(self):
        """Returns total pulsar binary delay.
        Return
        ----------
        Pulsar binary delay in the units of second
        """
        bdelay = np.longdouble(np.zeros(len(self.t)))*u.s
        for bdf in self.binary_delay_funcs:
            bdelay+= bdf()
        return bdelay

    @Cache.use_cache
    def d_binarydelay_d_par(self, par):
        """Get the binary delay derivatives respect to parameters.
        Parameter
        ---------
        par : str
            Parameter name.
        """
        # search for aliases
        if par not in self.binary_params and self.search_alias(par) is None:
            raise AttributeError('Can not find parameter '+par+' in '\
                                 + self.binary_name+' model')
        # Get first derivative in the delay derivative function
        result = self.d_binarydelay_d_par_funcs[0](par)
        if len(self.d_binarydelay_d_par_funcs) > 1:
            for df in self.d_binarydelay_d_par_funcs[1,:]:
                result += df(par)

        return result

    @Cache.use_cache
    def prtl_der(self,y,x):
        """Find the partial derivatives in binary model
           pdy/pdx
           Parameters
           ---------
           y : str
               Name of variable to be differentiated
           x : str
               Name of variable the derivative respect to
           Return
           ---------
           pdy/pdx : The derivatives
        """
        if y not in self.binary_params+self.inter_vars:
            errorMesg = y + " is not in binary parameter and variables list."
            raise ValueError(errorMesg)

        if x not in self.inter_vars+self.binary_params:
            errorMesg = x + " is not in binary parameters and variables list."
            raise ValueError(errorMesg)
        # derivative to itself
        if x == y:
            return np.longdouble(np.ones(len(self.tt0)))*u.Unit('')
        # Get the unit right

        yAttr = getattr(self,y)
        xAttr = getattr(self,x)
        U = [None,None]
        for i,attr in enumerate([yAttr, xAttr]):
            if type(attr).__name__ == 'Parameter':  # If attr is a PINT Parameter class type
                U[i] = u.Unit(attr.units)
            elif type(attr).__name__ == 'MJDParameter': # If attr is a PINT MJDParameter class type
                U[i] = u.Unit('day')
            elif hasattr(attr,'unit'):  # If attr is a Quantity type
                U[i] = attr.unit
            elif hasattr(attr,'__call__'):  # If attr is a method
                U[i] = attr().unit
            else:
                raise TypeError(type(attr)+'can not get unit')
            U[i] = 1*U[i]

            commonU = list(set(U[i].unit.bases).intersection([u.rad,u.deg]))
            if commonU != []:
                strU = U[i].unit.to_string()
                for cu in commonU:
                    scu = cu.to_string()
                    strU = strU.replace(scu,'1')
                U[i] = U[i].to(strU, equivalencies=u.dimensionless_angles())

        yU = U[0]
        xU = U[1]
        # Call derivtive functions
        derU =  ((yU/xU).decompose()).unit

        if hasattr(self,'d_'+y+'_d_'+x):
            dername = 'd_'+y+'_d_'+x
            result = getattr(self,dername)()

        elif hasattr(self,'d_'+y+'_d_par'):
            dername = 'd_'+y+'_d_par'
            result = getattr(self,dername)(x)

        else:
           result = np.longdouble(np.zeros(len(self.tt0)))

        if hasattr(result,'unit'):
            return result.to(derU,equivalencies=u.dimensionless_angles())
        else:
            return (result*derU).decompose()


    @Cache.cache_result
    def compute_eccentric_anomaly(self, eccentricity, mean_anomaly):
        """compute eccentric anomaly, solve for Kepler Equation,
        E - e * sin(E) = M
        Parameter
        ----------
        eccentricity : array_like
            Eccentricity of binary system
        mean_anomaly : array_like
            Mean anomaly of the binary system
        Returns
        -------
        array_like
            The eccentric anomaly in radians, given a set of mean_anomalies
            in radians.
        """
        if hasattr(eccentricity,'unit'):
            e = np.longdouble(eccentricity).value
        else:
            e = eccentricity

        if any(e<0) or any(e>=1):
            raise ValueError('Eccentricity should be in the range of [0,1).')

        if hasattr(mean_anomaly,'unit'):
            ma = np.longdouble(mean_anomaly).value
        else:
            ma = mean_anomaly
        k = lambda E: E-e*np.sin(E)-ma   # Kepler Equation
        dk = lambda E: 1-e*np.cos(E)     # derivative Kepler Equation
        U = ma
        while(np.max(abs(k(U)))>5e-15):  # Newton-Raphson method
            U = U-k(U)/dk(U)
        return U*u.rad


        ####################################
    @Cache.cache_result
    def get_tt0(self,barycentricTOA):
        """
        tt0 = barycentricTOA - T0
        """
        if barycentricTOA is None or self.T0 is None:
            tt0 = None
            return tt0
        T0 =  self.T0
    	if not hasattr(barycentricTOA,'unit') or barycentricTOA.unit == None:
            barycentricTOA = barycentricTOA*u.day
        tt0 = (barycentricTOA - T0).to('second')
        return tt0

    ####################################
    @Cache.cache_result
    def ecc(self):
        """Calculate ecctricity with EDOT
        """
        ECC = self.ECC
        EDOT = self.EDOT
        return ECC + (self.tt0*EDOT).decompose()

    @Cache.cache_result
    def d_ecc_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.EDOT.value)
        return result*u.Unit(self.EDOT.unit)

    @Cache.cache_result
    def d_ecc_d_ECC(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit("")

    @Cache.cache_result
    def d_ecc_d_EDOT(self):
        return self.tt0

    #####################################
    @Cache.cache_result
    def a1(self):
        return self.A1 + self.tt0*self.A1DOT

    @Cache.cache_result
    def d_a1_d_A1(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit('')

    @Cache.cache_result
    def d_a1_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.A1DOT.value)
        return result*u.Unit(self.A1DOT.unit)

    @Cache.cache_result
    def d_a1_d_A1DOT(self):
        return self.tt0
    ######################################
    @Cache.cache_result
    def M(self):
        """Orbit phase, this could be a generic function
        """
        PB = (self.PB).to('second')
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT

        orbits = (self.tt0/PB - 0.5*(PBDOT+XPBDOT)*(self.tt0/PB)**2).decompose()
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = (orbits - norbits)*2*np.pi*u.rad

        return phase

    @Cache.cache_result
    def d_M_d_T0(self):
        """dM/dT0  this could be a generic function
        """
        PB = self.PB.to('second')
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT

        return ((PBDOT - XPBDOT)*self.tt0/PB-1.0)*2*np.pi*u.rad/PB

    @Cache.cache_result
    def d_M_d_PB(self):
        """dM/dPB this could be a generic function
        """
        PB = self.PB.to('second')
        PBDOT = self.PBDOT
        XPBDOT = self.XPBDOT
        return 2*np.pi*u.rad*((PBDOT+XPBDOT)*self.tt0**2/PB**3 - self.tt0/PB**2)

    @Cache.cache_result
    def d_M_d_PBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        PB = self.PB.to('second')
        return -np.pi*u.rad * self.tt0**2/PB**2

    @Cache.cache_result
    def d_M_d_XPBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        PB = self.PB.to('second')
        return -np.pi*u.rad * self.tt0**2/PB**2

    ###############################################
    @Cache.cache_result
    def E(self):
        """Eccentric Anomaly
        """
        return self.compute_eccentric_anomaly(self.ecc(),self.M())

    # Analytically calculate derivtives.
    @Cache.cache_result
    def d_E_d_T0(self):
        """d(E-e*sinE)/dT0 = dM/dT0
           dE/dT0(1-cosE*e)-de/dT0*sinE = dM/dT0
           dE/dT0(1-cosE*e)+eDot*sinE = dM/dT0
        """
        RHS = self.prtl_der('M','T0')
        E = self.E()
        EDOT = self.EDOT
        ecc = self.ecc()
        return (RHS-EDOT*np.sin(E))/(1.0-np.cos(E)*ecc)

    @Cache.cache_result
    def d_E_d_PB(self):
        """d(E-e*sinE)/dPB = dM/dPB
           dE/dPB(1-cosE*e)-de/dPB*sinE = dM/dPB
           dE/dPB(1-cosE*e) = dM/dPB
        """
        RHS = self.d_M_d_PB()
        return RHS/(1.0-np.cos(self.E())*self.ecc())

    @Cache.cache_result
    def d_E_d_PBDOT(self):
        RHS = self.d_M_d_PBDOT()
        return RHS/(1.0-np.cos(self.E())*self.ecc())

    @Cache.cache_result
    def d_E_d_XPBDOT(self):
        RHS = self.d_M_d_XPBDOT()
        return RHS/(1.0-np.cos(self.E())*self.ecc())

    @Cache.cache_result
    def d_E_d_ECC(self):
        E = self.E()
        return np.sin(E) / (1.0 - self.ecc()*np.cos(E))

    @Cache.cache_result
    def d_E_d_EDOT(self):
        return self.tt0 * self.d_E_d_ECC()

    #####################################################
    @Cache.cache_result
    def nu(self):
        """True anomaly  (Ae)
        """
        return 2*np.arctan(np.sqrt((1.0+self.ecc())/(1.0-self.ecc()))*np.tan(self.E()/2.0))

    @Cache.cache_result
    def d_nu_d_E(self):
        brack1 = (1 + self.ecc()*np.cos(self.nu())) / (1 - self.ecc()*np.cos(self.E()))
        brack2 = np.sin(self.E()) / np.sin(self.nu())
        return brack1*brack2

    @Cache.cache_result
    def d_nu_d_ecc(self):
        return np.sin(self.E())**2/(self.ecc()*np.cos(self.E())-1)**2/np.sin(self.nu())

    @Cache.cache_result
    def d_nu_d_T0(self):
        """dnu/dT0 = dnu/de*de/dT0+dnu/dE*dE/dT0
           de/dT0 = -EDOT
        """
        return self.d_nu_d_ecc()*(-self.EDOT)+self.d_nu_d_E()*self.d_E_d_T0()

    @Cache.cache_result
    def d_nu_d_PB(self):
        """dnu(e,E)/dPB = dnu/de*de/dPB+dnu/dE*dE/dPB
           de/dPB = 0
           dnu/dPB = dnu/dE*dE/dPB
        """
        return self.d_nu_d_E()*self.d_E_d_PB()

    @Cache.cache_result
    def d_nu_d_PBDOT(self):
        """dnu(e,E)/dPBDOT = dnu/de*de/dPBDOT+dnu/dE*dE/dPBDOT
           de/dPBDOT = 0
           dnu/dPBDOT = dnu/dE*dE/dPBDOT
        """
        return self.d_nu_d_E()*self.d_E_d_PBDOT()

    @Cache.cache_result
    def d_nu_d_XPBDOT(self):
        """dnu/dPBDOT = dnu/dE*dE/dPBDOT
        """
        return self.d_nu_d_E()*self.d_E_d_XPBDOT()

    @Cache.cache_result
    def d_nu_d_ECC(self):
        """dnu(e,E)/dECC = dnu/de*de/dECC+dnu/dE*dE/dECC
           de/dECC = 1
           dnu/dPBDOT = dnu/dE*dE/dECC+dnu/de
        """
        return self.d_nu_d_ecc()+self.d_nu_d_E()*self.d_E_d_ECC()

    @Cache.cache_result
    def d_nu_d_EDOT(self):
        return self.tt0 * self.d_nu_d_ECC()

    #############################################
    @Cache.cache_result
    def omega(self):
        """T. Damour and N. Deruelle(1986)equation [25]
           omega = OM+nu*k
           k = OMDOT/n  (T. Damour and N. Deruelle(1986)equation between Eq 16
                         Eq 17)
        """
        PB = self.PB
        PB = PB.to('second')
        OMDOT = self.OMDOT
        OM = self.OM
        nu = self.nu()
        k = OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/PB)
        return (OM + nu*k).to(u.rad)

    @Cache.cache_result
    def d_omega_d_par(self,par):
        """derivative for omega respect to user input Parameter.
           if par is not 'OM','OMDOT','PB'
           dOmega/dPar =  k*dAe/dPar
           k = OMDOT/n
           Parameters
           ----------
           par : string
                 parameter name
           Return
           ----------
           Derivitve of omega respect to par
        """
        if par not in self.binary_params:
            errorMesg = par + "is not in binary parameter list."
            raise ValueError(errorMesg)

        PB = self.PB
        OMDOT = self.OMDOT
        OM = self.OM
        nu = self.nu()
        k = OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/PB)

        if par in ['OM','OMDOT','PB']:
            dername = 'd_omega_d_' + par
            return getattr(self,dername)()
        else:
            dername = 'd_nu_d_'+par # For parameters only in nu
            if hasattr(self,dername):
                return k*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))

    @Cache.cache_result
    def d_omega_d_OM(self):
        """dOmega/dOM = 1
        """
        return np.longdouble(np.ones((len(self.tt0))))*u.Unit('')

    @Cache.cache_result
    def d_omega_d_OMDOT(self):
        """dOmega/dOMDOT = 1/n*nu
           n = 2*pi/PB
           dOmega/dOMDOT = PB/2*pi*nu
        """
        PB = (self.PB).to('second')
        nu = self.nu()

        return PB/(2*np.pi*u.rad)*nu

    @Cache.cache_result
    def d_omega_d_PB(self):
        """dOmega/dPB = dnu/dPB*k+dk/dPB*nu
        """
        PB = self.PB
        OMDOT = self.OMDOT
        OM = self.OM
        nu = self.nu()
        k = OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/PB)
        return (self.d_nu_d_PB()*k)+nu*OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad)

    ############################################################
    @Cache.cache_result
    def TM2(self):
        return self.M2.value*Tsun

    def d_TM2_d_M2(self):
        return Tsun/(1.0*u.Msun)
    ###########################################################
    @Cache.cache_result
    def pbprime(self):

        return self.PB - self.PBDOT * self.tt0

    @Cache.cache_result
    def P(self):
        return self.P0 + self.P1*(self.t - self.PEPOCH).to('second')
