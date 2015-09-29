import astropy.units as u
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
SECS_PER_JUL_YEAR = SECS_PER_DAY*365.25

from .parameter import Parameter, MJDParameter
from .timing_model import Cache, TimingModel, MissingParameter
from ..phase import Phase
from ..utils import time_from_mjd_string, time_to_longdouble
from ..orbital.kepler import eccentric_from_mean
from .btmodel import BTmodel
import numpy as np
import time
from pint import ls,GMsun,Tsun
from pint import utils

class PSRbinary(TimingModel):
    """ A class for pulsar binary model.
        Binary variables naming:
        Eccentric Anomaly               E (not parameter ECC)
        Mean Anomaly                    M
        True Anomaly                    nu
        Eccentric                       ecc
        Longitude of periastron         omega
        projected semi-major axis of orbit   a1
    """
    def __init__(self,):
        super(PSRbinary, self).__init__()
        self.BinaryModelName = None
        self.barycentricTime = None
        #self.binary_params = ['A1','PB', 'PBDOT','PBX','XPBDOT',,'ECC', 'EDOT',
        #                      'TASC','EPS1','EPS2','EPS1DOT','EPS2DOT','OM',
        #                      'OMDOT','A1', 'A1DOT','SINI','M2','A0','B0', 'T0',
        #                      'GAMMA','DR','DTH','BP','BPP','DTHETA','XOMDOT',
        #                      'KOM','KIN','SHAPMAX','MTOT']
        self.binary_delay_funcs= []
        self.binary_params = []
        self.inter_vars = ['E','M','nu','ecc','omega','a1','TM2']
        self.add_param(Parameter(name="PB",
            units=u.day,
            description="Orbital period",
            parse_value=utils.data2longdouble),binary_param = True)


        self.add_param(Parameter(name="PBDOT",
            units=u.day/u.day,
            description="Orbital period derivitve respect to time",
            parse_value=utils.data2longdouble),binary_param = True)


        self.add_param(Parameter(name="XPBDOT",
            units=u.s/u.s,
            description="Rate of change of orbital period minus GR prediction",
            parse_value=utils.data2longdouble),binary_param = True)


        self.add_param(Parameter(name="A1",
            units=ls,
            description="Projected semi-major axis, a*sin(i)",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name = "A1DOT",
            units=ls/u.s,
            description="Derivitve of projected semi-major axis, da*sin(i)/dt",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="ECC",
            units="",
            aliases = ["E"],
            description="Eccentricity",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="EDOT",
             units="1/s",
             description="Eccentricity derivitve respect to time",
             parse_value=np.double),binary_param = True)

        self.add_param(MJDParameter(name="T0",
            parse_value=lambda x: time_from_mjd_string(x, scale='tdb'),
            description="Epoch of periastron passage"),binary_param = True)

        self.add_param(Parameter(name="OM",
            units=u.deg,
            description="Longitude of periastron",
            parse_value=lambda x: np.double(x)),binary_param = True)

        self.add_param(Parameter(name="OMDOT",
            units="deg/year",
            description="Longitude of periastron",
            parse_value=lambda x: np.double(x)),binary_param = True)

        self.add_param(Parameter(name="M2",
             units=u.M_sun,
             description="Mass of companian in the unit Sun mass",
             parse_value=np.double),binary_param = True)

        self.tt0 = None
        self.parDefault = {'PB':10.0, 'PBDOT':0.0, 'PBDOT' : 0.0, 'ECC': 0.0,
                           'EDOT':0.0, 'A1':10.0,'A1DOT':0.0, 'T0':'54000.0',
                           'OM':10.0,'OMDOT':0.0,'XPBDOT':0.0,'M2':0.0}

    def setup(self):
        super(PSRbinary, self).setup()# How to set up heres
        self.apply_units()


    def apply_units(self):
        for bpar in self.binary_params:
            bparObj = getattr(self,bpar)
            if bparObj.value == None:
                continue
            if type(bparObj).__name__ == 'MJDParameter':
                continue

            bparObj.value = bparObj.value*u.Unit(bparObj.units)

    def set_default_values(self,defaultDict):
        #TODO add alias part
        for key in defaultDict.keys():
            par = getattr(self,key)
            if par.value == None:
                if type(par).__name__ == 'MJDParameter':
                    par.set(defaultDict[key])
                else:
                    par.set(defaultDict[key],with_unit = True)


    def set_inter_vars(self,barycentricTOA):
        self.barycentricTime = barycentricTOA
        setattr(self,'tt0', self.get_tt0(barycentricTOA))
        setattr(self,'sinE',np.sin(self.E()))
        setattr(self,'cosE',np.cos(self.E()))
        setattr(self,'sinNu',np.sin(self.nu()))
        setattr(self,'cosNu',np.cos(self.nu()))
        setattr(self,'sinOmg',np.sin(self.omega()))
        setattr(self,'cosOmg',np.cos(self.omega()))
        setattr(self,'TM2',self.M2.value.value*Tsun)

    def binary_delay(self,barycentricTOA):
        """Returns total pulsar binary delay.
           Parameters
           ----------
           barycentricToa : ndarray
               barycentricTOA is the pulse arrival time at solar system
               barycenter. Unit is in MJD
           Return
           ----------
           Pulsar binary delay in the units of second
        """
        bdelay = np.longdouble(np.zeros(len(barycentricTOA)))*u.s
        self.set_inter_vars(barycentricTOA)
        for bdf in self.binary_delay_funcs:
            bdelay+= bdf()
        return bdelay

    def der(self,y,x):
        """Find the derivitives in binary model
           dy/dx
           Parameters
           ---------
           y : str
               Name of variable to be differentiated
           x : str
               Name of variable the derivitive respect to
           Return
           ---------
           dy/dx : The derivitives
        """
        if y not in self.binary_params+self.inter_vars:
            errorMesg = y + " is not in binary parameter and variables list."
            raise ValueError(errorMesg)

        if x not in self.inter_vars+self.binary_params:
            errorMesg = x + " is not in binary parameters and variables list."
            raise ValueError(errorMesg)
        # Derivitive to itself
        if x == y:
            return np.longdouble(np.ones(len(self.tt0)))*u.Unit('')
        # Get the unit right

        yAttr = getattr(self,y)
        xAttr = getattr(self,x)
        U = [None,None]
        for i,attr in enumerate([yAttr, xAttr]):
            if type(attr).__name__ == 'Parameter':  # If attr is a Parameter class type
                U[i] = u.Unit(attr.units)
            elif type(attr).__name__ == 'MJDParameter': # If attr is a MJDParameter class type
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


    @Cache.use_cache
    def compute_eccentric_anomaly(self, eccentricity, mean_anomaly):
        """compute eccentric anomaly, solve for Kepler Equation
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
        if hasattr(mean_anomaly,'unit'):
            ma = np.longdouble(mean_anomaly).value
        else:
            ma = mean_anomaly
        k = lambda E: E-e*np.sin(E)-ma   # Kepler Equation
        dk = lambda E: 1-e*np.cos(E)     # Derivitive Kepler Equation
        U = ma
        while(np.max(abs(k(U)))>5e-15):  # Newton-Raphson method
            U = U-k(U)/dk(U)
        return U*u.rad

    ####################################
    @Cache.use_cache
    def get_tt0(self,barycentricTOA):
        T0 = utils.time_to_longdouble(self.T0.value)*u.day
    	if not hasattr(barycentricTOA,'unit') or barycentricTOA.unit == None:
            barycentricTOA = barycentricTOA*u.day
        self.tt0 = (barycentricTOA - T0).to('second')
        return self.tt0

    ####################################
    @Cache.use_cache
    def ecc(self):
        ECC = self.ECC.value
        EDOT = self.EDOT.value
        return ECC + (self.tt0*EDOT).decompose()

    @Cache.use_cache
    def d_ecc_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.EDOT.value.value)
        return result*u.Unit(self.EDOT.units)

    @Cache.use_cache
    def d_ecc_d_ECC(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit("")

    @Cache.use_cache
    def d_ecc_d_EDOT(self):
        return self.tt0
    #####################################
    @Cache.use_cache
    def a1(self):
        return self.A1.value + self.tt0*self.A1DOT.value

    @Cache.use_cache
    def d_a1_d_A1(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit('')

    @Cache.use_cache
    def d_a1_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.A1DOT.value.value)
        return result*u.Unit(self.A1DOT.units)

    @Cache.use_cache
    def d_a1_d_A1DOT(self):
        return self.tt0
    ######################################
    @Cache.use_cache
    def M(self):
        """Orbit phase, this could be a generic function
        """
        PB = (self.PB.value).to('second')
        PBDOT = self.PBDOT.value
        XPBDOT = self.XPBDOT.value

        orbits = (self.tt0/PB - 0.5*(PBDOT+XPBDOT)*(self.tt0/PB)**2).decompose()
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = (orbits - norbits)*2*np.pi*u.rad

        return phase
    @Cache.use_cache
    def d_M_d_T0(self):
        """dM/dT0  this could be a generic function
        """
        PB = self.PB.value
        PBDOT = self.PBDOT.value
        XPBDOT = self.XPBDOT.value

        return ((PBDOT - XPBDOT)*self.tt0/PB-1.0)*2*np.pi*u.rad/PB

    @Cache.use_cache
    def d_M_d_PB(self):
        """dM/dPB this could be a generic function
        """
        PB = self.PB.value
        PBDOT = self.PBDOT.value
        XPBDOT = self.XPBDOT.value

        return 2*np.pi*u.rad*((PBDOT+XPBDOT)*self.tt0**2/PB**3 - self.tt0/PB**2)

    @Cache.use_cache
    def d_M_d_PBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        PB = self.PB.value
        return -np.pi*u.rad * self.tt0**2/PB**2

    @Cache.use_cache
    def d_M_d_XPBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        PB = self.PB.value
        return -np.pi*u.rad * self.tt0**2/PB**2

    ###############################################
    @Cache.use_cache
    def E(self):
        """Eccentric Anomaly
        """
        return self.compute_eccentric_anomaly(self.ecc(),self.M())

    # Analytically calculate derivtives.
    @Cache.use_cache
    def d_E_d_T0(self):
        """d(E-e*sinE)/dT0 = dM/dT0
           dE/dT0(1-cosE*e)-de/dT0*sinE = dM/dT0
           dE/dT0(1-cosE*e)+eDot*sinE = dM/dT0
        """
        RHS = self.der('M','T0')
        E = self.E()
        EDOT = self.EDOT.value
        ecc = self.ecc()
        return (RHS-EDOT*np.sin(E))/(1.0-np.cos(E)*ecc)

    @Cache.use_cache
    def d_E_d_PB(self):
        """d(E-e*sinE)/dPB = dM/dPB
           dE/dPB(1-cosE*e)-de/dPB*sinE = dM/dPB
           dE/dPB(1-cosE*e) = dM/dPB
        """
        RHS = self.d_M_d_PB()
        return RHS/(1.0-np.cos(self.E())*self.ecc())

    @Cache.use_cache
    def d_E_d_PBDOT(self):
        RHS = self.d_M_d_PBDOT()
        return RHS/(1.0-np.cos(self.E())*self.ecc())
    @Cache.use_cache
    def d_E_d_XPBDOT(self):
        RHS = self.d_M_d_XPBDOT()
        return RHS/(1.0-np.cos(self.E())*self.ecc())
    @Cache.use_cache
    def d_E_d_ECC(self):
        E = self.E()
        return np.sin(E) / (1.0 - self.ecc()*np.cos(E))
    @Cache.use_cache
    def d_E_d_EDOT(self):
        return self.tt0 * self.d_E_d_ECC()

    #####################################################
    @Cache.use_cache
    def nu(self):
        """True anomaly  (Ae)
        """
        return 2*np.arctan(np.sqrt((1.0+self.ecc())/(1.0-self.ecc()))*np.tan(self.E()/2.0))

    @Cache.use_cache
    def d_nu_d_E(self):
        brack1 = (1 + self.ecc()*np.cos(self.nu())) / (1 - self.ecc()*np.cos(self.E()))
        brack2 = np.sin(self.E()) / np.sin(self.nu())
        return brack1*brack2

    @Cache.use_cache
    def d_nu_d_ecc(self):
        return np.sin(self.E())**2/(self.ecc()*np.cos(self.E())-1)**2/np.sin(self.nu())

    @Cache.use_cache
    def d_nu_d_T0(self):
        """dnu/dT0 = dnu/de*de/dT0+dnu/dE*dE/dT0
           de/dT0 = -EDOT
        """
        return self.d_nu_d_ecc()*(-self.EDOT.value)+self.d_nu_d_E()*self.d_E_d_T0()

    @Cache.use_cache
    def d_nu_d_PB(self):
        """dnu(e,E)/dPB = dnu/de*de/dPB+dnu/dE*dE/dPB
           de/dPB = 0
           dnu/dPB = dnu/dE*dE/dPB
        """
        return self.d_nu_d_E()*self.d_E_d_PB()

    @Cache.use_cache
    def d_nu_d_PBDOT(self):
        """dnu(e,E)/dPBDOT = dnu/de*de/dPBDOT+dnu/dE*dE/dPBDOT
           de/dPBDOT = 0
           dnu/dPBDOT = dnu/dE*dE/dPBDOT
        """
        return self.d_nu_d_E()*self.d_E_d_PBDOT()
    @Cache.use_cache
    def d_nu_d_XPBDOT(self):
        """dnu/dPBDOT = dnu/dE*dE/dPBDOT
        """
        return self.d_nu_d_E()*self.d_E_d_XPBDOT()

    @Cache.use_cache
    def d_nu_d_ECC(self):
        """dnu(e,E)/dECC = dnu/de*de/dECC+dnu/dE*dE/dECC
           de/dECC = 1
           dnu/dPBDOT = dnu/dE*dE/dECC+dnu/de
        """
        return self.d_nu_d_ecc()+self.d_nu_d_E()*self.d_E_d_ECC()

    @Cache.use_cache
    def d_nu_d_EDOT(self):
        return self.tt0 * self.d_nu_d_ECC()
    #############################################
    @Cache.use_cache
    def omega(self):
        """T. Damour and N. Deruelle(1986)equation [25]
           omega = OM+nu*k
           k = OMDOT/n  (T. Damour and N. Deruelle(1986)equation between Eq 16
                         Eq 17)
        """
        PB = self.PB.value
        PB = PB.to('second')
        OMDOT = self.OMDOT.value
        OM = self.OM.value
        nu = self.nu()
        k = OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/PB)
        return (OM + nu*k).to(u.rad)

    @Cache.use_cache
    def d_omega_d_par(self,par):
        """Derivitive for omega respect to user input Parameter.
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

        PB = self.PB.value
        OMDOT = self.OMDOT.value
        OM = self.OM.value
        nu = self.nu()
        k = OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/PB)

        if par in ['OM','OMDOT','PB']:
            dername = 'd_omeg_d_' + par
            return getattr(self,dername)()
        else:
            dername = 'd_nu_d_'+par # For parameters only in nu
            if hasattr(self,dername):
                return k*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))

    @Cache.use_cache
    def d_omega_d_OM(self):
        """dOmega/dOM = 1
        """
        return np.longdouble(np.ones((len(self.tt0))))*u.Unit('')

    @Cache.use_cache
    def d_omega_d_OMDOT(self):
        """dOmega/dOMDOT = 1/n*nu
           n = 2*pi/PB
           dOmega/dOMDOT = PB/2*pi*nu
        """
        PB = (self.PB.value).to('second')
        nu = self.nu()

        return PB/(2*np.pi*u.rad)*nu


    @Cache.use_cache
    def d_omega_d_PB(self):
        """dOmega/dPB = dnu/dPB*k+dk/dPB*nu
        """
        PB = self.PB.value
        OMDOT = self.OMDOT.value
        OM = self.OM.value
        nu = self.nu()
        k = OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/PB)
        return (self.d_nu_d_PB()*k)+nu*OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad)


    ############################################################

    @Cache.use_cache
    def pbprime(self):

        return self.PB.value - self.PBDOT.value * self.tt0

    @Cache.use_cache
    def P(self):
        return self.P0.value + self.P1.value*(self.t - self.PEPOCH).to('second')

    ############################################################
    @Cache.use_cache
    def delay_designmatrix(self, params):
        npars = len(params)
        M = np.zeros((len(self.tt0), npars))

        for ii, par in enumerate(params):
            M[:,ii] = getattr(self, 'd_delay_d_par')(par)

        return M
