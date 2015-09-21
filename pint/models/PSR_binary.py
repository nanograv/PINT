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
        self.binary_delays = []
        self.binary_params = []
        self.inter_vars = ['E','M','nu','ecc','om','a1']
        self.add_param(Parameter(name="PB",
            units=u.day,
            description="Orbital period",
            parse_value=np.longdouble),binary_param = True)


        self.add_param(Parameter(name="PBDOT",
            units=u.s/u.s,
            description="Orbital period derivitve respect to time",
            parse_value=np.longdouble),binary_param = True)


        self.add_param(Parameter(name="XPBDOT",
            units=u.s/u.s,
            description="Rate of change of orbital period minus GR prediction",
            parse_value=np.longdouble),binary_param = True)


        self.add_param(Parameter(name="A1",
            units=ls,
            description="Projected semi-major axis, a*sin(i)",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name = "A1DOT",
            units=ls/u.s,
            description="Derivitve of projected semi-major axis, da*sin(i)/dt",
            parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="E",
            units="",
            aliases = ["ECC"],
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

        self.tt0 = None
        self.set_default_values()


    def setup(self):
        super(PSRbinary, self).setup()# How to set up heres

    def set_default_values(self):
        #self.PEPOCH.value = 54000.0*u.day      # MJDs (Period epoch)
        #self.P0 = 1.0*u.second           # Sec
        #self.P1 = 0.0                    # Sec/Sec
        paramsDict = {'PB':10.0, 'PBDOT':0.0, 'PBDOT' : 0.0, 'E': 0.0,
                      'EDOT':0.0, 'A1':10.0,'A1DOT':0.0,
                      'OM':10.0,'OMDOT':0.0,'XPBDOT':0.0}
        MJDparamsDict = {'T0':'54000.0',}
        #TODO add alias part
        for key in paramsDict.keys():
            par = getattr(self,key)
            par.set(paramsDict[key],with_unit = True)

        for key in MJDparamsDict.keys():
            MJDpar = getattr(self,key)
            MJDpar.set(MJDparamsDict[key])

    def set_inter_vals(self,barycentricTOA):
        self.barycentricTime = barycentricTOA
        setattr(self,'tt0', self.get_tt0())
        setattr(self,'sinE',np.sin(self.E()))
        setattr(self,'cosE',np.cos(self.E()))
        setattr(self,'sinNu',np.sin(self.nu()))
        setattr(self,'cosNu',np.cos(self.nu()))
        setattr(self,'sinOmg',np.sin(self.omega()))
        setattr(self,'cosOmg',np.sin(self.omega()))

    def der(self,varName,parName):
        if parName not in self.binary_params:
            errorMesg = parName + "is not in binary parameter list."
            raise ValueError(errorMesg)

        if varName not in self.inter_vars+self.pars():
            errorMesg = varName + "is not in variables list."
            raise ValueError(errorMesg)
        # Derivitive to itself
        if varName == parName:
            return np.longdouble(np.ones(len(self.t)))*u.Unit('')
        # Get the unit right
        var = getattr(self,varName)
        par = getattr(self,parName)

        if hasattr(var, '__call__'):
            varU = var().unit
        else:
            varU = var.unit

        if hasattr(par,'unit'):
            parU = par.unit
        else:
            parU = par().unit
        varU = 1*varU
        parU = 1*parU
        if varU in [u.rad, u.deg]:
            varU = varU.to('', equivalencies=u.dimensionless_angles())

        if parU in [u.rad, u.deg]:
            parU = parU.to('', equivalencies=u.dimensionless_angles())
        # Call derivtive functions
        derU =  ((1*varU/(1*parU)).decompose()).unit
        dername = 'd_'+varName+'_d_'+parName

        if hasattr(self,dername):
            return getattr(self,dername)().to(derU, \
                                         equivalencies=u.dimensionless_angles())
        else:
            return (np.longdouble(np.zeros(len(self.t)))*derU).decompose()

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
        if hasattr(eccentricity,unit):
            e = np.longdouble(eccentricity).value
        else:
            e = eccentricity
        if hasattr(mean_anomaly,unit):
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
    	if not hasattr(barycentricTOA,'unit'):
            barycentricTOA = barycentricTOA*u.day
        self.tt0 = (barycentricTOA - T0).to('second')
        return self.tt0

    ####################################
    @Cache.use_cache
    def ecc(self):
        ECC = self.E.value
        EDOT = self.EDOT.value
        return ECC + (self.tt0*EDOT).decompose()

    @Cache.use_cache
    def d_ecc_d_T0(self):
        result = np.empty(len(self.tt0))
        result.fill(-self.EDOT.value.value)
        return result*u.Unit(self.EDOT.units)

    @Cache.use_cache
    def d_ecc_d_ECC(self):
        return np.longdouble(np.ones(len(self.tt0)))

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

        return 2*np.pi*u.rad*((PBDOT+XPBDOT)*self.tt0**2/PB**3 - Sself.tt0/PB**2)

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
        return self.d_nu_d_ecc()*(-self.EDOT)+self.d_nu_d_E()*self.d_E_d_T0()

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
        if par not in self.pars():
            errorMesg = par + "is not in parameter list."
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
                return np.longdouble(np.zeros(len(self.t)))

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
        PB = self.PB.value
        nu = self.nu()

        return PB/2*np.pi*u.rad*nu


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
