import numpy as np
import functools
import collections
from astropy import log
from .timing_model import Cache
from scipy.optimize import newton
import astropy.units as u 
import astropy.constants as c
from pint import ls,GMsun,Tsun
class DDmodel(object):
    """The 'DD' binary model. Model as in 
    1. T. Damour and N. Deruelle, General relativistic celestial
    mechanics of binary systems. I. the post-Newtonian timing
    formula, Annales de l' I.H.P.,section A, tome 44,n 1 (1985), p. 107-132
    Parameters
    2. T. Damour and N. Deruelle, General relativistic celestial
    mechanics of binary systems. II. the post-Newtonian timing
    formula, Annales de l' I.H.P.,section A, tome 44,n 3 (1986), p. 263-292
    Parameters:
    @param P0:          The pulse period [sec]
    @param P1:          The pulse period derivative [sec/sec]
    @param PEPOCH:      Position EPOCH
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec) a*sin(i)
    @param A1DOT:       Time-derivative of A1 (lt-sec/sec)
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param SINI:           
    @param A0:
    @param B0:
    @param GAMMA:
    @param Dr:
    @param Dth:
    @param M2:
    @param RA_RAD:      Pulsar position (right ascension) [rad]
    @param DEC_RAD:     Pulsar position (declination) [rad]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param XPBDOT:
    @param OMDOT:       Time-derivative of OMEGA [0.0]
    """
    def __init__(self,t, **kwargs):
        self.binaryName = 'DD'
        if not isinstance(t, np.ndarray) and not isinstance(t,list):
            self.t = np.array([t,])
        else:
            self.t = t
        self.params = ['PEPOCH', 'P0', 'P1', 'PB', 'PBDOT', 'ECC', 'EDOT', \
                       'OM', 'OMDOT', 'A1', 'A1DOT','A0','B0', 'T0', 'GAMMA']

        self.set_default_values()
        for key, value in kwargs.iteritems():
            if key.upper() in self.params:
                setattr(self, key.upper(), value[0]*value[1])
                # How to get units work
        self.compute_inter_vars(self.t)
    
    def pars(self):
        """Parameter names in model"""
        return self.params

    def set_default_values(self):
        self.PEPOCH = 54000.0*u.day      # MJDs (Period epoch)
        self.P0 = 1.0*u.second           # Sec
        self.P1 = 0.0                    # Sec/Sec      
        self.PB = np.longdouble(10.0)*u.day             # Day
        self.PBDOT = 0.0*u.day/u.day     # Day/Day
        self.ECC =0.9*u.Unit(1)         # -
        self.T0 = 54000.0*u.day          # Day
        self.EDOT = 0.0/u.second         # 1/Sec
        self.A1 = 10.0*ls                # Light-Sec  ar*sin(i)
        self.A1DOT = 0.0*ls/u.second     # Light-Sec / Sec
        self.A0 = 0.0*u.second           # Sec
        self.B0 = 0.0*u.second           # Sec
        self.OM=0.0*u.deg                # Deg
        self.OMDOT=0.0*u.deg/u.year      # Deg/year 
        self.M2 = 0.0*u.M_sun            # Mass of companian in the unit Sun mass
        self.XPBDOT = 0.0*u.day/u.day    # Day/Day
        self.Dr = 0.0*u.Unit('')         # - 
        self.Dtheta = 0.0*u.Unit('')     # - 
        self.SINI = 0.0 *u.Unit('')      # -
        self.GAMMA = 0.0*u.second        # Sec

    def compute_eccentric_anomaly(self, eccentricity, mean_anomaly):
        """compute eccentric anomaly solve for Kepler Equation
        Parameter
        eccentric_anomaly(mean_anomaly):
        Return the eccentric anomaly in radians, given a set of mean_anomalies
        in radians.
        """
        e = np.longdouble(eccentricity).value
        ma = np.longdouble(mean_anomaly).value
        k = lambda E: E-e*np.sin(E)-ma   # Kepler Equation
        dk = lambda E: 1-e*np.cos(E)     # Derivitive Kepler Equation 
        U = ma
        while(np.max(abs(k(U)))>5e-16):  # Newton-Raphson method 
            U = U-k(U)/dk(U)
        return U*u.rad    

    @Cache.use_cache
    def compute_inter_vars(self,t):
    	"""Set up inter-step variables 
    	   @ tt0
           @ ecct
           @ eccentric_anomaly
    	   @ cos(eccentric_anomaly)
    	   @ sin(eccentric_anomaly)
    	   @ Ae(eccentric_anomaly)
           @ er
           @ eTheta
           @ ar
           @ omega
    	"""
        setattr(self,'TM2',self.M2.value*Tsun)
    	setattr(self,'tt0', self.get_tt0())
    	setattr(self,'ecct',self.ecc())
        setattr(self,'phase',self.M())
        # Solve Kepler equation
        setattr(self,'ecc_anom', self.E())

        # Obtain T. Damour and N. Deruelle equation 17a-17c
        setattr(self,'cosEcc_A',np.cos(self.ecc_anom))
        setattr(self,'sinEcc_A',np.sin(self.ecc_anom))
        cosAe = (self.cosEcc_A - self.ecct)/(1-self.ecct*self.cosEcc_A)
        sinAe = (1-self.ecct**2)**(1.0/2)*self.sinEcc_A/(1-self.ecct*self.cosEcc_A)
        setattr(self,'Ae',np.arctan2(sinAe,cosAe))
        # Obtain T. Damour and N. Deruelle equation er and etheta

        setattr(self,'er',self.ecct+self.Dr)  #???
        setattr(self,'eTheta',self.ecct+self.Dtheta)  #???
        setattr(self,'A1',self.a1())  
        # # Obtain T. Damour and N. Deruelle equation [29]
        setattr(self,'Omega',self.omega())
        setattr(self,'sOmg',np.sin(self.Omega))
        setattr(self,'cOmg',np.cos(self.Omega))
    @Cache.use_cache
    def delayR(self):
        """Binary Romoer delay
            T. Damour and N. Deruelle(1986)equation [24]
        """
    	
        rDelay = self.A1/c.c*(self.sOmg*(self.cosEcc_A-self.er)   \
                 +(1-self.eTheta**2)**0.5*self.cOmg*self.sinEcc_A)
        return rDelay.decompose()
    @Cache.use_cache
    def delayS(self):
        """Binary shapiro delay
           T. Damour and N. Deruelle(1986)equation [26]
        """

        sDelay = -2*self.TM2 * np.log(1-self.ecct*self.cosEcc_A-
        	                           self.SINI*(self.sOmg*(self.cosEcc_A
        	                           -self.ecct)+(1-self.ecct**2)**0.5
        	                           *self.cOmg*self.sinEcc_A))
        return sDelay 
    @Cache.use_cache
    def delayE(self):
        """Binary Einstein delay
            T. Damour and N. Deruelle(1986)equation [25]
        """
        return self.GAMMA*self.sinEcc_A

    @Cache.use_cache
    def delayA(self):
        """Binary Abberation delay
            T. Damour and N. Deruelle(1986)equation [27]
        """
        omgPlusAe = self.Omega+self.Ae 
        aDelay = self.A0*(np.sin(omgPlusAe)+self.ecct*self.sOmg)+\
                 self.B0*(np.cos(omgPlusAe)+self.ecct*self.cOmg) 
        return aDelay

    @Cache.use_cache
    def delay(self):
        """Full DD model delay"""
        print self.delayR()[0],self.delayS()[0],self.delayE()[0],self.delayA()[0]
        return self.delayR()+self.delayS()+self.delayE()+self.delayA()

    def delayInverse(self):
        alpha = self.A1/c.c*self.sOmg
        beta = self.A1/c.c*(1-self.eTheta**2)**0.5*self.cOmg
        Dre = alpha*(self.cosEcc_A-self.er)+(beta+self.GAMMA)*self.sinEcc_A
        Drep = -alpha*self.sinEcc_A+(beta+self.GAMMA)*self.cosEcc_A
        Drepp = -alpha*self.cosEcc_A-(beta+self.GAMMA)*self.sinEcc_A
        nHat = 2.0*np.pi/self.PB.to('second')/(1-self.ecct*self.cosEcc_A)

        return (Dre*(1-nHat*Drep+(nHat*Drep)**2+1.0/2*nHat**2*Dre*Drepp-\
                1.0/2*self.ecct*self.sinEcc_A/(1-self.ecct*self.cosEcc_A)*\
                nHat**2*Dre*Drep)).decompose()


    @Cache.use_cache
    def get_tt0(self):
    	if not hasattr(self.t,'unit'):
            self.t = self.t*u.day
        if not hasattr(self.T0,'unit'):
            self.T0 = self.T0*u.day 
        return (self.t-self.T0).to('second')
    @Cache.use_cache
    def ecc(self):
        return self.ECC + self.tt0*self.EDOT

    @Cache.use_cache
    def a1(self):
        return self.A1 + self.tt0*self.A1DOT
    @Cache.use_cache
    def omega(self):
        """T. Damour and N. Deruelle(1986)equation [25]
           k = OMDOT/n  (T. Damour and N. Deruelle(1986)equation between Eq 16 
                         Eq 17)
        """
        k = self.OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/self.PB)
        return (self.OM + self.Ae*k).to(u.rad)

    @Cache.use_cache
    def pbprime(self):
        return self.PB - self.PBDOT * self.tt0

    @Cache.use_cache
    def Pobs(self):
        if not np.all(np.logical_and(self.ecc() >= 0.0, self.ecc() <= 1.0)):
            return np.inf

        # Projected velocity of the pulsar in the direction of the line-of-sight
        # (Divided by speed of light due to units of A1)
        vl = 2*np.pi*self.a1()/(self.pbprime()*np.sqrt(1-self.ecc()**2))

        return self.P() * (1.0 + vl*((np.cos(self.nu())+self.ecc())*np.cos(self.om())-np.sin(self.nu())*np.sin(self.om())))
    ############################################################
    @Cache.use_cache
    def M(self):
        """Orbit phase, this could be a generic function
        """
        orbits = (self.tt0/self.PB.to('second') -  \
               0.5*(self.PBDOT+self.XPBDOT)*(self.tt0/self.PB)**2).decompose()
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = (orbits - norbits)*2*np.pi

        return phase
    @Cache.use_cache    
    def d_M_d_T0(self):
        """dM/dT0  this could be a generic function
        """
        return ((self.PBDOT-self.XPBDOT)*self.tt0/self.PB-1.0)*2*np.pi/self.PB
    
    @Cache.use_cache
    def d_M_d_PB(self):
        """dM/dPB this could be a generic function
        """
        return 2*np.pi*((self.PBDOT+self.XPBDOT)*self.tt0**2/self.PB**3 - \
                        self.tt0/self.PB**2)

    @Cache.use_cache
    def d_M_d_PBDOT(self):
        """dM/dPBDOT this could be a generic function
        """

        return -np.pi * self.tt0**2/self.PB**2
    
    @Cache.use_cache
    def d_M_d_XPBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        return -np.pi * self.tt0**2/self.PB**2
    ##########################################################
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
        RHS = self.d_M_d_T0()
        
        return (RHS-self.EDOT*np.sin(self.E())/(1.0-np.cos(self.E())*self.ecc())
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
        return np.sin(self.E()) / (1.0 - self.ecc()*np.cos(self.E()))
    @Cache.use_cache
    def d_E_d_EDOT(self):
        return self.tt0 * self.d_E_d_ECC()
    

    #########################################################
    @Cache.use_cache
    def nu(self):
        """True anomaly"""
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
    def d_nu_d_ECC2(self):
        """dnu(e,E)/dECC = dnu/de*de/dECC+dnu/dE*dE/dECC
           de/dECC = 1
           dnu/dPBDOT = dnu/dE*dE/dECC+dnu/de
        """
        return self.d_nu_d_ecc()+self.d_nu_d_E()*self.d_E_d_ECC
    
    @Cache.use_cache
    def d_nu_d_EDOT(self):
        return self.tt0() * self.d_nu_d_ECC()


    ########################################################
    @Cache.use_cache
    def Doppler(self):
        return 2*np.pi*self.a1() / (self.pbprime()*np.sqrt(1-self.ecc()**2))


    def d_delayR_d_A1(self):
        return 1.0/c.c*(sOmg*(self.cosEcc_A-self.er)   \
                 +(1-self.eTheta**2)**0.5*cOmg*self.sinEcc_A)

    def d_delayR_d_ECC(self):
        # Since d_r d_th is not clear, This does not code yet
        pass
    
    def d_delayR_d_OM(self):
        pass
    
    def d_delayR_d_OMDOT(self):
        pass
    
    def d_delayR_d_Dr(self):
        pass
    def d_delayR_d_Dth(self):
        pass
    # For Shaprio Delay
    def d_delayS_d_M2(self):
        pass 
    # For Einstein delay
    def d_delayE_d_GAMMA(self):
        return self.sinEcc_A

   