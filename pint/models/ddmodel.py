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
                       'OM', 'OMDOT', 'A1', 'A1DOT','A0','B0', 'T0', 'GAMMA',\
                       'SINI']

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
        self.ECC =0.9*u.Unit('')         # -
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
        #setattr(self,'Ae',np.arctan2(sinAe,cosAe))
        setattr(self,'Ae',self.nu())
        # Obtain T. Damour and N. Deruelle equation er and etheta

        #setattr(self,'er',self.ecct+self.Dr)  #???
        #setattr(self,'eTheta',self.ecct+self.Dtheta)  #???
        setattr(self,'A1',self.a1())
        # # Obtain T. Damour and N. Deruelle equation [29]
        setattr(self,'Omega',self.omega())
        setattr(self,'sOmg',np.sin(self.Omega))
        setattr(self,'cOmg',np.cos(self.Omega))

        # FIXME more robust
        varL = ['TM2','tt0','ecc','M','E','nu','er','eTheta','a1','omega',
                   'P','Pobs','alpha','beta','Dre','Drep','Drepp','nhat',]
        setattr(self,'varList',varL)

    def der(self,varName,parName):
        if parName not in self.pars():
            errorMesg = parName + "is not in parameter list."
            raise ValueError(errorMesg)

        if varName not in self.varList+self.pars():
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
    def get_tt0(self):
    	if not hasattr(self.t,'unit'):
            self.t = self.t*u.day
        if not hasattr(self.T0,'unit'):
            self.T0 = self.T0*u.day
        return (self.t-self.T0).to('second')
    ####################################
    @Cache.use_cache
    def ecc(self):
        return self.ECC + self.tt0*self.EDOT
    @Cache.use_cache
    def d_ecc_d_T0(self):
        return -self.EDOT
    @Cache.use_cache
    def d_ecc_d_ECC(self):
        return np.longdouble(np.ones(len(self.t)))*u.Unit('')
    @Cache.use_cache
    def d_ecc_d_EDOT(self):
        return self.get_tt0()
    ####################################
    @Cache.use_cache
    def a1(self):
        return self.A1 + self.tt0*self.A1DOT

    ############################################################
    @Cache.use_cache
    def omega(self):
        """T. Damour and N. Deruelle(1986)equation [25]
           omega = OM+Ae*k
           k = OMDOT/n  (T. Damour and N. Deruelle(1986)equation between Eq 16
                         Eq 17)
        """
        k = self.OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/self.PB)
        return (self.OM + self.Ae*k).to(u.rad)
    @Cache.use_cache
    def d_omega_d_par(self,par):
        """Derivitive for omega respect to user input Parameter.
           if par is not 'OM','OMDOT','PB'
           dOmega/dPar =  k*dAe/dPar
           k = OMDOT/n
           Parameter
           ----------
           par : string
                 parameter name
        """
        if par not in self.pars():
            errorMesg = par + "is not in parameter list."
            raise ValueError(errorMesg)

        k = self.OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/self.PB)
        if par in ['OM','OMDOT','PB']:
            dername = 'd_omeg_d_' + par
            return getattr(self,dername)()
        else:
            dername = 'd_nu_d_'+par # For parameters only in Ae
            if hasattr(self,dername):
                return k*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.t)))

    @Cache.use_cache
    def d_omega_d_OM(self):
        """dOmega/dOM = 1
        """
        return np.longdouble(np.ones((len(self.t))))*u.Unit('')

    @Cache.use_cache
    def d_omega_d_OMDOT(self):
        """dOmega/dOMDOT = 1/n*Ae
           n = 2*pi/PB
           dOmega/dOMDOT = PB/2*pi*Ae
        """
        return self.PB/2*np.pi*u.rad*self.Ae


    @Cache.use_cache
    def d_omega_d_PB(self):
        """dOmega/dPB = dAe/dPB*k+dk/dPB*Ae
        """
        k = self.OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad/self.PB)

        return (self.d_nu_d_PB()*k)+  \
               self.Ae*self.OMDOT.to(u.rad/u.second)/(2*np.pi*u.rad)


    ############################################################

    @Cache.use_cache
    def pbprime(self):
        return self.PB - self.PBDOT * self.tt0

    @Cache.use_cache
    def P(self):
        return self.P0 + self.P1*(self.t - self.PEPOCH).to('second')

    ############################################################


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
        phase = (orbits - norbits)*2*np.pi*u.rad

        return phase
    @Cache.use_cache
    def d_M_d_T0(self):
        """dM/dT0  this could be a generic function
        """
        return ((self.PBDOT-self.XPBDOT)*self.tt0/self.PB-1.0)*2*np.pi*u.rad/self.PB

    @Cache.use_cache
    def d_M_d_PB(self):
        """dM/dPB this could be a generic function
        """
        return 2*np.pi*u.rad*((self.PBDOT+self.XPBDOT)*self.tt0**2/self.PB**3 - \
                        self.tt0/self.PB**2)

    @Cache.use_cache
    def d_M_d_PBDOT(self):
        """dM/dPBDOT this could be a generic function
        """

        return -np.pi*u.rad * self.tt0**2/self.PB**2

    @Cache.use_cache
    def d_M_d_XPBDOT(self):
        """dM/dPBDOT this could be a generic function
        """
        return -np.pi*u.rad * self.tt0**2/self.PB**2
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
        RHS = self.der('M','T0')
        return (RHS-self.EDOT*np.sin(self.E()))/(1.0-np.cos(self.E())*self.ecc())

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


    #########################################################
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


    ########################################################
    @Cache.use_cache
    def Doppler(self):
        return 2*np.pi*self.a1() / (self.pbprime()*np.sqrt(1-self.ecc()**2))

    ########################################################
    # Derivity for delays in DD model
    @Cache.use_cache
    def er(self):
        return self.ecc()+self.Dr
    @Cache.use_cache
    def d_er_d_Dr(self):
        return np.longdouble(np.ones(len(self.t)))
    @Cache.use_cache
    def d_er_d_par(self,par):
        if par not in self.pars():
            errorMesg = par + "is not in parameter list."
            raise ValueError(errorMesg)

        if par in ['Dr']:
            dername = 'd_er_d_'+par
            return getattr(self,dername)()
        else:
            dername = 'd_ecc_d_'+par
            if hasattr(self,dername):
                return getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.t),1))

    ##########
    @Cache.use_cache
    def eTheta(self):
        return self.ecc()+self.Dtheta
    @Cache.use_cache
    def d_eTheta_d_Dtheta(self):
        return np.longdouble(np.ones(len(self.t)))
    @Cache.use_cache
    def d_eTheta_d_par(self,par):
        if par not in self.pars():
            errorMesg = par + "is not in parameter list."
            raise ValueError(errorMesg)

        if par in ['Dtheta']:
            dername = 'd_eTheta_d_'+par
            return getattr(self,dername)()
        else:
            dername = 'd_ecc_d_'+par
            if hasattr(self,dername):
                return getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.t),1))
    ##########
    @Cache.use_cache
    def alpha(self):
        """Alpha defined in
           T. Damour and N. Deruelle(1986)equation [46]
           alpha = A1/c*sin(omega)
        """
        return self.A1/c.c*self.sOmg
    @Cache.use_cache
    def d_alpha_d_par(self,par):
        """T. Damour and N. Deruelle(1986)equation [46]
           alpha = A1/c*sin(omega)

           dAlpha/dA1 = 1.0/c*sin(omega)

           dAlpha/dPar = A1/c*cos(omega)*dOmega/dPar
        """

        if par not in self.pars():
            errorMesg = par + "is not in parameter list."
            raise ValueError(errorMesg)

        if par in ['A1']:
            return 1.0/c.c*self.sOmg

        else:
            dername = 'd_omega_d_'+par # For parameters only in Ae
            if hasattr(self,dername):
                return self.A1/c.c*self.cOmg*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.t)))

    ##############################################
    @Cache.use_cache
    def beta(self):
        """Beta defined in
           T. Damour and N. Deruelle(1986)equation [47]
           beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        return self.A1/c.c*(1-eTheta**2)**0.5*self.cOmg
    @Cache.use_cache
    def d_beta_d_par(self,par):
        """beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
           eTheta = ecc+Dtheta  ??
           dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
           dBeta/dECC = A1/c*((-(e+dr)/sqrt((e+dr)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           dBeta/dEDOT = A1/c*((-(e+dr)/sqrt((e+dr)**2)*cos(omega)*de/dEDOT-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           dBeta/dDth = A1/c*(-(e+dr)/sqrt((e+dr)**2)*cos(omega)
           Other parameters
           dBeta/dPar = -A1/c*(1-eTheta**2)**0.5*sin(omega)*dOmega/dPar
        """
        if par not in self.pars():
            errorMesg = par + "is not in parameter list."
            raise ValueError(errorMesg)

        if par in ['A1','ECC','EDOT','Dth']:
            dername = 'd_beta_d_'+par
            return getattr(self,dername)()

        else:
            dername = 'd_omega_d_'+par # For parameters only in Ae
            if hasattr(self,dername):
                eTheta = self.eTheta()
                a1 = self.a1()
                return -a1/c.c*(1-eTheta**2)**0.5*self.sOmg*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.t)))
    @Cache.use_cache
    def d_beta_d_A1(self):
        """dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        return 1.0/c.c*(1-eTheta**2)**0.5*self.cOmg
    @Cache.use_cache
    def d_beta_d_ECC(self):
        """dBeta/dECC = A1/c*((-(e+dtheta)/sqrt((e+dtheta)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           de/dECC = 1
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()

        return a1/c.c*((-eTheta)/np.sqrt(eTheta**2)*self.cOmg- \
               (1-eTheta**2)**0.5*self.sOmg*self.d_omega_d_par('ECC'))

    @Cache.use_cache
    def d_beta_d_EDOT(self):
        """dBeta/dEDOT = A1/c*((-(e+dtheta)/sqrt((e+dtheta)**2)*cos(omega)*de/dEDOT- \
           (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           de/dEDOT = tt0
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()
        return a1/c.c*((-eTheta)/np.sqrt(eTheta**2)*self.cOmg*self.tt0- \
               (1-eTheta**2)**0.5*self.sOmg*self.d_omega_d_par('EDOT'))
    @Cache.use_cache
    def d_beta_d_Dth(self):
        """dBeta/dDth = A1/c*((-(e+dr)/sqrt((e+dr)**2)*cos(omega)
        """
        eTheta = self.eTheta()
        return self.A1/c.c*(-eTheta)/np.sqrt(eTheta**2)*self.cOmg



    ##################################################
    @Cache.use_cache
    def Dre(self):
        """Dre defined in
           T. Damour and N. Deruelle(1986)equation [48]
           Dre = alpha*(cos(E)-er)+(beta+gamma)*sin(E)
        """
        er = self.er()
        return self.alpha()*(self.cosEcc_A- er)+   \
               (self.beta()+self.GAMMA)*self.sinEcc_A
    @Cache.use_cache
    def d_Dre_d_par(self,par):
        """Dre = alpha*(cos(E)-er)+(beta+gamma)*sin(E)
           dDre = alpha*(-der-dE*sin(E)) + (cos[E]-er)*dalpha +
                  (dBeta+dGamma)*sin(E) + (beta+gamma)*cos(E)*dE

           dDre/dpar = alpha*(-der/dpar-dE/dpar*sin(E)) +
                       (cos[E]-er)*dalpha/dpar +
                       (dBeta/dpar+dGamma/dpar)*sin(E) +
                       (beta+gamma)*cos(E)*dE/dpar

            er = e + Dr
        """
        # First term
        term1 = self.alpha()*(-self.der('er',par)-self.der('E',par)*self.sinEcc_A)
        # Second term
        term2 = (self.cosEcc_A-self.er())*self.der('alpha',par)
        # Third term
        term3 = (self.der('beta',par)+self.der('GAMMA',par))*self.sinEcc_A
        # Fourth term
        term4 = (self.beta()+self.GAMMA)*self.cosEcc_A*self.der('E',par)

        return term1 + term2 + term3 +term4
    #################################################
    @Cache.use_cache
    def Drep(self):
        """Dervitive of Dre respect to E
           T. Damour and N. Deruelle(1986)equation [49]
           Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
        """
        return -self.alpha()*self.sinEcc_A+(self.beta()+self.GAMMA)*self.cosEcc_A
    @Cache.use_cache
    def d_Drep_d_par(self,par):
        """Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
           dDrep = -alpha*cos(E)*dE + cos(E)*(dbeta+dgamma)
                   -(beta+gamma)*dE*sin(E)-dalpha*sin(E)
           dDrep/dPar = -sin(E)*dalpha/dPar
                        -(alpha*cos(E)+(beta+gamma)*sin(E))*dE/dPar
                        + cos(E)(dbeta/dPar+dgamma/dPar)

        """
        # first term
        term1 = -self.sinEcc_A*self.der('alpha',par)
        # second term
        term2 = -(self.alpha()*self.cosEcc_A+  \
                (self.beta()+self.GAMMA)*self.sinEcc_A)*self.der('E',par)
        # Third term
        term3 = self.cosEcc_A*(self.der('beta',par)+self.der('GAMMA',par))

        return term1+term2+term3

    #################################################
    @Cache.use_cache
    def Drepp(self):
        """Dervitive of Drep respect to E
           T. Damour and N. Deruelle(1986)equation [50]
           Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
        """
        return -self.alpha()*self.cosEcc_A-(self.beta()+self.GAMMA)*self.sinEcc_A
    @Cache.use_cache
    def d_Drepp_d_par(self,par):
        """Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
           dDrepp = -(beta+gamma)*cos(E)*dE - cos(E)*dalpha
                    +alpha*sin(E)*dE - (dbeta+dgamma)*sin(E)

           dDrepp/dPar = -cos(E)*dalpha/dPar
                         +(alpha*sin(E)-(beta+gamma)*cos(E))*dE/dPar
                         -(dbeta/dPar+dgamma/dPar)*sin(E)
        """

        # first term
        term1 = -self.cosEcc_A*self.der('alpha',par)
        # second term
        term2 = (self.alpha()*self.sinEcc_A -  \
                (self.beta()+self.GAMMA)*self.cosEcc_A)*self.der('E',par)
        # Third term
        term3 = -self.sinEcc_A*(self.der('beta',par)+self.der('GAMMA',par))

        return term1+term2+term3
    #################################################
    @Cache.use_cache
    def nhat(self):
        """nhat defined as
           T. Damour and N. Deruelle(1986)equation [51]
           nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()
        """
        return 2.0*np.pi/self.PB.to('second')/(1-self.ecc()*self.cosEcc_A)
    @Cache.use_cache
    def d_nhat_d_par(self,par):
        """nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()
           dnhat = -2*pi*dPB/PB^2*(1-ecc*cos(E))
                   -2*pi*(-cos(E)*decc+ecc*sin(E)*dE)/PB*(1-ecc*cos(E))^2

           dnhat/dPar = -2*pi/(PB*(1-ecc*cos(E))*((dPB/dPar)/PB -
                        (-cos(E)*decc/dPar+ecc*sin(E)*dE/dpar)/(1-e*cos(E)))
        """
        oneMeccTcosE = (1-self.ecc()*self.cosEcc_A)
        fctr = -2*np.pi/self.PB/oneMeccTcosE

        return fctr*(self.der('PB',par)/self.PB - \
               (self.cosEcc_A*self.der('ecc',par)+ \
                self.ecc()*self.sinEcc_A*self.der('E',par))/oneMeccTcosE)
    #################################################
    @Cache.use_cache
    def delayInverse(self):
        """DD model Inverse timing delay.
           T. Damour and N. Deruelle(1986)equation [46-52]

           From proper time to coordinate time.
           The Romoer delay and Einstein are included.
        """
        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()

        return (Dre*(1-nHat*Drep+(nHat*Drep)**2+1.0/2*nHat**2*Dre*Drepp-\
                1.0/2*self.ecct*self.sinEcc_A/(1-self.ecct*self.cosEcc_A)*\
                nHat**2*Dre*Drep)).decompose()
    @Cache.use_cache
    def d_delayI_d_par(self,par):
        """ddelayI/dPar = dDre/dPar*delayI/Dre       [1] half
                          +Dre*d(delayI/Dre)/dPar    [2] half
           d(delayI/Dre)/dPar = -nhat*dDrep/dPar-Drep*dnhat/dPar   part (1)
                                +2*Drep*nhat*(nhat*dDrep/dPar+Drep*dnhat/dPar  part (2)
                                +1/2*nhat*(Drepp*nhat*dDre/dPar+Dre*nhat*dDrepp/dPar+2*Dre*Drepp*dnhat/dPar) part (3)
                                +Part4    part (4)
           Define x:= -1.0/2*ecct*sin(E)/(1-ecct*cos(E))
           part4 = nhat*(Dre*Drep*nhat*dx/dPar+x*(Drep*nHat*dDre/dPar+Dre*nHat*dDrep/dPar+
                   2*Dre*Drep*dnhat/dPar))
           dx/dPar = (-ecc*cos(E)*dE/dPar-sin(E)*decc/dPar
                     +ecc*sin(E)*(-cos(E)*decc/dPar+ecc*sin(E)*dE/dPar)/(1-ecc*cos(E)))/(2*(1-ecc*cos(E)))
        """
        e = self.ecc()
        sE = self.sinEcc_A
        cE = self.cosEcc_A
        dE_dpar = self.der('E',par)
        decc_dpar = self.der('ecc',par)

        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        delayI = self.delayInverse()

        dDre_dpar = self.d_Dre_d_par(par)
        dDrep_dpar = self.d_Drep_d_par(par)
        dDrepp_dpar = self.d_Drepp_d_par(par)
        dnhat_dpar = self.d_nhat_d_par(par)
        oneMeccTcosE = (1-e*cE) # 1-e*cos(E)
        x =  -1.0/2.0*e*sE/oneMeccTcosE # -1/2*e*sin(E)/(1-e*cos(E))
        dx_dpar = (-e*cE*dE_dpar-sE*decc_dpar+
                   e*sE*(-cE*decc_dpar+e*sE*dE_dpar)/oneMeccTcosE)/(2*oneMeccTcosE)
        #First half
        H1 = dDre_dpar*delayI/Dre
        # For second half
        part1 = -nHat*dDrep_dpar-Drep*dnhat_dpar
        part2 = 2*Drep*nHat*(nHat*dDrep_dpar+Drep*dnhat_dpar)
        part3 = 1.0/2.0*nHat*(Drepp*nHat*dDre_dpar+Dre*nHat*dDrepp_dpar+2*Dre*Drepp*dnhat_dpar)
        part4 = nHat*(Dre*Drep*nHat*dx_dpar+  \
                x*(Drep*nHat*dDre_dpar+Dre*nHat*dDrep_dpar+2*Dre*Drep*dnhat_dpar))

        H2 = Dre*(part1+part2+part3+part4)

        return H1+H2
    @Cache.use_cache
    def d_delayI_d_par2(self,par):
        """
        """
        e = self.ecc()
        sE = self.sinEcc_A
        cE = self.cosEcc_A
        dE_dpar = self.der('E',par)
        decc_dpar = self.der('ecc',par)

        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        delayI = self.delayInverse()

        dDre_dpar = self.d_Dre_d_par(par)
        dDrep_dpar = self.d_Drep_d_par(par)
        dDrepp_dpar = self.d_Drepp_d_par(par)
        dnhat_dpar = self.d_nhat_d_par(par)
        oneMeccTcosE = (1-e*cE) # 1-e*cos(E)
        x =  -1.0/2.0*e*sE/oneMeccTcosE # -1/2*e*sin(E)/(1-e*cos(E))
        dx_dpar = (-e*cE*dE_dpar-sE*decc_dpar+
                   e*sE*(-cE*decc_dpar+e*sE*dE_dpar)/oneMeccTcosE)/(2*oneMeccTcosE)

        diDelay_dDre = 1+(Drep*nHat)**2+Dre*Drepp*nHat**2-Drep*nHat*(1+2*Dre*nHat*x)
        diDelay_dDrep = -Dre*nHat*(1-2*Drep*nHat+Dre*nHat*x)
        diDelay_dDrepp = (Dre*nHat)**2/2
        diDelay_dx = -(Dre*nHat)**2*Drep

        return dDre_dpar*diDelay_dDre + dDrep_dpar*diDelay_dDrep + \
               dDrepp_dpar*diDelay_dDrepp + dx_dpar*diDelay_dx

    #################################################
    @Cache.use_cache
    def delayS(self):
        """Binary shapiro delay
           T. Damour and N. Deruelle(1986)equation [26]
        """
        e = self.ecc()
        cE = self.cosEcc_A
        sE = self.sinEcc_A
        sOmega = self.sOmg
        cOmega = sefl.cOmg


        sDelay = -2*self.TM2 * np.log(1-e*cE-self.SINI*(sOmega*(cE-e)+
                 (1-e**2)**0.5*cOmega*sE))
        return sDelay
    @Cache.use_cache
    def d_delayS_d_par(self,par):
        """dsDelay/dPar = dsDelay/dTM2*dTM2/dPar+
                          dsDelay/decc*decc/dPar+
                          dsDelay/dE*dE/dPar+
                          dsDelay/domega*domega/dPar+
                          dsDelay/dSINI*dSINI/dPar
        """
        e = self.ecc()
        cE = self.cosEcc_A
        sE = self.sinEcc_A
        sOmega = self.sOmg
        cOmega = self.cOmg

        logNum = 1-e*cE-self.SINI*(sOmega*(cE-e)+
                 (1-e**2)**0.5*cOmega*sE)
        dTM2_dpar = self.der('TM2',par)
        dsDelay_dTM2 = -2*np.log(logNum)
        decc_dpar = self.der('ecc',par)
        dsDelay_decc = -2*self.TM2/logNum*(-cE-self.SINI*(-e*cOmega*sE/np.sqrt(1-e**2)-sOmega))
        dE_dpar = self.der('E',par)
        dsDelay_dE =  -2*self.TM2/logNum*(e*sE-self.SINI*(np.sqrt(1-e**2)*cE*cOmega-sE*sOmega))
        domega_dpar = self.der('omega',par)
        dsDelay_domega = -2*self.TM2/logNum*self.SINI*((cE-e)*cOmega-np.sqrt(1-e**2)*sE*sOmega)
        dSINI_dpar = self.der('SINI',par)
        dsDelay_dSINI = -2*self.TM2/logNum*(-np.sqrt(1-e**2)*cOmega*sE-(cE-e)*sOmega)
        return dTM2_dpar*dsDelay_dTM2 + decc_dpar*dsDelay_decc + \
               dE_dpar*dsDelay_dE +domega_dpar*dsDelay_domega +  \
               dSINI_dpar*dsDelay_dSINI
    #################################################
    @Cache.use_cache
    def delayE(self):
        """Binary Einstein delay
            T. Damour and N. Deruelle(1986)equation [25]
        """
        return self.GAMMA*self.sinEcc_A
    @Cache.use_cache
    def d_delayE_d_par(self,par):
        """eDelay = gamma*sin[E]
           deDelay_dPar = deDelay/dgamma*dgamma/dPar +
                          deDelay/dE*dE/dPar
        """
        cE = self.cosEcc_A
        sE = self.sinEcc_A

        return sE*self.der('GAMMA',par)+self.GAMMA*cE*self.der('E',par)
    #################################################
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
    def d_delayA_d_par(self,par):
        """aDelay = A0*(sin(omega+E)+e*sin(omega))+B0*(cos(omega+E)+e*cos(omega))
           daDelay/dpar = daDelay/dA0*dA0/dPar+     (1)
                          daDelay/dB0*dB0/dPar+     (2)
                          daDelay/domega*domega/dPar+    (3)
                          daDelay/dnu*dnu/dPar+        (4)
                          daDelay/decc*decc/dPar        (5)
        """
        e = self.ecc()
        sOmega = self.sOmg
        cOmega = self.cOmg
        snu = np.sin(self.nu())
        cnu = np.cos(self.nu())
        omgPlusAe = self.Omega+self.Ae
        if par =='A0':
            return e*sOmega+np.sin(omgPlusAe)
        elif par == 'B0':
            return e*cOmega+np.cos(omgPlusAe)
        else:
            domega_dpar = self.der('omega',par)
            daDelay_domega = self.A0*(np.cos(omgPlusAe)+e*cOmega)- \
                             self.B0*(np.sin(omgPlusAe)+e*sOmega)

            dnu_dpar = self.der('nu',par)
            daDelay_dnu = self.A0*np.cos(omgPlusAe)-self.B0*np.sin(omgPlusAe)

            decc_dpar = self.der('ecc',par)
            daDelay_decc = self.A0*sOmega+self.B0*cOmega

            return domega_dpar*daDelay_domega + dnu_dpar*daDelay_dnu + decc_dpar*daDelay_decc
    #################################################
    @Cache.use_cache
    def delay(self):
        """Full DD model delay"""
        return self.delayInverse()+self.delayS()+self.delayA()

    @Cache.use_cache
    def delayR(self):  # Is this needed?
        """Binary Romoer delay
            T. Damour and N. Deruelle(1986)equation [24]
        """

        rDelay = self.A1/c.c*(self.sOmg*(self.cosEcc_A-self.er)   \
                 +(1-self.eTheta**2)**0.5*self.cOmg*self.sinEcc_A)
        return rDelay.decompose()

    def d_delay_d_par(self,par):
        """Full DD model delay derivtive
        """
        return self.d_delayI_d_par(par)+self.d_delayS_d_par(par)+ \
               self.d_delayA_d_par(par)
