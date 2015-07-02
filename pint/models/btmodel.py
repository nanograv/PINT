import numpy as np
import functools
import collections
from astropy import log
from .timing_model import Cache

"""
This version of the BT model is under construction. Overview of progress:

[v] = Done, [x] = With errors, [ ] = Not done

Calculations
============
[v] Pulse period (Pobs)
[v] Pulse delay (delay)
[v] Derivatives of Pobs (d_Pobs_d_xxx)
[x] Derivatives of delay (d_delay_d_xxx)

Interface
=========
[v] Caching (with decorator)
[ ] Astropy units
[v] Setting & getting parameters

Code quality
============
[ ] Docstrings
[ ] Unit tests (wrt tempo2 or internally?)
[x] Formatting (pylint)

Open issues
===========
[x] In delayR(), I would think we need to use self.pbprime().
    However, tempo2 seems consistent with self.pb()
    -- RvH: July 2, 2015
[ ] We are ignoring the derivatives of delayR() at the moment. This is a decent
    approximation for non-relativistic orbital velocities (1 part in ~10^6)

"""

class BTmodel(object):
    """
    The 'BT' binary model for the pulse period. Model as in:
    W.M. Smart, (1962), "Spherical Astronomy", p359

    See also: Blandford & Teukolsky (1976), ApJ, 205, 580-591

    @param P0:          The pulse period [sec]
    @param P1:          The pulse period derivative [sec/sec]
    @param PEPOCH:      Position EPOCH
    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec)
    @param A1DOT:       Time-derivative of A1 (lt-sec/sec)
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param RA_RAD:      Pulsar position (right ascension) [rad]
    @param DEC_RAD:     Pulsar position (declination) [rad]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param OMDOT:       Time-derivative of OMEGA [0.0]
    """

    def __init__(self, t, **kwargs):
        self.t = t              # MJDs
        self.params = ['PEPOCH', 'P0', 'P1', 'PB', 'PBDOT', 'ECC', 'EDOT', \
                       'OM', 'OMDOT', 'A1', 'A1DOT', 'T0', 'GAMMA']
        self.set_default_values()
        
        for key, value in kwargs.iteritems():
            if key.upper() in self.params:
                setattr(self, key.upper(), value)

    def set_default_values(self):
        """Set the default values to something reasonable"""
        # TODO: Start using astropy units!
        self.C = 299792458.0        # m/s
        self.SECS_PER_DAY = 86400.0
        self.SECS_PER_YEAR = self.SECS_PER_DAY * 365.25
        self.DEG2RAD = np.pi/180.0

        self.P0 = 1.0           # Sec
        self.P1 = 0.0           # Sec/Sec
        self.PB = 10.0          # Days
        self.PBDOT = 0.0        # Sec/Sec
        self.ECC = 0.0          # -
        self.EDOT = 0.0         # /Sec
        self.OM = 10.0          # Degrees
        self.OMDOT = 0.0        # Degrees/yr
        self.A1 = 10.0          # Light-Sec
        self.A1DOT = 0.0        # Light-Sec / Sec
        self.T0 = 54000.0       # MJDs (TASC / binary epoch)
        self.GAMMA = 0.0        # Time dilation & gravitational redshift
        self.PEPOCH = 54000.0   # MJDs (Period epoch)

    def pars(self):
        """Parameter names in model"""
        return self.params

    def vals(self, values=None, params=None):
        """Parameter values, setting & getting"""
        if params is None:
            params = self.params
        if values is None:
            return np.array([getattr(self, par) for par in params])
        elif isinstance(values, collections.Mapping):
            for par in values:
                setattr(self, par, values[par])
        elif isinstance(values, collections.Iterable):
            for par,val in zip(params, values):
                setattr(self, par, val)

    
    @Cache.use_cache
    def eccentric_anomaly(self, eccentricity, mean_anomaly):
        """
        eccentric_anomaly(mean_anomaly):
        Return the eccentric anomaly in radians, given a set of mean_anomalies
        in radians.
        """
        ma = np.fmod(mean_anomaly, 2*np.pi)
        ma = np.where(ma < 0.0, ma+2*np.pi, ma)
        ecc_anom_old = ma
        ecc_anom = ma + eccentricity*np.sin(ecc_anom_old)
        # This is a simple iteration to solve Kepler's Equation
        while (np.maximum.reduce(np.fabs(ecc_anom-ecc_anom_old)) > 5e-15):
            ecc_anom_old = ecc_anom[:]
            ecc_anom = ma + eccentricity * np.sin(ecc_anom_old)
        return ecc_anom
    
    @Cache.use_cache
    def P(self):
        return self.P0 + self.P1*(self.t - self.PEPOCH)*self.SECS_PER_DAY
    
    @Cache.use_cache
    def om(self):
        return (self.OM + self.OMDOT*self.tt0()/self.SECS_PER_YEAR) * self.DEG2RAD
    
    @Cache.use_cache
    def ecc(self):
        return self.ECC + self.tt0()*self.EDOT
    
    @Cache.use_cache
    def a1(self):
        return self.A1 + self.tt0()*self.A1DOT

    @Cache.use_cache
    def Pobs(self):
        if not np.all(np.logical_and(self.ecc() >= 0.0, self.ecc() <= 1.0)):
            return np.inf

        # Projected velocity of the pulsar in the direction of the line-of-sight
        # (Divided by speed of light due to units of A1)
        vl = 2*np.pi*self.a1()/(self.pbprime()*np.sqrt(1-self.ecc()**2))

        return self.P() * (1.0 + vl*((np.cos(self.nu())+self.ecc())*np.cos(self.om())-np.sin(self.nu())*np.sin(self.om())))

    @Cache.use_cache
    def delayL1(self):
        """First left-hand term of delay equation"""
        return self.a1()*np.sin(self.om())*(np.cos(self.E())-self.ecc())

    @Cache.use_cache
    def delayL2(self):
        """Second left-hand term of delay equation"""
        return (self.a1()*np.cos(self.om())*\
                np.sqrt(1-self.ecc()**2)+self.GAMMA)*np.sin(self.E())

    @Cache.use_cache
    def delayR(self):
        """Right-hand term of delay equation"""
        num = self.a1()*np.cos(self.om())*np.sqrt(1-self.ecc()**2)*np.cos(self.E()) -\
              self.a1()*np.sin(self.om())*np.sin(self.E())
        den = 1.0 - self.ecc()*np.cos(self.E())

        # In BTmodel.C, they do not use pbprime here, just pb...
        # Is it not more appropriate to include the effects of PBDOT?
        #return 1.0 - 2*np.pi*num / (den * self.pbprime())
        return 1.0 - 2*np.pi*num / (den * self.pb())

    @Cache.use_cache
    def delay(self):
        """Full BT model delay"""
        return (self.delayL1() + self.delayL2()) * self.delayR()
    
    @Cache.use_cache
    def d_Pobs_d_par(self, parname, pct=0.001):
        par = getattr(self, parname)
        x1 = par
        x2 = (1+pct)*par
        dx = x2-x1
        
        y1 = self.Pobs()
        setattr(self, parname, x2)
        y2 = self.Pobs()
        dy = y2-y1

        setattr(self, parname, x1)
        return dy / dx

    @Cache.use_cache
    def tt0(self):
        return (self.t-self.T0) * self.SECS_PER_DAY
    
    @Cache.use_cache
    def t0(self):
        return (self.t-self.PEPOCH) * self.SECS_PER_DAY
    
    @Cache.use_cache
    def pb(self):
        return self.PB * self.SECS_PER_DAY
    
    @Cache.use_cache
    def pbprime(self):
        return self.pb() - self.PBDOT * self.tt0()
        
    @Cache.use_cache
    def M(self):
        return np.fmod(self.tt0()/self.pb() - 0.5*self.PBDOT*(self.tt0()/self.pb())**2, 1.0) * 2*np.pi

    @Cache.use_cache
    def d_M_d_T0(self):
        return self.SECS_PER_DAY*(self.PBDOT*(self.tt0()/self.pb())-1.0)*2*np.pi/self.pb()
    
    @Cache.use_cache
    def d_M_d_PB(self):
        return self.SECS_PER_DAY*2*np.pi*(self.PBDOT*self.tt0()**2/self.pb()**3 - self.tt0()/self.pb()**2)
    
    @Cache.use_cache
    def d_M_d_PBDOT(self):
        return -np.pi * self.tt0()**2/self.pb()**2
    
    @Cache.use_cache
    def E(self):
        return self.eccentric_anomaly(self.ecc(), self.M())
    
    @Cache.use_cache
    def d_E_d_T0(self):
        brack1 = self.PBDOT*self.tt0()/self.pb() - 1.0
        brack2 = 2*np.pi/(self.pb()*(1-self.ecc()*np.cos(self.E())))
        return self.SECS_PER_DAY * brack1 * brack2
    
    @Cache.use_cache
    def d_E_d_PB(self):
        brack1 = self.PBDOT*self.tt0()**2/self.pb()**3 - self.tt0()/self.pb()**2
        brack2 = np.pi * 2 / (1-self.ecc()*np.cos(self.E()))
        return self.SECS_PER_DAY * brack1 * brack2

    @Cache.use_cache
    def d_E_d_PBDOT(self):
        return -np.pi*self.tt0()**2/(self.pb()**2 * (1-self.ecc()*np.cos(self.E())))
    
    @Cache.use_cache
    def d_E_d_ECC(self):
        return np.sin(self.E()) / (1 - self.ecc()*np.cos(self.E()))
    
    @Cache.use_cache
    def d_E_d_EDOT(self):
        return self.tt0() * self.d_E_d_ECC()
    
    @Cache.use_cache
    def nu(self):
        """True anomaly"""
        return 2*np.arctan(np.sqrt((1.0+self.ecc())/(1.0-self.ecc()))*np.tan(self.E()/2.0))
    
    @Cache.use_cache
    def d_nu_d_T0(self):
        brack1 = (1 + self.ecc()*np.cos(self.nu())) / (1 - self.ecc()*np.cos(self.E()))
        brack2 = self.d_E_d_T0() * np.sin(self.E()) / np.sin(self.nu())
        return brack1*brack2
    
    @Cache.use_cache
    def d_nu_d_PB(self):
        brack1 = (1 + self.ecc()*np.cos(self.nu())) / (1 - self.ecc()*np.cos(self.E()))
        brack2 = self.d_E_d_PB() * np.sin(self.E()) / np.sin(self.nu())
        return brack1*brack2
    
    @Cache.use_cache
    def d_nu_d_PBDOT(self):
        brack1 = (1 + self.ecc()*np.cos(self.nu())) / (1 - self.ecc()*np.cos(self.E()))
        brack2 = self.d_E_d_PBDOT() * np.sin(self.E()) / np.sin(self.nu())
        return brack1*brack2

    @Cache.use_cache
    def d_nu_d_ECC(self):
        brack1 = self.d_E_d_ECC() * np.sin(self.E()) * (1 + self.ecc()*np.cos(self.nu())) - np.cos(self.E())*np.cos(self.nu()) + 1.0
        brack2 = np.sin(self.nu()) * (1 - self.ecc() * np.cos(self.E()))
        return brack1 / brack2
    
    @Cache.use_cache
    def d_nu_d_EDOT(self):
        return self.tt0() * self.d_nu_d_ECC()

    @Cache.use_cache
    def Doppler(self):
        return 2*np.pi*self.a1() / (self.pbprime()*np.sqrt(1-self.ecc()**2))
    
    @Cache.use_cache
    def d_Pobs_d_P0(self):
        geom = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        ds = self.Doppler() * geom
        return 1.0 + ds

    @Cache.use_cache
    def d_Pobs_d_P1(self):
        geom = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        ds = self.Doppler() * geom
        return self.t0() * (1 + ds)
    
    @Cache.use_cache
    def d_Pobs_d_A1(self):
        geom = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        return 2*np.pi * self.P() * geom / (self.pbprime() * np.sqrt(1-self.ecc()**2))
    
    @Cache.use_cache
    def d_Pobs_d_PB(self):
        geom1 = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        geom2 = (-np.cos(self.nu())*np.sin(self.om())-np.sin(self.nu())*np.cos(self.om()))
        pref1 = -self.P() * 2 * np.pi * self.a1() / (self.pbprime()**2 * np.sqrt(1-self.ecc()**2)) * self.SECS_PER_DAY
        pref2 = self.P() * self.Doppler() * self.d_nu_d_PB()
        return pref1 * geom1 + pref2 * geom2
    
    @Cache.use_cache
    def d_Pobs_d_PBDOT(self):
        geom1 = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        geom2 = (-np.cos(self.nu())*np.sin(self.om())-np.sin(self.nu())*np.cos(self.om()))
        pref1 = self.P() * self.tt0() * 2 * np.pi * self.a1() / (self.pbprime()**2 * np.sqrt(1-self.ecc()**2))
        pref2 = self.P() * self.Doppler() * self.d_nu_d_PBDOT()
        return pref1 * geom1 + pref2 * geom2
    
    @Cache.use_cache
    def d_Pobs_d_OM(self):
        geom = (-np.sin(self.nu())*np.cos(self.om())-(np.cos(self.nu())+self.ecc())*np.sin(self.om()))
        return self.P() * self.Doppler() * geom * self.DEG2RAD

    @Cache.use_cache
    def d_Pobs_d_ECC(self):
        geom1 = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        geom2 = (-np.cos(self.nu())*np.sin(self.om())-np.sin(self.nu())*np.cos(self.om()))
        pref1 = self.P() * self.ecc() * 2 * np.pi * self.a1() / (self.pbprime() * (1-self.ecc()**2)**(1.5))
        pref2 = self.P() * self.Doppler() * self.d_nu_d_ECC()
        return pref1 * geom1 + pref2 * geom2 + self.P0 * self.Doppler() * np.cos(self.om())

    @Cache.use_cache
    def d_Pobs_d_T0(self):
        geom1 = (-np.sin(self.nu())*np.sin(self.om())+(np.cos(self.nu())+self.ecc())*np.cos(self.om()))
        geom2 = (-np.cos(self.nu())*np.sin(self.om())-np.sin(self.nu())*np.cos(self.om()))
        pref1 = -self.P() * self.PBDOT * 2 * np.pi * self.a1() / (self.pbprime()**2 * np.sqrt(1-self.ecc()**2)) * self.SECS_PER_DAY
        pref2 = self.P() * self.Doppler() * self.d_nu_d_T0()
        return pref1 * geom1 + pref2 * geom2

    @Cache.use_cache
    def d_Pobs_d_EDOT(self):
        return self.tt0() * self.d_Pobs_d_ECC()
    
    @Cache.use_cache
    def d_Pobs_d_OMDOT(self):
        return self.tt0() * self.d_Pobs_d_OM() / self.SECS_PER_YEAR

    @Cache.use_cache
    def d_Pobs_d_A1DOT(self):
        return self.tt0() * self.d_Pobs_d_A1()

    # NOTE: Below, OMEGA is supposed to be in RADIANS!
    # TODO: Fix UNITS!!!
    @Cache.use_cache
    def d_delayL1_d_E(self):
        return -self.a1()*np.sin(self.om())*np.sin(self.E())

    @Cache.use_cache
    def d_delayL2_d_E(self):
        return (self.a1()*np.cos(self.om())*np.sqrt(1-self.ecc()**2)+self.GAMMA)

    @Cache.use_cache
    def d_delayL1_d_A1(self):
        return np.sin(self.om())*(np.cos(self.E()) - self.ecc())

    @Cache.use_cache
    def d_delayL1_d_A1DOT(self):
        return self.tt0() * self.d_delayL1_d_A1()

    @Cache.use_cache
    def d_delayL2_d_A1(self):
        return np.cos(self.om())*np.sqrt(1-self.ecc()**2)*np.sin(self.E())

    @Cache.use_cache
    def d_delayL2_d_A1DOT(self):
        return self.tt0() * self.d_delayL2_d_A1()

    @Cache.use_cache
    def d_delayL1_d_OM(self):
        return self.a1()*np.cos(self.om())*(np.cos(self.E())-self.ecc())*self.DEG2RAD

    @Cache.use_cache
    def d_delayL1_d_OMDOT(self):
        return self.tt0() * self.d_delayL1_d_OM() / self.SECS_PER_DAY

    @Cache.use_cache
    def d_delayL2_d_OM(self):
        return self.a1()*np.sin(self.om())*np.sqrt(1-self.ecc()**2)*np.sin(self.E())*self.DEG2RAD

    @Cache.use_cache
    def d_delayL2_d_OMDOT(self):
        return self.tt0() * self.d_delayL2_d_OM() / self.SECS_PER_DAY

    @Cache.use_cache
    def d_delayL1_d_ECC(self):
        return self.a1()*np.sin(self.om()) + \
                self.d_delayL1_d_E() * self.d_E_d_ECC()

    @Cache.use_cache
    def d_delayL1_d_EDOT(self):
        return self.tt0() * self.d_delayL1_d_ECC()

    @Cache.use_cache
    def d_delayL2_d_ECC(self):
        num = -self.a1()*np.cos(self.om())*self.ecc()*np.sin(self.E()) 
        den = np.sqrt(1-self.ecc()**2)
        return num/den + self.d_delayL2_d_E() * self.d_E_d_ECC()

    @Cache.use_cache
    def d_delayL2_d_EDOT(self):
        return self.tt0() * self.d_delayL2_d_ECC()

    @Cache.use_cache
    def d_delayL1_d_GAMMA(self):
        return np.zeros_like(self.t)

    @Cache.use_cache
    def d_delayL2_d_GAMMA(self):
        return np.sin(self.E())

    @Cache.use_cache
    def d_delayL1_d_T0(self):
        return self.d_delayL1_d_E() * self.d_E_d_T0()

    @Cache.use_cache
    def d_delayL2_d_T0(self):
        return self.d_delayL2_d_E() * self.d_E_d_T0()

    @Cache.use_cache
    def d_delayL1_d_PB(self):
        return self.d_delayL1_d_E() * self.d_E_d_PB()

    @Cache.use_cache
    def d_delayL2_d_PB(self):
        return self.d_delayL2_d_E() * self.d_E_d_PB()

    @Cache.use_cache
    def d_delayL1_d_PBDOT(self):
        return self.d_delayL1_d_E() * self.d_E_d_PBDOT()

    @Cache.use_cache
    def d_delayL2_d_PBDOT(self):
        return self.d_delayL2_d_E() * self.d_E_d_PBDOT()

    @Cache.use_cache
    def d_delay_d_A1(self):
        return self.delayR() * (self.d_delayL1_d_A1() + self.d_delayL2_d_A1())

    @Cache.use_cache
    def d_delay_d_A1DOT(self):
        return self.delayR() * (self.d_delayL1_d_A1DOT() + \
                self.d_delayL2_d_A1DOT())

    @Cache.use_cache
    def d_delay_d_OM(self):
        return self.delayR() * (self.d_delayL1_d_OM() + self.d_delayL2_d_OM())

    @Cache.use_cache
    def d_delay_d_OMDOT(self):
        return self.delayR() * (self.d_delayL1_d_OMDOT() + \
                self.d_delayL2_d_OMDOT())

    @Cache.use_cache
    def d_delay_d_ECC(self):
        return self.delayR() * (self.d_delayL1_d_ECC() + self.d_delayL2_d_ECC())

    @Cache.use_cache
    def d_delay_d_EDOT(self):
        return self.delayR() * (self.d_delayL1_d_EDOT() + \
                self.d_delayL2_d_EDOT())

    @Cache.use_cache
    def d_delay_d_PB(self):
        return self.delayR() * (self.d_delayL1_d_PB() + self.d_delayL2_d_PB())

    @Cache.use_cache
    def d_delay_d_PBDOT(self):
        return self.delayR() * (self.d_delayL1_d_PBDOT() + \
                self.d_delayL2_d_PBDOT())

    @Cache.use_cache
    def d_delay_d_T0(self):
        return self.delayR() * (self.d_delayL1_d_T0() + self.d_delayL2_d_T0())

    @Cache.use_cache
    def Pobs_designmatrix(self, params):
        npars = len(params)
        M = np.zeros((len(self.t), npars))

        for ii, par in enumerate(params):
            dername = 'd_Pobs_d_' + par
            M[:,ii] = getattr(self, dername)()

        return M

    @Cache.use_cache
    def delay_designmatrix(self, params):
        npars = len(params)
        M = np.zeros((len(self.t), npars))

        for ii, par in enumerate(params):
            dername = 'd_delay_d_' + par
            M[:,ii] = getattr(self, dername)()

        return M
