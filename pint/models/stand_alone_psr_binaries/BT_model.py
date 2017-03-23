from .binary_generic import PSR_BINARY
from pint.models.timing_model import Cache
import numpy as np
import astropy.units as u
import astropy.constants as c
from pint import ls,GMsun,Tsun
"""
This version of the BT model is under construction. Overview of progress:

[v] = Done, [x] = With errors, [ ] = Not done

Calculations
============
[v] Pulse period (Pobs)
[v] Pulse delay (delay)
[v] Derivatives of Pobs (d_Pobs_d_xxx)
[v] Derivatives of delay (d_delay_d_xxx)
[ ] Wrapper function for Derivatives
Interface
=========
[v] Caching (with decorator)
[v] Astropy units
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
[ ] Tempo2 BTmodel automatically sets EDOT to zero

"""

class BTmodel(PSR_BINARY):
    """This is a class independent from PINT platform for pulsar BT binary model.
    It is a subclass of PSR_BINARY class defined in file binary_generic.py in
    the same dierectory. This class is desined for PINT platform but can be used
    as an independent module for binary delay calculation.
    To interact with PINT platform, a pulsar_binary wrapper is needed.
    See the source file pint/models/binary_bt.py
    Refence
    ---------
    The 'BT' binary model for the pulse period. Model as in:
    W.M. Smart, (1962), "Spherical Astronomy", p35
    Blandford & Teukolsky (1976), ApJ, 205, 580-591

    Return
    ----------
    A bt binary model class with paramters, delay calculations and derivatives.
    Example
    ----------
    >>> import numpy
    >>> t = numpy.linspace(54200.0,55000.0,800)
    >>> binary_model = BTmodel()
    >>> paramters_dict = {'A0':0.5,'ECC':0.01}
    >>> binary_model.update_input(t, paramters_dict)
    Here the binary has time input and parameters input, the delay can be
    calculated.

    @param PB:          Binary period [days]
    @param ECC:         Eccentricity
    @param A1:          Projected semi-major axis (lt-sec)
    @param A1DOT:       Time-derivative of A1 (lt-sec/sec)
    @param T0:          Time of ascending node (TASC)
    @param OM:          Omega (longitude of periastron) [deg]
    @param EDOT:        Time-derivative of ECC [0.0]
    @param PBDOT:       Time-derivative of PB [0.0]
    @param OMDOT:       Time-derivative of OMEGA [0.0]
    """

    def __init__(self, t=None, input_params=None):
        super(BTmodel, self).__init__()
        self.binary_name = 'BT'
        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values() # Set parameters to default values.
        self.binary_delay_funcs += [self.BTdelay]
        self.d_binarydelay_d_par_funcs += [self.d_BTdelay_d_par]
        if t is not None:
            self.t = t
        if input_params is not None:
            self.update_input(param_dict=input_params)

    @Cache.use_cache
    def delayL1(self):
        """First term of Blandford & Teukolsky (1976), ApJ, 205,
        580-591, eq 2.33/ First left-hand term of W.M. Smart, (1962),
        "Spherical Astronomy", p35 delay equation.

        alpha * (cosE-e)
        alpha = a1*sin(omega)
        Here a1 is in the unit of light second as distance
        """
        return self.a1()/c.c*np.sin(self.omega())*(np.cos(self.E())-self.ecc())

    @Cache.use_cache
    def delayL2(self):
        """Second term of Blandford & Teukolsky (1976), ApJ, 205,
        580-591, eq 2.33/ / Second left-hand term of W.M. Smart, (1962),
        "Spherical Astronomy", p35 delay equation.

        (beta + gamma) * sinE
        beta = (1-e^2)*a1*cos(omega)
        Here a1 is in the unit of light second as distance
        """
        a1 = self.a1()/c.c
        return (a1*np.cos(self.omega())*\
                np.sqrt(1-self.ecc()**2)+self.GAMMA)*np.sin(self.E())

    @Cache.use_cache
    def delayR(self):
        """Third term of Blandford & Teukolsky (1976), ApJ, 205,
        580-591, eq 2.33 / Right-hand term of W.M. Smart, (1962),
        "Spherical Astronomy", p35 delay equation.

        (alpha*(cosE-e)+(beta+gamma)*sinE)*(1-alpha*sinE - beta*sinE)/
        (pb*(1-e*coeE))
        (alpha*(cosE-e)+(beta+gamma)*sinE) is define in delayL1
        and delayL2
        """
        a1 = self.a1()/c.c
        num = a1*np.cos(self.omega())*np.sqrt(1-self.ecc()**2)*np.cos(self.E()) -\
              a1*np.sin(self.omega())*np.sin(self.E())
        den = 1.0 - self.ecc()*np.cos(self.E())

        # In BTmodel.C, they do not use pbprime here, just pb...
        # Is it not more appropriate to include the effects of PBDOT?
        #return 1.0 - 2*np.pi*num / (den * self.pbprime())
        return 1.0 - 2*np.pi*num / (den * self.PB.to(u.second))

    @Cache.use_cache
    def BTdelay(self):
        """Full BT model delay"""
        return (self.delayL1() + self.delayL2()) * self.delayR()



    # NOTE: Below, OMEGA is supposed to be in RADIANS!
    # TODO: Fix UNITS!!!
    @Cache.use_cache
    def d_delayL1_d_E(self):
        a1 = self.a1()/c.c
        return -a1*np.sin(self.omega())*np.sin(self.E())

    @Cache.use_cache
    def d_delayL2_d_E(self):
        a1 = self.a1()/c.c
        return (a1*np.cos(self.omega())*np.sqrt(1-self.ecc()**2)+self.GAMMA) * np.cos(self.E())

    @Cache.use_cache
    def d_delayL1_d_A1(self):
        return np.sin(self.omega())*(np.cos(self.E()) - self.ecc())/c.c

    @Cache.use_cache
    def d_delayL1_d_A1DOT(self):
        return self.tt0 * self.d_delayL1_d_A1()

    @Cache.use_cache
    def d_delayL2_d_A1(self):
        return np.cos(self.omega())*np.sqrt(1-self.ecc()**2)*np.sin(self.E())/c.c

    @Cache.use_cache
    def d_delayL2_d_A1DOT(self):
        return self.tt0 * self.d_delayL2_d_A1()

    @Cache.use_cache
    def d_delayL1_d_OM(self):
        a1 = self.a1()/c.c
        return a1*np.cos(self.omega())*(np.cos(self.E())-self.ecc())

    @Cache.use_cache
    def d_delayL1_d_OMDOT(self):
        return self.tt0 * self.d_delayL1_d_OM()

    @Cache.use_cache
    def d_delayL2_d_OM(self):
        a1 = self.a1()/c.c
        return -a1*np.sin(self.omega())*np.sqrt(1-self.ecc()**2)*np.sin(self.E())

    @Cache.use_cache
    def d_delayL2_d_OMDOT(self):
        return self.tt0 * self.d_delayL2_d_OM()

    @Cache.use_cache
    def d_delayL1_d_ECC(self):
        a1 = self.a1()/c.c
        return a1*np.sin(self.omega()) + \
               self.d_delayL1_d_E() * self.d_E_d_ECC()

    @Cache.use_cache
    def d_delayL1_d_EDOT(self):
        return self.tt0 * self.d_delayL1_d_ECC()

    @Cache.use_cache
    def d_delayL2_d_ECC(self):
        a1 = self.a1()/c.c
        num = -a1*np.cos(self.omega())*self.ecc()*np.sin(self.E())
        den = np.sqrt(1-self.ecc()**2)
        return num/den + self.d_delayL2_d_E() * self.d_E_d_ECC()

    @Cache.use_cache
    def d_delayL2_d_EDOT(self):
        return self.tt0 * self.d_delayL2_d_ECC()

    @Cache.use_cache
    def d_delayL1_d_GAMMA(self):
        return np.zeros(len(self.t)) * u.second/u.second

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
    def d_delay_d_GAMMA(self):
        return self.delayR() * (self.d_delayL1_d_GAMMA() + self.d_delayL2_d_GAMMA())

    def d_BTdelay_d_par(self, par):
        if hasattr(self, 'd_delay_d_'+par):
            return getattr(self, 'd_delay_d_'+par)()
        else:
            return np.zeros(len(self.tt0))*u.second/getattr(self, par).unit
