from .pulsar_binary import PSR_BINARY
from pint.models.timing_model import Cache
import numpy as np
import astropy.units as u
import astropy.constants as c
from pint import ls,GMsun,Tsun


class ELL1model(PSR_BINARY):
    """This is a class for ELL1 pulsar binary model.
       ELL1 model is BT model in the small eccentricity case.
    """
    def __init__(self):
        super(ELL1model, self).__init__()
        self.binary_name = 'ELL1'
        self.param_default_value.update({'EPS1': 0*u.Unit(''),
                                         'EPS2': 0*u.Unit(''),
                                         'EPS1DOT': 0/u.second,
                                         'EPS2DOT': 0/u.second,
                                         'TASC': np.longdouble(54000.0)*u.day})
        self.binary_params = self.param_default_value.keys()
        self.set_param_values() # Set parameters to default values.
        self.binary_delay_funcs += [self.ELL1delay]
        self.d_binarydelay_d_par_funcs += [self.d_ELL1delay_d_par]

    def ttasc(self):
        """
        ttasc = t - TASC
        """
        if not hasattr(self.t,'unit') or self.t.unit == None:
            t = self.t * u.day
        t = self.t
        ttasc = (t - self.TASC).to('second')
        return ttasc

    def eps1(self):
        return self.EPS1 + self.ttasc() * self.EPS1DOT

    def eps2(self):
        return self.EPS2 + self.ttasc() * self.EPS2DOT

    # TODO Duplicate code. Need to be changed here.
    def Phi(self):
        """Orbit phase, this could be a generic function
        """
        PB = (self.PB).to('second')
        PBDOT = self.PBDOT
        ttasc = self.ttasc()
        orbits = (ttasc/PB - 0.5*PBDOT*(ttasc/PB)**2).decompose()
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = (orbits - norbits)*2*np.pi*u.rad
        return phase

    def delayR(self):
        Phi = self.Phi()
        return (self.a1()/c.c*(np.sin(Phi) + 0.5 * (self.eps2() * np.sin(2*Phi)
                          - self.eps1() * np.cos(2*Phi)))).decompose()

    def Drep(self):
        """ dDr/dPhi
        """
        a1 = self.a1()
        eps1 = self.eps1()
        eps2 = self.eps1()
        Phi = self.Phi()
        return a1/c.c*(np.cos(Phi) + eps1 * np.cos(Phi) + eps2 * np.sin(Phi))

    def Drepp(self):
        a1 = self.a1()
        eps1 = self.eps1()
        eps2 = self.eps1()
        Phi = self.Phi()
        return a1/c.c*(-np.sin(Phi) - eps1 * np.sin(Phi) + eps2 * np.cos(Phi))

    def delayS(self):
        TM2 = self.M2.value*Tsun
        Phi = self.Phi()
        sDelay = -2 * TM2 * np.log(1 - self.SINI * np.sin(Phi))
        return sDelay

    def delayI(self):
        """Inverse time delay formular. The treatment is similar to the one
        in DD model(T. Damour and N. Deruelle(1986)equation [46-52])
        Dre = a1*(sin(Phi)+eps1/2*sin(Phi)+eps1/2*cos(Phi))
        Drep = dDre/dt
        Drep = d^2 Dre/dt^2
        nhat = dPhi/dt = 2pi/pb
        nhatp = d^2Phi/dt^2 = 0
        Dre(t-Dre(t-Dre(t)))  = Dre(Phi) - Drep(Phi)*nhat*Dre(t-Dre(t))
                              = Dre(Phi) - Drep(Phi)*nhat*(Dre(Phi)-Drep(Phi)*nhat*Dre(t))
                                + 1/2 (Drepp(u)*nhat^2 + Drep(u) * nhat * nhatp) * (Dre(t)-...)^2
                              = Dre(Phi)(1 - nhat*Drep(Phi) + (nhat*Drep(Phi))^2
                                + 1/2*nhat^2* Dre*Drepp)
        """
        Dre = self.delayR()
        Drep = self.Drep()
        Drepp = self.Drepp()
        PB = self.PB.to('second')
        nhat = 2*np.pi/self.PB
        return (Dre*(1 - nhat*Drep + (nhat*Drep)**2 + 1.0/2*nhat**2*Dre*Drepp)).decompose()

    def ELL1delay(self):
        # TODO need add aberration delay
        return self.delayI() + self.delayS()

    def d_ELL1delay_d_par(self):
        pass

    def ELL1_om(self):
        # arctan(om)
        om = np.arctan2(self.eps1(), self.eps2())
        return om.to(u.deg, equivalencies=u.dimensionless_angles())

    def ELL1_ecc(self):
        return np.sqrt(self.eps1()**2 + self.eps2()**2)

    def ELL1_T0(self):
        return self.TASC + self.PB/(2*np.pi) * \
        (np.arctan(self.eps1()/self.eps2())).to(u.Unit(''), equivalencies=u.dimensionless_angles())
