from .binary_generic import PSR_BINARY
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
        self.ELL1_interVars = ['eps1', 'eps2', 'Phi', 'Dre', 'Drep', 'Drepp', 'nhat']
        self.add_inter_vars(self.ELL1_interVars)
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

    @Cache.cache_result
    def a1(self):
        """ELL1 model a1 calculation. This method overrides the a1() method in
        pulsar_binary.py. Instead of tt0, it uses ttasc.
        """
        return self.A1 + self.ttasc()*self.A1DOT

    @Cache.cache_result
    def d_a1_d_A1(self):
        return np.longdouble(np.ones(len(self.ttasc())))*u.Unit('')

    @Cache.cache_result
    def d_a1_d_T0(self):
        result = np.empty(len(self.ttasc()))
        result.fill(-self.A1DOT.value)
        return result*u.Unit(self.A1DOT.unit)

    @Cache.cache_result
    def d_a1_d_A1DOT(self):
        return self.ttasc()

    def eps1(self):
        return self.EPS1 + self.ttasc() * self.EPS1DOT

    def d_eps1_d_EPS1(self):
        return np.longdouble(np.ones(len(self.t))) * u.Unit('')

    def d_eps1_d_TASC(self):
        result = np.empty(len(self.t))
        result.fill(-self.EPS1DOT.value)
        return result*u.Unit(self.EPS1DOT.unit)

    def d_eps1_d_EPS1DOT(self):
        return self.ttasc()

    def eps2(self):
        return self.EPS2 + self.ttasc() * self.EPS2DOT

    def d_eps2_d_EPS2(self):
        return np.longdouble(np.ones(len(self.t))) * u.Unit('')

    def d_eps2_d_TASC(self):
        result = np.empty(len(self.t))
        result.fill(-self.EPS2DOT.value)
        return result*u.Unit(self.EPS2DOT.unit)

    def d_eps2_d_EPS2DOT(self):
        return self.ttasc()

    # TODO Duplicate code. Do we need change here.
    def Phi(self):
        """Orbit phase in ELL1 model. Using TASC
        """
        PB = (self.PB).to('second')
        PBDOT = self.PBDOT
        ttasc = self.ttasc()
        orbits = (ttasc/PB - 0.5*PBDOT*(ttasc/PB)**2).decompose()
        norbits = np.array(np.floor(orbits), dtype=np.long)
        phase = (orbits - norbits)*2*np.pi*u.rad
        return phase

    @Cache.cache_result
    def d_Phi_d_TASC(self):
        """dPhi/dTASC
        """
        PB = self.PB.to('second')
        PBDOT = self.PBDOT
        ttasc = self.ttasc()
        return (PBDOT*ttasc/PB-1.0)*2*np.pi*u.rad/PB

    @Cache.cache_result
    def d_Phi_d_PB(self):
        """dPhi/dPB
        """
        PB = self.PB.to('second')
        PBDOT = self.PBDOT
        ttasc = self.ttasc()
        return 2*np.pi*u.rad*(PBDOT*ttasc**2/PB**3 - ttasc/PB**2)

    @Cache.cache_result
    def d_Phi_d_PBDOT(self):
        """dPhi/dPBDOT
        """
        PB = self.PB.to('second')
        ttasc = self.ttasc()
        return -np.pi*u.rad * ttasc**2/PB**2

    def d_Dre_d_par(self, par):
        """Dre = delayR = a1/c.c*(sin(phi) - 0.5* eps1*cos(2*phi) +  0.5* eps2*sin(2*phi))
        d_Dre_d_par = d_a1_d_par /c.c*(sin(phi) - 0.5* eps1*cos(2*phi) +  0.5* eps2*sin(2*phi)) +
                      d_Dre_d_Phi * d_Phi_d_par + d_Dre_d_eps1*d_eps1_d_par + d_Dre_d_eps2*d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der('a1', par)
        d_Dre_d_Phi = self.Drep()
        d_Phi_d_par = self.prtl_der('Phi', par)
        d_Dre_d_eps1 = a1/c.c*(-0.5 * np.cos(2 * Phi))
        d_Dre_d_eps2 = a1/c.c*(0.5 * np.sin(2*Phi))

        return d_a1_d_par /c.c*(np.sin(Phi) - 0.5* eps1*np.cos(2*Phi) + 0.5* eps2*np.sin(2*Phi)) + \
               d_Dre_d_Phi * d_Phi_d_par + d_Dre_d_eps1 * self.prtl_der('eps1', par) + \
               d_Dre_d_eps2 * self.prtl_der('eps2', par)

    def Drep(self):
        """ dDr/dPhi
        """
        a1 = self.a1()
        eps1 = self.eps1()
        eps2 = self.eps1()
        Phi = self.Phi()
        return a1/c.c*(np.cos(Phi) + eps1 * np.sin(Phi) + eps2 * np.cos(Phi))

    def d_Drep_d_par(self, par):
        """Drep = d_Dre_d_Phi = a1/c.c*(cos(Phi) + eps1 * sin(Phi) + eps2 * cos(Phi))
        d_Drep_d_par = d_a1_d_par /c.c*(cos(Phi) + eps1 * sin(Phi) + eps2 * cos(Phi)) +
                      d_Drep_d_Phi * d_Phi_d_par + d_Drep_d_eps1*d_eps1_d_par +
                      d_Drep_d_eps2*d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der('a1', par)
        d_Drep_d_Phi = self.Drepp()
        d_Phi_d_par = self.prtl_der('Phi', par)
        d_Drep_d_eps1 = a1/c.c*np.sin(Phi)
        d_Drep_d_eps2 = a1/c.c*np.cos(Phi)

        return d_a1_d_par /c.c*(np.cos(Phi) + eps1 * np.sin(Phi) + eps2 * np.cos(Phi)) + \
               d_Drep_d_Phi * d_Phi_d_par + d_Drep_d_eps1 * self.prtl_der('eps1', par) + \
               d_Drep_d_eps2 * self.prtl_der('eps2', par)

    def Drepp(self):
        a1 = self.a1()
        eps1 = self.eps1()
        eps2 = self.eps1()
        Phi = self.Phi()
        return a1/c.c*(-np.sin(Phi) + eps1 * np.cos(Phi) - eps2 * np.sin(Phi))

    def d_Drepp_d_par(self, par):
        """Drepp = d_Drep_d_Phi = a1/c.c*(-sin(Phi) + eps1 * cos(Phi) - eps2 * sin(Phi))
        d_Drep_d_par = d_a1_d_par /c.c*(-sin(Phi) + eps1 * cos(Phi) - eps2 * sin(Phi)) +
                      d_Drepp_d_Phi * d_Phi_d_par + d_Drepp_d_eps1*d_eps1_d_par +
                      d_Drepp_d_eps2*d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der('a1', par)
        d_Drepp_d_Phi = a1/c.c*(-np.cos(Phi) - eps1*np.sin(Phi) - eps2 * np.cos(Phi))
        d_Phi_d_par = self.prtl_der('Phi', par)
        d_Drepp_d_eps1 = a1/c.c*np.cos(Phi)
        d_Drepp_d_eps2 = -a1/c.c*np.sin(Phi)

        return d_a1_d_par /c.c*(-np.sin(Phi) + eps1 * np.cos(Phi) - eps2 * np.sin(Phi)) + \
               d_Drepp_d_Phi * d_Phi_d_par + d_Drepp_d_eps1 * self.prtl_der('eps1', par) + \
               d_Drepp_d_eps2 * self.prtl_der('eps2', par)

    def delayR(self):
        """ELL1 Roemer delay in proper time. Ch. Lange,1 F. Camilo, 2001 eq. A6
        """
        Phi = self.Phi()
        return (self.a1()/c.c*(np.sin(Phi) + 0.5 * (self.eps2() * np.sin(2*Phi)
                          - self.eps1() * np.cos(2*Phi)))).decompose()

    def delayS(self):
        """ELL1 Shaprio delay. Ch. Lange,1 F. Camilo, 2001 eq. A16
        """
        TM2 = self.TM2()
        Phi = self.Phi()
        sDelay = -2 * TM2 * np.log(1 - self.SINI * np.sin(Phi))
        return sDelay

    def d_delayS_d_par(self, par):
        """Derivative for bianry Shaprio delay.
        delayS = -2 * TM2 * np.log(1 - self.SINI * np.sin(Phi))
        d_delayS_d_par = d_delayS_d_TM2 * d_TM2_d_par + d_delayS_d_SINI*d_SINI_d_par +
                         d_delayS_d_Phi * d_Phi_d_par
        """
        TM2 = self.TM2()
        Phi = self.Phi()
        d_delayS_d_TM2 = -2*np.log(1 - self.SINI * np.sin(Phi))
        d_delayS_d_SINI = -2 * TM2 * 1.0/(1 - self.SINI * np.sin(Phi))*(-np.sin(Phi))
        d_delayS_d_Phi = -2 * TM2 * 1.0/(1 - self.SINI * np.sin(Phi))*(-self.SINI)
        d_TM2_d_par = self.prtl_der('TM2', par)
        d_SINI_d_par = self.prtl_der('SINI', par)
        d_Phi_d_par = self.prtl_der('Phi', par)

        return d_delayS_d_TM2 * d_TM2_d_par + d_delayS_d_SINI*d_SINI_d_par + \
               d_delayS_d_Phi * d_Phi_d_par


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

    def nhat(self):
        return 2*np.pi/self.PB

    def d_nhat_d_PB(self):
        return -2*np.pi/self.PB**2

    def d_delayI_d_par(self, par):
        """delayI = Dre*(1 - nhat*Drep + (nhat*Drep)**2 + 1.0/2*nhat**2*Dre*Drepp)
        d_delayI_d_par = d_delayI_d_Dre * d_Dre_d_par + d_delayI_d_Drep * d_Drep_d_par +
                         d_delayI_d_Drepp * d_Drepp_d_par + d_delayI_d_nhat * d_nhat_d_par
        """
        Dre = self.delayR()
        Drep = self.Drep()
        Drepp = self.Drepp()
        PB = self.PB.to('second')
        nhat = 2*np.pi/self.PB

        d_delayI_d_Dre = (1 - nhat*Drep + (nhat*Drep)**2 + 1.0/2*nhat**2*Dre*Drepp) + \
                         Dre * 1.0/2*nhat**2*Drepp
        d_delayI_d_Drep = -Dre*nhat + 2*(nhat*Drep)*nhat*Dre
        d_delayI_d_Drepp = 1.0/2*(nhat*Dre)**2
        d_delayI_d_nhat = Dre*(-Drep + 2*(nhat*Drep)*Drep + nhat*Dre*Drepp)
        d_nhat_d_par = self.prtl_der('nhat', par)
        d_Dre_d_par = self.d_Dre_d_par(par)
        d_Drep_d_par = self.d_Drep_d_par(par)
        d_Drepp_d_par = self.d_Drepp_d_par(par)

        return d_delayI_d_Dre * d_Dre_d_par + d_delayI_d_Drep * d_Drep_d_par + \
               d_delayI_d_Drepp * d_Drepp_d_par + d_delayI_d_nhat * d_nhat_d_par

    def ELL1delay(self):
        # TODO need add aberration delay
        return self.delayI() + self.delayS()

    def d_ELL1delay_d_par(self, par):
        return self.d_delayI_d_par(par) + self.d_delayS_d_par(par)

    def ELL1_om(self):
        # arctan(om)
        om = np.arctan2(self.eps1(), self.eps2())
        return om.to(u.deg, equivalencies=u.dimensionless_angles())

    def ELL1_ecc(self):
        return np.sqrt(self.eps1()**2 + self.eps2()**2)

    def ELL1_T0(self):
        return self.TASC + self.PB/(2*np.pi) * \
        (np.arctan(self.eps1()/self.eps2())).to(u.Unit(''), equivalencies=u.dimensionless_angles())
