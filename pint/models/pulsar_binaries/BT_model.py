from .pulsar_binary import PSR_BINARY
from pint.models.timing_model import Cache
import numpy as np
import astropy.units as u
import astropy.constants as c
from pint import ls,GMsun,Tsun, time_distance


class BTmodel(PSR_BINARY):
    """This is a class for BT pulsar binary model.
    """
    def __init__(self,t):
        super(BTmodel, self).__init__()
        self.binary_name = 'BT'
        if not isinstance(t, np.ndarray) and not isinstance(t,list):
            self.t = np.array([t,])
        else:
            self.t = t
        # If any parameter has aliases, it should be updated
        #self.param_aliases.update({})
        self.binary_params = self.param_default_value.keys()
        self.set_param_values()
        self.binary_delay_funcs += [self.BTdelay]
        #self.d_binarydelay_d_par_funcs += [self.d_BTdelay_d_par]

    @Cache.use_cache
    def delayL1(self):
        """First left-hand term of delay equation"""
        return self.a1()/c.c*np.sin(self.omega())*(np.cos(self.E())-self.ecc())

    @Cache.use_cache
    def delayL2(self):
        """Second left-hand term of delay equation"""
        return (self.a1()/c.c*np.cos(self.omega())*\
                np.sqrt(1-self.ecc()**2)+self.GAMMA)*np.sin(self.E())

    @Cache.use_cache
    def delayR(self):
        """Right-hand term of delay equation"""
        num = self.a1()/c.c*np.cos(self.omega())*np.sqrt(1-self.ecc()**2)*np.cos(self.E()) -\
              self.a1()/c.c*np.sin(self.omega())*np.sin(self.E())
        den = 1.0 - self.ecc()*np.cos(self.E())

        # In BTmodel.C, they do not use pbprime here, just pb...
        # Is it not more appropriate to include the effects of PBDOT?
        #return 1.0 - 2*np.pi*num / (den * self.pbprime())
        return 1.0 - 2*np.pi*num / (den * self.PB)

    @Cache.use_cache
    def BTdelay(self):
        """Full BT model delay"""
        return ((self.delayL1() + self.delayL2()) * self.delayR()).decompose()
############### Derivatives.
    @Cache.use_cache
    def d_delayL1_d_E(self):
        return -self.a1()/c.c*np.sin(self.omega())*np.sin(self.E())

    @Cache.use_cache
    def d_delayL2_d_E(self):
        return (self.a1()/c.c*np.cos(self.omega())*np.sqrt(1-self.ecc()**2)+self.GAMMA) * np.cos(self.E())

    @Cache.use_cache
    def d_delayL1_d_A1(self):
        return np.sin(self.omega())*(np.cos(self.E()) - self.ecc())

    @Cache.use_cache
    def d_delayL1_d_A1DOT(self):
        return (self.tt0 * self.d_delayL1_d_A1())

    @Cache.use_cache
    def d_delayL2_d_A1(self):
        return np.cos(self.omega())*np.sqrt(1-self.ecc()**2)*np.sin(self.E())

    @Cache.use_cache
    def d_delayL2_d_A1DOT(self):
        return (self.tt0 * self.d_delayL2_d_A1())

    @Cache.use_cache
    def d_delayL1_d_OM(self):
        par = getattr(self, 'OM')
        dunit = u.s/par.unit
        return (self.a1()/c.c*np.cos(self.omega())*(np.cos(self.E())
                -self.ecc())).to(dunit, equivalencies=u.dimensionless_angles())

    @Cache.use_cache
    def d_delayL1_d_OMDOT(self):
        par = getattr(self, 'OMDOT')
        dunit = u.s/par.unit
        return (self.tt0 * self.d_delayL1_d_OM()).to(dunit, equivalencies=u.dimensionless_angles())

    @Cache.use_cache
    def d_delayL2_d_OM(self):
        par = getattr(self, 'OM')
        dunit = u.s/par.unit
        return -(self.a1()/c.c*np.sin(self.omega())*np.sqrt(1-self.ecc()**2)*np.sin(self.E())).to(dunit, equivalencies=u.dimensionless_angles())

    @Cache.use_cache
    def d_delayL2_d_OMDOT(self):
        par = getattr(self, 'OMDOT')
        dunit = u.s/par.unit
        return (self.tt0 * self.d_delayL2_d_OM()).to(dunit, equivalencies=u.dimensionless_angles())

    @Cache.use_cache
    def d_delayL1_d_ECC(self):
        return -self.a1()/c.c*np.sin(self.omega()) + \
                self.d_delayL1_d_E() * self.d_E_d_ECC()

    @Cache.use_cache
    def d_delayL1_d_EDOT(self):
        return self.tt0 * self.d_delayL1_d_ECC()

    @Cache.use_cache
    def d_delayL2_d_ECC(self):
        num = -self.a1()/c.c*np.cos(self.omega())*self.ecc()*np.sin(self.E())
        den = np.sqrt(1-self.ecc()**2)
        return num/den + self.d_delayL2_d_E() * self.d_E_d_ECC()

    @Cache.use_cache
    def d_delayL2_d_EDOT(self):
        return self.tt0 * self.d_delayL2_d_ECC()

    @Cache.use_cache
    def d_delayL1_d_GAMMA(self):
        return np.zeros_like(self.t) * u.Unit('')

    @Cache.use_cache
    def d_delayL2_d_GAMMA(self):
        return np.sin(self.E())

    @Cache.use_cache
    def d_delayL1_d_T0(self):
        return (self.d_delayL1_d_E() * self.d_E_d_T0()).to(u.s/u.day)

    @Cache.use_cache
    def d_delayL2_d_T0(self):
        return (self.d_delayL2_d_E() * self.d_E_d_T0()).to(u.s/u.day)

    @Cache.use_cache
    def d_delayL1_d_PB(self):
        return (self.d_delayL1_d_E() * self.d_E_d_PB()).to(u.s/u.day, equivalencies=u.dimensionless_angles())

    @Cache.use_cache
    def d_delayL2_d_PB(self):
        return (self.d_delayL2_d_E() * self.d_E_d_PB()).to(u.s/u.day, equivalencies=u.dimensionless_angles())

    @Cache.use_cache
    def d_delayL1_d_PBDOT(self):
        return (self.d_delayL1_d_E() * self.d_E_d_PBDOT())

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
        return (self.delayR() * (self.d_delayL1_d_EDOT() + \
                self.d_delayL2_d_EDOT())).to(u.s * u.s)

    @Cache.use_cache
    def d_delay_d_PB(self):
        return self.delayR() * (self.d_delayL1_d_PB() + self.d_delayL2_d_PB())

    @Cache.use_cache
    def d_delay_d_PBDOT(self):
        return self.delayR() * (self.d_delayL1_d_PBDOT() + \
                self.d_delayL2_d_PBDOT())

    @Cache.use_cache
    def d_delay_d_T0(self):
        return (self.delayR() * (self.d_delayL1_d_T0() + self.d_delayL2_d_T0())).to(u.s/u.day)

    @Cache.use_cache
    def d_delay_d_GAMMA(self):
        return self.delayR() * (self.d_delayL1_d_GAMMA() + self.d_delayL2_d_GAMMA())
