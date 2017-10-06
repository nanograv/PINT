from .DD_model import DDmodel
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy import log
from pint import ls,GMsun,Tsun

class DDKmodel(DDmodel):
    """
    This is a class for implement DDK model, a Kopeikin method corrected DD model.
    The main difference is that DDK model considers the annual parallax of earth and
    the proper motion of the pulsar.
    effects on the pulsar binary parameters.
    Speical parameters are:
    @ KIN the inclination angle
    @ KOM the longitude of the ascending node, Kopeikin (1995) Eq 9. OMEGA

    """
    def __init__(self, t=None, input_params=None):
        super(DDKmodel, self).__init__()
        self.binary_name = 'DDK'
        # Add parameter that specific for DD model, with default value and units
        self.param_default_value.update({'KIN':0*u.deg,'PMRA':0*u.mas/u.year,
                               'PMDEC':0.5*u.mas/u.year, 'PX': 0*u.mas,
                               'KOM': 0*u.deg
                               })
        # Remove unused parameter SINI
        del self.param_default_value['SINI']
        # If any parameter has aliases, it should be updated
        #self.param_aliases.update({})
        self.binary_params = list(self.param_default_value.keys())
        self.set_param_values()

    @property
    def KOM(self):
        return self._KOM

    @KOM.setter
    def KOM(self,val):
        self._KOM = val
        self._sin_KOM = np.sin(self._KOM)
        self._cos_KOM = np.cos(self._KOM)

    @property
    def sin_KOM(self):
        return self._sin_KOM

    @property
    def cos_KOM(self):
        return self._cos_KOM

    @property
    def psr_pos(self):
        return self._psr_pos

    @psr_pos.setter
    def psr_pos(self, val):
        """The pointing unit vector of a pulsar. alpha and delta described in
        (Kopeikin 1995 L6 Eq10)
        """
        self._psr_pos = val
        self._sin_delta = self._psr_pos[:,2]
        self._cos_delta = np.cos(np.arcsin(self._sin_delta))
        self._sin_alpha = self._psr_pos[:,1] / self._cos_delta
        self._cos_alpha = self._psr_pos[:,0] / self._cos_delta

    @property
    def sin_delta(self):
        return self._sin_delta

    @property
    def cos_delta(self):
        return self._cos_delta

    @property
    def sin_alpha(self):
        return self._sin_alpha

    @property
    def cos_alpha(self):
        return self._cos_alpha

    @property
    def SINI(self):
        return np.sin(self.KIN) + self.delta_sini_parallax()

    @SINI.setter
    def SINI(self, val):
        log.warning("DDK model uses KIN as inclination angle. SINI will not be"
        " used.")

    # @property
    # def SINI(self):
    #     return np.sin()
    # The code below is to apply the KOPEIKIN correction due to pulser proper motion
    # Reference:  KOPEIKIN. 1996 Eq 7 -> Eq 10.
    # Update binary parameters due to the pulser proper motion

    def delta_kin_proper_motion(self):
        """The time dependent inclination angle.
        (KOPEIKIN. 1996 Eq 10.)
        ki = KIN + d_KIN
        d_KIN = (-PMRA * sin(KOM) + PMDEC * cos(KOM)) * (t-T0)
        """
        d_KIN = (-self.PMRA * self.sin_KOM +
                  self.PMDEC * self.cos_KOM) * self.tt0
        return d_KIN.to(self.KIN.unit)

    def kin_proper_motion(self):
        return self.KIN + self.delta_kin_proper_motion()

    def delta_a1_proper_motion(self):
        """The correction on a1 (projected semi-major axis)
        due to the pulsar proper motion
        (KOPEIKIN. 1996 Eq 8.)
        d_x = a1 * cot(kin) * (-PMRA * sin(KOM) + PMDEC * cos(KOM)) * (t-T0)
        d_kin = (-PMRA * sin(KOM) + PMDEC * cos(KOM)) * (t-T0)
        d_x = a1 * d_kin * cot(kin)
        """
        a1 = self.a1(False, False)
        kin = self.kin_proper_motion()
        tan_kin = np.tan(kin)
        d_kin = self.delta_kin_proper_motion()
        d_a1 = a1 * d_kin / tan_kin
        return d_a1.to(a1.unit, equivalencies=u.dimensionless_angles())

    def delta_omega_proper_motion(self):
        """The correction on omega (Longitude of periastron)
        due to the pulsar proper motion
        (KOPEIKIN. 1996 Eq 9.)
        d_omega = csc(i) * (PMRA * cos(KOM) + PMDEC * sin(KOM)) * (t-T0)
        """
        kin = self.kin_proper_motion()
        sin_kin = np.sin(kin)
        omega_dot = 1.0/sin_kin * (self.PMRA * self.cos_KOM +
                                        self.PMDEC * self.sin_KOM)
        return (omega_dot * self.tt0).to(self.OM.unit)


    # The code below is to implement the binary model parameter correction due
    # to the parallax.
    # Reference KOPEIKIN. 1995 Eq 18 -> Eq 19.

    def delta_I0(self):
        """
        Refernce: (Kopeikin 1995 Eq 15)
        """
        return -self.obs_pos[:,0] * self.sin_alpha + self.obs_pos[:,1] * self.cos_alpha

    def delta_J0(self):
        """
        Reference: (Kopeikin 1995 Eq 16)
        """
        return -self.obs_pos[:,0] * self.sin_delta * self.cos_alpha - \
                self.obs_pos[:,1] * self.sin_delta * self.sin_alpha + \
                self.obs_pos[:,2] * self.cos_delta

    def delta_sini_parallax(self):
        """
        Reference (Kopeikin 1995 Eq 18)
        x_obs = ap * sini_obs/c
        Since ap and c will not be changed by parallax.
        x_obs = ap /c *(sini_intrisic + delta_sini)
        delta_sini = sini_intrisic * coti_intrisic / d * (deltaI0 * sin_kom - deltaJ0 * cos_kom)
        """
        PX_kpc= self.PX.to(u.kpc, equivalencies=u.parallax())
        delta_sini = np.cos(self.KIN) / PX_kpc * (self.delta_I0() * self.sin_KOM - \
                                                  self.delta_J0() * self.cos_KOM)
        return delta_sini.to("")

    def delta_a1_parallax(self):
        """
        Reference: (Kopeikin 1995 Eq 18)
        """
        a1 = self.a1(parallax=False)
        kin = self.kin_proper_motion()
        tan_kin = np.tan(kin)
        PX_kpc= self.PX.to(u.kpc, equivalencies=u.parallax())
        delta_a1 = a1 /tan_kin / PX_kpc * (self.delta_I0() * self.sin_KOM - \
                                            self.delta_J0() * self.cos_KOM)
        return delta_a1.to(a1.unit, )

    def delta_omega_parallax(self):
        """
        Reference: (Kopeikin 1995 Eq 19)
        """
        kin = self.kin_proper_motion()
        sin_kin = np.sin(kin)
        PX_kpc= self.PX.to(u.kpc, equivalencies=u.parallax())
        delta_omega = -1.0 / sin_kin / PX_kpc * (self.delta_I0() * self.cos_KOM - \
                                                  self.delta_J0() * self.sin_KOM)
        return delta_omega.to(self.OM.unit, equivalencies=u.dimensionless_angles())

    def a1(self, proper_motion=True, parallax=True):
        """
        A function to compute Kopeikin corrected projected semi-major axis.
        Parameter
        ---------
        proper_motion: boolean, optional, default True
            Flag for proper_motion correction
        parallax: boolean, optional, default True
            Flag for parallax correction
        """
        a1 = super(DDKmodel, self).a1()
        corr_funs = [self.delta_a1_proper_motion, self.delta_a1_parallax]
        mask = [proper_motion, parallax]
        for ii, cf in enumerate(corr_funs):
            if mask[ii]:
                a1 += cf()
        return a1

    def omega(self, proper_motion=True, parallax=True):
        """
        A function to compute Kopeikin corrected projected omega.
        Parameter
        ---------
        proper_motion: boolean, optional, default True
            Flag for proper_motion correction
        parallax: boolean, optional, default True
            Flag for parallax correction
        """
        omega = super(DDKmodel, self).omega()
        corr_funs = [self.delta_omega_proper_motion, self.delta_omega_parallax]
        mask = [proper_motion, parallax]
        for ii, cf in enumerate(corr_funs):
            if mask[ii]:
                omega += cf()
        return omega
