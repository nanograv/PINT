# astrometry.py
# Defines Astrometry timing model class
import numpy
import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
import parameter as p
from .timing_model import TimingModel, MissingParameter, Cache
from ..utils import time_from_mjd_string, time_to_longdouble, str2longdouble
from pint import ls
from pint import utils
import time

mas_yr = (u.mas / u.yr)

try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY

class Astrometry(TimingModel):

    def __init__(self):
        super(Astrometry, self).__init__()

        self.add_param(p.AngleParameter(name="RAJ",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"]))

        self.add_param(p.AngleParameter(name="DECJ",
            units="D:M:S",
            description="Declination (J2000)",
            aliases=["DECJ"]))

        self.add_param(p.MJDParameter(name="POSEPOCH",
            description="Reference epoch for position"))

        self.add_param(p.floatParameter(name="PMRA",
            units="mas/year", value=0.0,
            description="Proper motion in RA"))

        self.add_param(p.floatParameter(name="PMDEC",
            units="mas/year", value=0.0,
            description="Proper motion in DEC"))

        self.add_param(p.floatParameter(name="PX",
            units="mas", value=0.0,
            description="Parallax"))

        self.delay_funcs['L1'] += [self.solar_system_geometric_delay,]

    def setup(self):
        super(Astrometry, self).setup()
        # RA/DEC are required
        for p in ("RAJ", "DECJ"):
            if getattr(self, p).value is None:
                raise MissingParameter("Astrometry", p)
        # If PM is included, check for POSEPOCH
        if self.PMRA.value != 0.0 or self.PMDEC.value != 0.0:
            if self.POSEPOCH.quantity is None:
                if self.PEPOCH.quantity is None:
                    raise MissingParameter("Astrometry", "POSEPOCH",
                            "POSEPOCH or PEPOCH are required if PM is set.")
                else:
                    self.POSEPOCH.value = self.PEPOCH.value
        self.delay_derivs += [self.d_delay_astrometry_d_RAJ,
                              self.d_delay_astrometry_d_DECJ,
                              self.d_delay_astrometry_d_PMRA,
                              self.d_delay_astrometry_d_PMDEC,
                              self.d_delay_astrometry_d_PX]


    @Cache.cache_result
    def coords_as_ICRS(self, epoch=None):
        """Returns pulsar sky coordinates as an astropy ICRS object instance.

        If epoch (MJD) is specified, proper motion is included to return
        the position at the given epoch.
        """
        if epoch is None or (self.PMRA.value == 0.0 and self.PMDEC.value == 0.0):
            return coords.ICRS(ra=self.RAJ.quantity, dec=self.DECJ.quantity)
        else:
            dt = (epoch - self.POSEPOCH.quantity.mjd) * u.d
            dRA = dt * self.PMRA.quantity / numpy.cos(self.DECJ.quantity.radian)
            dDEC = dt * self.PMDEC.quantity
            return coords.ICRS(ra=self.RAJ.quantity+dRA, dec=self.DECJ.quantity+dDEC)

    @Cache.cache_result
    def ssb_to_psb_xyz(self, epoch=None):
        """Returns unit vector(s) from SSB to pulsar system barycenter.

        If epochs (MJD) are given, proper motion is included in the calculation.
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?
        return self.coords_as_ICRS(epoch=epoch).cartesian.xyz.transpose()

    @Cache.cache_result
    def barycentric_radio_freq(self, toas):
        """Return radio frequencies (MHz) of the toas corrected for Earth motion"""
        L_hat = self.ssb_to_psb_xyz(epoch=toas['tdbld'].astype(numpy.float64))
        v_dot_L_array = numpy.sum(toas['ssb_obs_vel']*L_hat, axis=1)
        return toas['freq'] * (1.0 - v_dot_L_array / const.c)

    def solar_system_geometric_delay(self, toas):
        """Returns geometric delay (in sec) due to position of site in
        solar system.  This includes Roemer delay and parallax.

        NOTE: currently assumes XYZ location of TOA relative to SSB is
        available as 3-vector toa.xyz, in units of light-seconds.
        """
        L_hat = self.ssb_to_psb_xyz(epoch=toas['tdbld'].astype(numpy.float64))
        re_dot_L = numpy.sum(toas['ssb_obs_pos']*L_hat, axis=1)
        delay = -re_dot_L.to(ls).value
        if self.PX.value != 0.0:
            L = ((1.0 / self.PX.value) * u.kpc)
            # TODO: numpy.sum currently loses units in some cases...
            re_sqr = numpy.sum(toas['ssb_obs_pos']**2, axis=1) * toas['ssb_obs_pos'].unit**2
            delay += (0.5 * (re_sqr / L) * (1.0 - re_dot_L**2 / re_sqr)).to(ls).value
        return delay

    @Cache.use_cache
    def get_d_delay_quantities(self, toas):
        """Calculate values needed for many d_delay_d_param functions """
        # TODO: Move all these calculations in a separate class for elegance
        rd = dict()

        # TODO: Should delay not have units of u.second?
        delay = self.delay(toas)

        # TODO: toas['tdbld'].quantity should have units of u.day
        # NOTE: Do we need to include the delay here?
        rd['epoch'] = toas['tdbld'].quantity * u.day #- delay * u.second

        # Distance from SSB to observatory, and from SSB to psr
        ssb_obs = toas['ssb_obs_pos'].quantity
        ssb_psr = self.ssb_to_psb_xyz(epoch=numpy.array(rd['epoch']))

        # Cartesian coordinates, and derived quantities
        rd['ssb_obs_r'] = numpy.sqrt(numpy.sum(ssb_obs**2, axis=1))
        rd['ssb_obs_z'] = ssb_obs[:,2]
        rd['ssb_obs_xy'] = numpy.sqrt(ssb_obs[:,0]**2 + ssb_obs[:,1]**2)
        rd['ssb_obs_x'] = ssb_obs[:,0]
        rd['ssb_obs_y'] = ssb_obs[:,1]
        rd['in_psr_obs'] = numpy.sum(ssb_obs * ssb_psr, axis=1)

        # Earth right ascension and declination
        rd['earth_dec'] = numpy.arctan2(rd['ssb_obs_z'], rd['ssb_obs_xy'])
        rd['earth_ra'] = numpy.arctan2(rd['ssb_obs_y'], rd['ssb_obs_x'])

        return rd


    @Cache.use_cache
    def d_delay_astrometry_d_RAJ(self, toas):
        """Calculate the derivative wrt RAJ

        For the RAJ and DEC derivatives, use the following approximate model for
        the pulse delay. (Inner-product between two Cartesian vectors)

        de = Earth declination (wrt SSB)
        ae = Earth right ascension
        dp = pulsar declination
        aa = pulsar right ascension
        r = distance from SSB to Earh
        c = speed of light

        delay = r*[cos(de)*cos(dp)*cos(ae-aa)+sin(de)*sin(dp)]/c
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity
        psr_dec = self.DECJ.quantity

        geom = numpy.cos(rd['earth_dec'])*numpy.cos(psr_dec)*\
                numpy.sin(psr_ra-rd['earth_ra'])
        dd_draj = rd['ssb_obs_r'] * geom / (const.c * u.radian)

        return dd_draj.decompose(u.si.bases)

    @Cache.use_cache
    def d_delay_astrometry_d_DECJ(self, toas):
        """Calculate the derivative wrt DECJ

        Definitions as in d_delay_d_RAJ
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity
        psr_dec = self.DECJ.quantity

        geom = numpy.cos(rd['earth_dec'])*numpy.sin(psr_dec)*\
                numpy.cos(psr_ra-rd['earth_ra']) - numpy.sin(rd['earth_dec'])*\
                numpy.cos(psr_dec)
        dd_ddecj = rd['ssb_obs_r'] * geom / (const.c * u.radian)

        return dd_ddecj.decompose(u.si.bases)

    @Cache.use_cache
    def d_delay_astrometry_d_PMRA(self, toas):
        """Calculate the derivative wrt PMRA

        Definitions as in d_delay_d_RAJ. Now we have a derivative in mas/yr for
        the pulsar RA
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity

        te = rd['epoch'] - time_to_longdouble(self.POSEPOCH.quantity) * u.day
        geom = numpy.cos(rd['earth_dec'])*numpy.sin(psr_ra-rd['earth_ra'])

        deriv = rd['ssb_obs_r'] * geom * te / (const.c * u.radian)
        dd_dpmra = deriv * u.mas / u.year

        # We want to return sec / (mas / yr)
        return dd_dpmra.decompose(u.si.bases) / (u.mas / u.year)

    @Cache.use_cache
    def d_delay_astrometry_d_PMDEC(self, toas):
        """Calculate the derivative wrt PMDEC

        Definitions as in d_delay_d_RAJ. Now we have a derivative in mas/yr for
        the pulsar DEC
        """
        rd = self.get_d_delay_quantities(toas)

        psr_ra = self.RAJ.quantity
        psr_dec = self.DECJ.quantity

        te = rd['epoch'] - time_to_longdouble(self.POSEPOCH.quantity) * u.day
        geom = numpy.cos(rd['earth_dec'])*numpy.sin(psr_dec)*\
                numpy.cos(psr_ra-rd['earth_ra']) - numpy.cos(psr_dec)*\
                numpy.sin(rd['earth_dec'])

        deriv = rd['ssb_obs_r'] * geom * te / (const.c * u.radian)
        dd_dpmdec = deriv * u.mas / u.year

        # We want to return sec / (mas / yr)
        return dd_dpmdec.decompose(u.si.bases) / (u.mas / u.year)

    @Cache.use_cache
    def d_delay_astrometry_d_PX(self, toas):
        """Calculate the derivative wrt PX

        Roughly following Smart, 1977, chapter 9.

        px_r:   Extra distance to Earth, wrt SSB, from pulsar
        r_e:    Position of earth (vector) wrt SSB
        u_p:    Unit vector from SSB pointing to pulsar
        t_d:    Parallax delay
        c:      Speed of light
        delta:  Parallax

        The parallax delay is due to a distance orthogonal to the line of sight
        to the pulsar from the SSB:

        px_r = sqrt( r_e**2 - (r_e.u_p)**2 ),

        with delay

        t_d = 0.5 * px_r * delta'/ c,  and delta = delta' * px_r / (1 AU)

        """
        rd = self.get_d_delay_quantities(toas)

        px_r = numpy.sqrt(rd['ssb_obs_r']**2-rd['in_psr_obs']**2)
        dd_dpx = 0.5*(px_r**2 / (u.AU*const.c)) * (u.mas / u.radian)

        # We want to return sec / mas
        return dd_dpx.decompose(u.si.bases) / u.mas

    @Cache.cache_result
    def d_delay_astrometry_d_POSEPOCH(self, toas):
        """Calculate the derivative wrt POSEPOCH
        """
        pass
