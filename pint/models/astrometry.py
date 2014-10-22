# astrometry.py
# Defines Astrometry timing model class
import numpy
import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
from .parameter import Parameter, MJDParameter
from .timing_model import TimingModel, MissingParameter, Cache
from pint import ls
from pint import utils
import time

class Astrometry(TimingModel):

    def __init__(self):
        super(Astrometry, self).__init__()

        self.add_param(Parameter(name="RAJ",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"],
            parse_value=lambda x: Angle(x+'h'),
            print_value=lambda x: x.to_string(sep=':',
                precision=8)))

        self.add_param(Parameter(name="DECJ",
            units="D:M:S",
            description="Declination (J2000)",
            aliases=["DECJ"],
            parse_value=lambda x: Angle(x+'deg'),
            print_value=lambda x: x.to_string(sep=':',
                alwayssign=True, precision=8)))

        self.add_param(MJDParameter(name="POSEPOCH",
            description="Reference epoch for position"))

        self.add_param(Parameter(name="PMRA",
            units="mas/year", value=0.0,
            description="Proper motion in RA"))

        self.add_param(Parameter(name="PMDEC",
            units="mas/year", value=0.0,
            description="Proper motion in DEC"))

        self.add_param(Parameter(name="PX",
            units="mas", value=0.0,
            description="Parallax"))

        self.delay_funcs += [self.solar_system_geometric_delay,]

    def setup(self):
        super(Astrometry, self).setup()
        # RA/DEC are required
        for p in ("RAJ", "DECJ"):
            if getattr(self, p).value is None:
                raise MissingParameter("Astrometry", p)
        # If PM is included, check for POSEPOCH
        if self.PMRA.value != 0.0 or self.PMDEC.value != 0.0:
            if self.POSEPOCH.value is None:
                if self.PEPOCH.value is None:
                    raise MissingParameter("Astrometry", "POSEPOCH",
                            "POSEPOCH or PEPOCH are required if PM is set.")
                else:
                    self.POSEPOCH.value = self.PEPOCH.value

    @Cache.cache_result
    def coords_as_ICRS(self, epoch=None):
        """Returns pulsar sky coordinates as an astropy ICRS object instance.

        If epoch (MJD) is specified, proper motion is included to return
        the position at the given epoch.
        """
        if epoch is None:
            return coords.ICRS(ra=self.RAJ.value, dec=self.DECJ.value)
        else:
            mas_yr = (u.mas / u.yr)
            dt = (epoch - self.POSEPOCH.value.mjd) * u.d
            dRA = (dt * self.PMRA.value * mas_yr / \
                    numpy.cos(self.DECJ.value)).to(u.mas)
            dDEC = (dt * self.PMDEC.value * mas_yr).to(u.mas)
            return coords.ICRS(ra=self.RAJ.value+dRA, dec=self.DECJ.value+dDEC)
    
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
