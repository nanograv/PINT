# astrometry.py
# Defines Astrometry timing model class
import numpy
import astropy 
import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const
from astropy.coordinates.angles import Angle
from timing_model import Parameter, MJDParameter, TimingModel, MissingParameter

# No light-seconds in astropy, WTF? ;)
ls = u.def_unit('ls', const.c * 1.0 * u.s)

class Astrometry(TimingModel):

    def __init__(self):
        super(Astrometry, self).__init__()

        self.add_param(Parameter(name="RA",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"],
            parse_value=lambda x: Angle(x+'h'),
            print_value=lambda x: x.to_string(sep=':', 
                precision=8)))

        self.add_param(Parameter(name="DEC",
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
        super(Astrometry,self).setup()
        # RA/DEC are required
        for p in ("RA","DEC"):
            if getattr(self,p).value is None:
                raise MissingParameter("Astrometry",p)
        # If PM is included, check for POSEPOCH
        if self.PMRA.value!=0.0 or self.PMDEC.value!=0.0: 
            if self.POSEPOCH.value is None:
                if self.PEPOCH.value is None:
                    raise MissingParameter("Astrometry","POSEPOCH",
                            "POSEPOCH or PEPOCH are required if PM is set.")
                else:
                    self.POSEPOCH.value = self.PEPOCH.value

    def coords_as_ICRS(self,epoch=None):
        """
        coords_as_ICRS(epoch=None)

        Returns pulsar sky coordinates as an astropy ICRS object instance.  
        If epoch (MJD) is specified, proper motion is included to return 
        the position at the given epoch.
        """
        if epoch is None:
            return coords.ICRS(ra=self.RA.value, dec=self.DEC.value)
        else:
            dt = epoch - self.POSEPOCH.value
            dRA = (dt * self.PMRA.value * (u.mas/u.yr)
                    / numpy.cos(self.DEC.value)).to(u.mas)
            dDEC = (dt * self.PMDEC.value * (u.mas/u.yr)).to(u.mas)
            return coords.ICRS(ra=self.RA.value+dRA, dec=self.DEC.value+dDEC)
    
    def ssb_to_psb_xyz(self,epoch=None):
        """
        ssb_to_psb(epoch=None)

        Returns a XYZ unit vector pointing from the solar system barycenter
        to the pulsar system barycenter.   If epoch (MJD) is given, proper 
        motion is included in the calculation.
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?
        return self.coords_as_ICRS(epoch=epoch).cartesian

    def solar_system_geometric_delay(self,toa):
        """
        solar_system_geometric_delay(toa)

        Returns geometric delay (in sec) due to position of site in 
        solar system.  This includes Roemer delay and parallax.

        NOTE: currently assumes XYZ location of TOA relative to SSB is
        available as 3-vector toa.xyz, in units of light-seconds.
        """
        L_hat = self.ssb_to_psb_xyz(epoch=toa.mjd)
        re_dot_L = toa.obs_pvs.pos.dot(L_hat)
        delay = -re_dot_L.to(ls).value
        if self.PX.value!=0.0:
            L = ((1.0/self.PX.value)*u.kpc)
            re_sqr = toa.obs_pvs.pos.dot(toa.obs_pvs.pos)
            delay += (0.5*(re_sqr/L)*(1.0-re_dot_L**2/re_sqr)).to(ls).value
        return delay

