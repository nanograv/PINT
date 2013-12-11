# astrometry.py
# Defines Astrometry timing model class
import math
import astropy 
import astropy.coordinates as coords
from astropy.coordinates.angles import Angle
from timing_model import Parameter, TimingModel

class Astrometry(TimingModel):

    def __init__(self):
        super(Astrometry, self).__init__()

        self.add_param(Parameter(name="RA",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"],
            parse_value=lambda x: Angle(x+'h').hour,
            print_value=lambda x: Angle(x,unit='h').to_string(sep=':', 
                precision=8)))

        self.add_param(Parameter(name="DEC",
            units="D:M:S",
            description="Declination (J2000)",
            aliases=["DECJ"],
            parse_value=lambda x: Angle(x+'deg').degree,
            print_value=lambda x: Angle(x,unit='deg').to_string(sep=':',
                alwayssign=True, precision=8)))

        self.add_param(Parameter(name="POSEPOCH",
            units="MJD",
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

        # etc, also add  PX, ...

    def setup(self):
        super(Astrometry,self).setup()
        print "Astrometry setup"
    
    def ssb2psb_xyz(self,epoch=None):
        """
        ssb_to_bbc(epoch=None)

        Returns a XYZ unit vector pointing from the solar system barycenter
        to the pulsar system barycenter.   If epoch is given, proper motion
        is included in the calculation.
        """
        d_ra = 0.0
        d_dec = 0.0
        day2year = 1.0/365.24 # We need to make a place for these..
        deg2rad = math.pi / 180.0
        if epoch!=None and \
                (self.PMRA.value!=0.0 and self.PMDEC.value!=0.0):
            if self.POSEPOCH.value==None:
                self.POSEPOCH.value = self.PEPOCH.value
            dt = epoch - self.POSEPOCH.value
            # TODO: Check this formula for cos(dec).  Would be nice if astropy
            # could do proper motion, but I didn't see it so far..
            d_ra = self.PMRA.value * day2year * dt / math.cos(
                    self.DEC.value * deg2rad)
            d_dec = self.PMDEC.value * day2year * dt
        radec = coords.ICRS(ra=self.RA.value + d_ra,
                dec=self.DEC.value + d_dec,
                unit=('h','deg'))
        return radec.cartesian

