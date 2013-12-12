# astrometry.py
# Defines Astrometry timing model class
import numpy
import astropy 
import astropy.coordinates as coords
import astropy.units as u
from astropy.coordinates.angles import Angle
from timing_model import Parameter, TimingModel, MissingParameter

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

    def setup(self):
        super(Astrometry,self).setup()
        print "Astrometry setup"
        # RA/DEC are required
        for p in ("RA","DEC"):
            if getattr(self,p).value==None:
                raise MissingParameter("Astrometry",p)
        # If PM is included, check for POSEPOCH
        if self.PMRA.value!=0.0 or self.PMDEC.value!=0.0: 
            if self.POSEPOCH.value==None:
                if self.PEPOCH.value==None:
                    raise MissingParameter("Astrometry","POSEPOCH",
                            "POSEPOCH or PEPOCH are required if PM is set.")
                else:
                    self.POSEPOCH.value = self.PEPOCH.value

    def as_ICRS(self,epoch=None):
        """
        as_ICRS(epoch=None)

        Returns coordinates as an astropy ICRS instance.  If epoch (MJD) is
        specified, proper motion is included to return the position at
        the given epoch.
        """
        if epoch==None:
            return coords.ICRS(ra=self.RA.value, dec=self.DEC.value)
        else:
            dt = (epoch - self.POSEPOCH.value) * u.day
            dRA = self.PMRA.value*(u.mas/u.yr) * dt / numpy.cos(self.DEC.value)
            dDEC = self.PMDEC.value*(u.mas/u.yr) * dt
            return coords.ICRS(ra=self.RA.value+dRA, dec=self.DEC.value+dDEC)
    
    def ssb2psb_xyz(self,epoch=None):
        """
        ssb_to_psb(epoch=None)

        Returns a XYZ unit vector pointing from the solar system barycenter
        to the pulsar system barycenter.   If epoch (MJD) is given, proper 
        motion is included in the calculation.
        """
        # TODO: would it be better for this to return a 6-vector (pos, vel)?
        return self.as_ICRS(epoch=epoch).cartesian

