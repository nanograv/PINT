# special_locations.py

# Special "site" locations (eg, barycenter) which do not need clock
# corrections or much else done.

from . import Observatory
import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation
from ..utils import PosVel
from ..solar_system_ephemerides import objPosVel_wrt_SSB

class SpecialLocation(Observatory):
    """Observatory-derived class for special sites that are not really
    observatories but sometimes are used as TOA locations (eg, solar
    system barycenter).  Currently the only feature of this class is
    that clock corrections are zero."""
    def clock_corrections(self, t):
        return numpy.zeros(t.shape)*u.s

class BarycenterObs(SpecialLocation):
    """Observatory-derived class for the solar system barycenter.  Time
    scale is assumed to be tdb."""
    @property
    def timescale(self):
        return 'tdb'
    @property
    def tempo_code(self):
        return '@'
    @property
    def tempo2_code(self):
        return 'bat'
    def posvel(self, t, ephem):
        vdim = (3,) + t.shape
        return PosVel(numpy.zeros(vdim)*u.m, numpy.zeros(vdim)*u.m/u.s,
                obj=self.name, origin='ssb')

class GeocenterObs(SpecialLocation):
    """Observatory-derived class for the Earth geocenter."""
    @property
    def timescale(self):
        return 'utc'
    def earth_location_itrf(self, time=None):
        return EarthLocation.from_geocentric(0.0,0.0,0.0,unit=u.m)
    @property
    def tempo_code(self):
        return '0'
    @property
    def tempo2_code(self):
        return 'coe'
    def posvel(self, t, ephem):
        return objPosVel_wrt_SSB('earth', t, ephem)

# Need to initialize one of each so that it gets added to the list
BarycenterObs('barycenter', aliases=['@','ssb','bary'])
GeocenterObs('geocenter', aliases=['0','o','coe','geo'])
