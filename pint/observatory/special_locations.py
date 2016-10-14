# special_locations.py

# Special "site" locations (eg, barycenter) which do not need clock
# corrections or much else done.

from . import Observatory
import numpy
import astropy.units as u
from ..utils import PosVel
from ..solar_system_ephemerides import objPosVel2SSB

class SpecialLocation(Observatory):
    def clock_corrections(self, t):
        return numpy.zeros(t.shape)*u.s

class BarycenterObs(SpecialLocation):
    def timescale(self): 
        return 'tdb'

    def posvel(self, t, ephem):
        vdim = (3,) + t.shape
        return PosVel(numpy.zeros(vdim)*u.m, numpy.zeros(vdim)*u.m/u.s,
                obj=self.name, origin='ssb')

class GeocenterObs(SpecialLocation):
    def timescale(self): 
        return 'utc' # OK?

    def posvel(self, t, ephem):
        return obsPosVel2SSB('earth', t, ephem)

# Need to initialize one of each so that it gets added to the list
BarycenterObs('barycenter', aliases=['@','ssb','bary'])
GeocenterObs('geocenter', aliases=['0','o','coe','geo'])
