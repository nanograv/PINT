# special_locations.py
from __future__ import division, print_function

# Special "site" location for RXTE satellite

from . import Observatory
from .special_locations import SpecialLocation
import astropy.units as u
from astropy.coordinates import GCRS, ITRS, EarthLocation, CartesianRepresentation
from ..utils import PosVel
from ..fits_utils import read_fits_event_mjds
from ..solar_system_ephemerides import objPosVel_wrt_SSB
import numpy as np
from astropy.time import Time
from astropy.table import Table
import astropy.io.fits as pyfits
from astropy.extern import six
from astropy import log
from scipy.interpolate import interp1d
from .nicer_obs import load_FPorbit

class RXTEObs(SpecialLocation):
    """Observatory-derived class for the RXTE photon data.

    Note that this must be instantiated once to be put into the Observatory registry.

    Parameters
    ----------

    name: str
        Observatory name
    ft2name: str
        File name to read spacecraft position information from
    tt2tdb_mode: str
        Selection for mode to use for TT to TDB conversion.
        'none' = Give no position to astropy.Time()
        'geo' = Give geocenter position to astropy.Time()
        'spacecraft' = Give spacecraft ITRF position to astropy.Time()
    """

    def __init__(self, name, FPorbname, tt2tdb_mode = 'none'):
        self.FPorb = load_FPorbit(FPorbname)
        # Now build the interpolator here:
        self.X = interp1d(self.FPorb['MJD_TT'],self.FPorb['X'])
        self.Y = interp1d(self.FPorb['MJD_TT'],self.FPorb['Y'])
        self.Z = interp1d(self.FPorb['MJD_TT'],self.FPorb['Z'])
        self.Vx = interp1d(self.FPorb['MJD_TT'],self.FPorb['Vx'])
        self.Vy = interp1d(self.FPorb['MJD_TT'],self.FPorb['Vy'])
        self.Vz = interp1d(self.FPorb['MJD_TT'],self.FPorb['Vz'])
        self.tt2tdb_mode = tt2tdb_mode
        super(RXTEObs, self).__init__(name=name)

    @property
    def timescale(self):
        return 'tt'

    def earth_location_itrf(self, time=None):
        '''Return RXTE spacecraft location in ITRF coordinates'''

        if self.tt2tdb_mode.lower().startswith('none'):
            log.warning('Using location=None for TT to TDB conversion')
            return None
        elif self.tt2tdb_mode.lower().startswith('geo'):
            log.warning('Using location geocenter for TT to TDB conversion')
            return EarthLocation.from_geocentric(0.0*u.m,0.0*u.m,0.0*u.m)
        elif self.tt2tdb_mode.lower().startswith('spacecraft'):
            # First, interpolate ECI geocentric location from orbit file.
            # These are inertial coorinates aligned with ICRF
            pos_gcrs =  GCRS(CartesianRepresentation(self.X(time.tt.mjd)*u.m,
                                                     self.Y(time.tt.mjd)*u.m,
                                                     self.Z(time.tt.mjd)*u.m),
                             obstime=time)

            # Now transform ECI (GCRS) to ECEF (ITRS)
            # By default, this uses the WGS84 ellipsoid
            pos_ITRS = pos_gcrs.transform_to(ITRS(obstime=time))

            # Return geocentric ITRS coordinates as an EarthLocation object
            return pos_ITRS.earth_location
        else:
            log.error('Unknown tt2tdb_mode %s, using None', self.tt2tdb_mode)
            return None

    @property
    def tempo_code(self):
        return None

    def posvel(self, t, ephem):
        '''Return position and velocity vectors of RXTE.

        t is an astropy.Time or array of astropy.Times
        '''
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB('earth', t, ephem)
        # Now add vector from Earth to RXTE
        rxte_pos_geo = np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])*self.FPorb['X'].unit
        rxte_vel_geo = np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])*self.FPorb['Vx'].unit
        rxte_posvel = PosVel( rxte_pos_geo, rxte_vel_geo, origin='earth', obj='rxte')
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + rxte_posvel
