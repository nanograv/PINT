# special_locations.py
from __future__ import division, print_function

# Special "site" location for RXTE satellite

from . import Observatory
from .special_locations import SpecialLocation
import astropy.units as u
from astropy.coordinates import EarthLocation
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

    Note that this must be instantiated once to be put into the Observatory registry."""

    def __init__(self, name, FPorbname):
        self.FPorb = load_FPorbit(FPorbname)
        # Now build the interpolator here:
        self.X = interp1d(self.FPorb['MJD_TT'],self.FPorb['X'])
        self.Y = interp1d(self.FPorb['MJD_TT'],self.FPorb['Y'])
        self.Z = interp1d(self.FPorb['MJD_TT'],self.FPorb['Z'])
        self.Vx = interp1d(self.FPorb['MJD_TT'],self.FPorb['Vx'])
        self.Vy = interp1d(self.FPorb['MJD_TT'],self.FPorb['Vy'])
        self.Vz = interp1d(self.FPorb['MJD_TT'],self.FPorb['Vz'])
        super(RXTEObs, self).__init__(name=name)

    @property
    def timescale(self):
        return 'tt'

    @property
    def earth_location(self):
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
