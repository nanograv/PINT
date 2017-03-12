# special_locations.py
from __future__ import division, print_function

# Special "site" location for NICER experiment

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

def load_FPorbit(orbit_filename):
    '''Load data from an (RXTE or NICER) FPorbit file

        Reads a FPorbit FITS file

        Parameters
        ----------
        orbit_filename : str
            Name of file to load

        Returns
        -------
        astropy Table containing Time, x, y, z, v_x, v_y, v_z data

    '''
    # Load photon times from FT1 file
    hdulist = pyfits.open(orbit_filename)
    FPorbit_hdr=hdulist[1].header
    FPorbit_dat=hdulist[1].data

    log.info('Opened FPorbit FITS file {0}'.format(orbit_filename))
    # TIMESYS should be 'TT'

    # TIMEREF should be 'LOCAL', since no delays are applied

    timesys = FPorbit_hdr['TIMESYS']
    log.info("FPorbit TIMESYS {0}".format(timesys))
    timeref = FPorbit_hdr['TIMEREF']
    log.info("FPorbit TIMEREF {0}".format(timeref))

    mjds_TT = read_fits_event_mjds(hdulist[1])
    mjds_TT = mjds_TT*u.d
    X = FPorbit_dat.field('X')*u.m
    Y = FPorbit_dat.field('Y')*u.m
    Z = FPorbit_dat.field('Z')*u.m
    Vx = FPorbit_dat.field('Vx')*u.m/u.s
    Vy = FPorbit_dat.field('Vy')*u.m/u.s
    Vz = FPorbit_dat.field('Vz')*u.m/u.s
    log.info('Building FPorbit table covering MJDs {0} to {1}'.format(mjds_TT.min(), mjds_TT.max()))
    FPorbit_table = Table([mjds_TT, X, Y, Z, Vx, Vy, Vz],
            names = ('MJD_TT', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz'),
            meta = {'name':'FPorbit'} )
    return FPorbit_table

class NICERObs(SpecialLocation):
    """Observatory-derived class for the NICER photon data.

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
        super(NICERObs, self).__init__(name=name)

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
        '''Return position and velocity vectors of NICER.

        t is an astropy.Time or array of astropy.Times
        '''
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB('earth', t, ephem)
        # Now add vector from Earth to NICER
        nicer_pos_geo = np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])*self.FPorb['X'].unit
        nicer_vel_geo = np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])*self.FPorb['Vx'].unit
        nicer_posvel = PosVel( nicer_pos_geo, nicer_vel_geo, origin='earth', obj='nicer')
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + nicer_posvel
