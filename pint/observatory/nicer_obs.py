# special_locations.py
from __future__ import division, print_function

# Special "site" location for NICER experiment

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
from scipy.interpolate import InterpolatedUnivariateSpline

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
    log.info("FPorbit spacing is {0}".format((mjds_TT[1]-mjds_TT[0]).to(u.s)))
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
        self.X = InterpolatedUnivariateSpline(self.FPorb['MJD_TT'],self.FPorb['X'])
        self.Y = InterpolatedUnivariateSpline(self.FPorb['MJD_TT'],self.FPorb['Y'])
        self.Z = InterpolatedUnivariateSpline(self.FPorb['MJD_TT'],self.FPorb['Z'])
        self.Vx = InterpolatedUnivariateSpline(self.FPorb['MJD_TT'],self.FPorb['Vx'])
        self.Vy = InterpolatedUnivariateSpline(self.FPorb['MJD_TT'],self.FPorb['Vy'])
        self.Vz = InterpolatedUnivariateSpline(self.FPorb['MJD_TT'],self.FPorb['Vz'])
        self.tt2tdb_mode = tt2tdb_mode
        super(NICERObs, self).__init__(name=name)

    @property
    def timescale(self):
        return 'tt'

    def earth_location_itrf(self, time=None):
        '''Return NICER spacecraft location in ITRF coordinates'''

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
