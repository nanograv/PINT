# special_locations.py
from __future__ import division, print_function

# Special "site" location for Fermi satelite

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

def load_FT2(ft2_filename):
    '''Load data from a Fermi FT2 file

        The contents of the FT2 file are described here:
        https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_Data_Columns.html#SpacecraftFile
        The coordinates are X, Y, Z in the ECI (Earth-Centered Inertial)
        frame. I (@paulray) **believe** this is the same as astropy's GCRS
        <http://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html>,
        but this should be confirmed.

        Parameters
        ----------
        ft2_filename : str
            Name of file to load

        Returns
        -------
        astropy Table containing Time, x, y, z, v_x, v_y, v_z data

    '''
    # Load photon times from FT1 file
    hdulist = pyfits.open(ft2_filename)
    FT2_hdr=hdulist[1].header
    FT2_dat=hdulist[1].data

    log.info('Opened FT2 FITS file {0}'.format(ft2_filename))
    # TIMESYS should be 'TT'
    # TIMEREF should be 'LOCAL', since no delays are applied
    timesys = FT2_hdr['TIMESYS']
    log.info("FT2 TIMESYS {0}".format(timesys))
    timeref = FT2_hdr['TIMEREF']
    log.info("FT2 TIMEREF {0}".format(timeref))

    # The X, Y, Z position are for the START time
    mjds_TT = read_fits_event_mjds(hdulist[1],timecolumn='START')
    mjds_TT = mjds_TT*u.d
    # SC_POS is in meters in X,Y,Z Earth-centered Inertial (ECI) coordinates
    SC_POS = FT2_dat.field('SC_POSITION')
    X = SC_POS[:,0]*u.m
    Y = SC_POS[:,1]*u.m
    Z = SC_POS[:,2]*u.m
    # Compute velocities by differentiation because FT2 does not have velocities
    # This is not the best way. Should fit an orbit and determine velocity from that.
    dt = mjds_TT[1]-mjds_TT[0]
    log.info('FT2 spacing is '+str(dt.to(u.s)))
    # Trim off last point because array.diff() is one shorter
    Vx = np.gradient(X)[:-1]/(mjds_TT.diff().to(u.s))
    Vy = np.gradient(Y)[:-1]/(mjds_TT.diff().to(u.s))
    Vz = np.gradient(Z)[:-1]/(mjds_TT.diff().to(u.s))
    X = X[:-1]
    Y = Y[:-1]
    Z = Z[:-1]
    mjds_TT = mjds_TT[:-1]
    log.info('Building FT2 table covering MJDs {0} to {1}'.format(mjds_TT.min(), mjds_TT.max()))
    FT2_table = Table([mjds_TT, X, Y, Z, Vx, Vy, Vz],
            names = ('MJD_TT', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz'),
            meta = {'name':'FT2'} )
    return FT2_table

class FermiObs(SpecialLocation):
    """Observatory-derived class for the Fermi FT1 data.

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

    def __init__(self, name, ft2name, tt2tdb_mode = 'NONE'):
        self.FT2 = load_FT2(ft2name)
        # Now build the interpolator here:
        self.X = InterpolatedUnivariateSpline(self.FT2['MJD_TT'],self.FT2['X'])
        self.Y = InterpolatedUnivariateSpline(self.FT2['MJD_TT'],self.FT2['Y'])
        self.Z = InterpolatedUnivariateSpline(self.FT2['MJD_TT'],self.FT2['Z'])
        self.Vx = InterpolatedUnivariateSpline(self.FT2['MJD_TT'],self.FT2['Vx'])
        self.Vy = InterpolatedUnivariateSpline(self.FT2['MJD_TT'],self.FT2['Vy'])
        self.Vz = InterpolatedUnivariateSpline(self.FT2['MJD_TT'],self.FT2['Vz'])
        self.tt2tdb_mode = tt2tdb_mode
        super(FermiObs, self).__init__(name=name)

    @property
    def timescale(self):
        return 'tt'

    def earth_location_itrf(self, time=None):
        '''Return Fermi spacecraft location in ITRF coordinates'''

        if self.tt2tdb_mode.lower().startswith('none'):
            log.warning('Using location=None for TT to TDB conversion')
            return None
        elif self.tt2tdb_mode.lower().startswith('geo'):
            log.warning('Using location geocenter for TT to TDB conversion')
            return EarthLocation.from_geocentric(0.0*u.m,0.0*u.m,0.0*u.m)
        elif self.tt2tdb_mode.lower().startswith('spacecraft'):
            # First, interpolate Earth-Centered Inertial (ECI) geocentric
            # location from orbit file.
            # These are inertial coordinates aligned with ICRS, called GCRS
            # <http://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html>
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
        '''Return position and velocity vectors of Fermi, wrt SSB.

        These positions and velocites are in inertial coordinates
        (i.e. aligned with ICRS)

        t is an astropy.Time or array of astropy.Times
        '''
        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB('earth', t, ephem)
        # Now add vector from Earth to Fermi
        fermi_pos_geo = np.array([self.X(t.tt.mjd), self.Y(t.tt.mjd), self.Z(t.tt.mjd)])*self.FT2['X'].unit
        fermi_vel_geo = np.array([self.Vx(t.tt.mjd), self.Vy(t.tt.mjd), self.Vz(t.tt.mjd)])*self.FT2['Vx'].unit
        fermi_posvel = PosVel( fermi_pos_geo, fermi_vel_geo, origin='earth', obj='Fermi')
        # Vector add to geo_posvel to get full posvel vector.
        return geo_posvel + fermi_posvel
