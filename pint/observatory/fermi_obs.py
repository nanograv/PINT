# special_locations.py
from __future__ import division, print_function

# Special "site" location for Fermi satelite

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

def load_FT2(ft2_filename):
    '''Load data from a Fermi FT2 file

        The contents of the FT2 file are described here:
        https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_Data_Columns.html#SpacecraftFile

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

    Note that this must be instantiated once to be put into the Observatory registry."""

    def __init__(self, name, ft2name):
        self.FT2 = load_FT2(ft2name)
        # Now build the interpolator here:
        self.X = interp1d(self.FT2['MJD_TT'],self.FT2['X'])
        self.Y = interp1d(self.FT2['MJD_TT'],self.FT2['Y'])
        self.Z = interp1d(self.FT2['MJD_TT'],self.FT2['Z'])
        self.Vx = interp1d(self.FT2['MJD_TT'],self.FT2['Vx'])
        self.Vy = interp1d(self.FT2['MJD_TT'],self.FT2['Vy'])
        self.Vz = interp1d(self.FT2['MJD_TT'],self.FT2['Vz'])
        super(FermiObs, self).__init__(name=name)

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
        '''Return position and velocity vectors of Fermi.

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
