# special_locations.py

# Special "site" locations (eg, barycenter) which do not need clock
# corrections or much else done.
from __future__ import absolute_import, print_function, division
from . import Observatory
import numpy
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy import log
from ..utils import PosVel
from ..solar_system_ephemerides import objPosVel_wrt_SSB

class SpecialLocation(Observatory):
    """Observatory-derived class for special sites that are not really
    observatories but sometimes are used as TOA locations (eg, solar
    system barycenter).  Currently the only feature of this class is
    that clock corrections are zero."""
    def clock_corrections(self, t):
        log.info('Special observatory location. No clock corrections applied.')
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
    def get_gcrs(self, t, ephem=None, grp=None):
        if ephem is None:
            raise ValueError('Ephemeris needed for BarycenterObs get_gcrs') 
        ssb_pv = objPosVel_wrt_SSB('earth', t, ephem)
        return -1 * ssb_pv.pos
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
    def get_gcrs(self, t, ephem=None, grp=None):
        vdim = (3,) + t.shape
        return numpy.zeros(vdim) * u.m
    def posvel(self, t, ephem):
        return objPosVel_wrt_SSB('earth', t, ephem)

class SpacecraftObs(SpecialLocation):
    """Observatory-derived class for a spacecraft observatory."""
    @property
    def timescale(self):
        return 'utc'
    
    def get_gcrs(self, t, ephem=None, grp=None):
        """Return spacecraft GCRS position; this assumes position flags in tim file are in km"""

        if grp is None:
            raise ValueError('TOA group table needed for SpacecraftObs get_gcrs')

        try:
            # Is there a better way to do this?
            x = numpy.array([flags['telx'] for flags in grp['flags']])
            y = numpy.array([flags['tely'] for flags in grp['flags']])
            z = numpy.array([flags['telz'] for flags in grp['flags']])
        except:
            log.error('Missing flag. TOA line should have x,y,z flags for GCRS position in km.')
            raise Exception('Missing flag. TOA line should have x,y,z flags for GCRS position in km.')
            
        pos = numpy.vstack((x,y,z))
        vdim = (3,) + t.shape
        if pos.shape != vdim:
            raise ValueError('GCRS position vector has wrong shape: ',pos.shape,' instead of ',vdim.shape)

        return pos * u.km

    def posvel_gcrs(self, t, grp):
        """Return spacecraft GCRS position and velocity; this assumes position flags in tim file are in km and velocity flags are in km/s"""
        
        if grp is None:
            raise ValueError('TOA group table needed for SpacecraftObs posvel_gcrs')

        try:
            # Is there a better way to do this?
            vx = numpy.array([flags['vx'] for flags in grp['flags']])
            vy = numpy.array([flags['vy'] for flags in grp['flags']])
            vz = numpy.array([flags['vz'] for flags in grp['flags']])
        except:
            log.error('Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s.')
            raise Exception('Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s.')
            
        vel_geo = numpy.vstack((vx,vy,vz)) * (u.km/u.s)
        vdim = (3,) + t.shape
        if vel_geo.shape != vdim:
            raise ValueError('GCRS velocity vector has wrong shape: ',vel.shape,' instead of ',vdim.shape)

        pos_geo = self.get_gcrs(t,ephem=None,grp=grp)
        
        stl_posvel = PosVel(pos_geo, vel_geo, origin='earth', obj='spacecraft')
        return stl_posvel

    def posvel(self, t, ephem, grp):

        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB('earth', t, ephem)

        # Spacecraft posvel w.r.t. Earth
        stl_posvel = self.posvel_gcrs(t,grp)

        # Vector add to geo_posvel to get full posvel vector w.r.t. SSB.
        return geo_posvel + stl_posvel

    
# Need to initialize one of each so that it gets added to the list
BarycenterObs('barycenter', aliases=['@','ssb','bary','bat'])
GeocenterObs('geocenter', aliases=['0','o','coe','geo'])
SpacecraftObs('spacecraft', aliases=['STL_GEO'])
