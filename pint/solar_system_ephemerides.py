import numpy as np
import astropy.units as u
import astropy.coordinates as coor
from astropy.time import Time
from .utils import PosVel
from astropy import log
from jplephem.spk import SPK
from .config import datapath
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY

kernel_link_base = 'http://naif.jpl.nasa.gov/pub/naif/' + \
                   'generic_kernels/spk/planets/a_old_versions/'
jpl_obj_code = {'ssb': 0,
                'sun': 10,
                'mercury': 199,
                'venus': 299,
                'earth-moon-barycenter': 3,
                'earth': 399,
                'moon': 301,
                'mars': 499,
                'jupiter': 5,
                'saturn': 6,
                'uranus': 7,
                'neptune': 8,
                'pluto': 9}

def objPosVel_wrt_SSB(objname, t, ephem):
    """This function computes a solar system object position and velocity with respect
    to the solar system barycenter (SSB) using astropy coordinates get_body_barycentric()
    method.
    
    The coordinate frame is that of the underlying solar system ephemeris, which 
    has been the ICRF (J2000) since the DE4XX series.
    
    Parameters
    ----------
    objname: str
        Solar system object name. Current support solar system bodies are listed in
        astropy.coordinates.solar_system_ephemeris.bodies attribution.
    t: Astropy.time.Time object
        Observation time in Astropy.time.Time object format.
    ephem: str
        The ephem to for computing solar system object position and velocity
        
    Returns
    -------
    PosVel object with 3-vectors for the position and velocity of the object
    
    """
    ephem = ephem.lower()
    objname = objname.lower()
    # Use astropy to compute postion.
    try:
        pos, vel = coor.get_body_barycentric_posvel(objname, t, ephemeris=ephem)
    except ValueError:
        pos, vel = coor.get_body_barycentric_posvel(objname, t, ephemeris= \
                            kernel_link_base + "%s.bsp" % ephem)
    return PosVel(pos.xyz, vel.xyz.to(u.km/u.second), origin='ssb', obj=objname)

def objPosVel(obj1, obj2, t, ephem):
    """Compute the position and velocity for solar system obj2 referenced at obj1.
    This function uses astropy solar system Ephemerides module.
    Parameters
    ----------
    obj1: str
        The name of reference solar system object
    obj2: str
        The name of target solar system object
    tdb: Astropy.time.Time object
        TDB time in Astropy.time.Time object format
    ephem: str
        The ephem to for computing solar system object position and velocity
    """
    if obj1.lower() == 'ssb' and obj2.lower() != 'ssb':
        obj2pv = objPosVel_wrt_SSB(obj2,t,ephem)
        return obj2pv
    elif obj2.lower() == 'ssb' and obj1.lower() != 'ssb':
        obj1pv = objPosVel_wrt_SSB(obj1,t,ephem)
        return -obj1pv
    elif obj2.lower() != 'ssb' and obj1.lower() != 'ssb':
        obj1pv = objPosVel_wrt_SSB(obj1,t,ephem)
        obj2pv = objPosVel_wrt_SSB(obj2,t,ephem)
        return obj2pv - obj1pv
    else:
        return PosVel(np.zeros((3,len(t)))*u.km, np.zeros((3,len(t)))*u.km/u.second)
