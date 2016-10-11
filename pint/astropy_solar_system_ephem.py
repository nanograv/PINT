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

def objPosVel2SSB(objname, t, ephem):
    """This function computes a solar system object position and velocity respect
    to solar system barycenter using astropy coordinates get_body_barycentric()
    method.
    Parameters
    ----------
    objname: str
        Solar system object name. Current support solar system bodies are listed in
        astropy.coordinates.olar_system_ephemeris.bodies attribution.
    t: Astropy.time.Time object
        Observation time in Astropy.time.Time object format.
    ephem: str
        The ephem to for computing solar system object position and velocity
    """
    ephem = ephem.lower()
    objname = objname.lower()
    # Use astropy to compute postion.
    try:
        pos = coor.get_body_barycentric(objname, t, ephemeris=ephem)
    except ValueError:
        pos = coor.get_body_barycentric(objname, t, ephemeris= kernel_link_base + "%s.bsp" % ephem)
    # Use jplephem to compute velocity.
    # TODO: Astropy 1.3 will have velocity calculation availble.
    kernel = SPK.open(datapath("%s.bsp" % ephem))
    # Compute vel from planet barycenter to solar system barycenter
    lcod = len(str(jpl_obj_code[objname]))
    tjd1 = t.tdb.jd1
    tjd2 = t.tdb.jd2
    # Planets with barycenter computing
    if lcod == 3:
        _, vel_pbary_ssb = kernel[0,jpl_obj_code[objname]/100].compute_and_differentiate(tjd1, tjd2)
        _, vel_p_pbary = kernel[jpl_obj_code[objname]/100,
                        jpl_obj_code[objname]].compute_and_differentiate(tjd1, tjd2)
        vel = vel_pbary_ssb + vel_p_pbary
    # Planets without barycenter computing
    else:
         _, vel = kernel[0,jpl_obj_code[objname]].compute_and_differentiate(tjd1, tjd2)
    return PosVel(pos.xyz, vel / SECS_PER_DAY * u.km/u.second, origin='ssb', obj=objname)

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
        obj2pv = objPosVel2SSB(obj2,t,ephem)
        return obj2pv
    elif obj2.lower() == 'ssb' and obj1.lower() != 'ssb':
        obj1pv = objPosVel2SSB(obj1,t,ephem)
        return -obj1pv
    elif obj2.lower() != 'ssb' and obj1.lower() != 'ssb':
        obj1pv = objPosVel2SSB(obj1,t,ephem)
        obj2pv = objPosVel2SSB(obj2,t,ephem)
        return obj2pv - obj1pv
    else:
        return PosVel(np.zeros((3,len(t)))*u.km, np.zeros((3,len(t)))*u.km/u.second)
