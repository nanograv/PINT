import numpy as np
import os
import astropy.units as u
import astropy.coordinates as coor
from astropy.extern.six.moves import urllib
from astropy.time import Time
from .utils import PosVel
from astropy import log
import astropy.utils as aut
from jplephem.spk import SPK
from .config import datapath
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY

jpl_kernel_http = 'http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/'
jpl_kernel_ftp = 'ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/'

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


def _load_kernel_link(ephem, link=''):
    load_kernel = False # a flag for checking if the kernel has been loaded
    search_list = [link, jpl_kernel_http, jpl_kernel_ftp]
    if link != '':
        search_list.append('')
    for l in search_list:
        if l == '':
            ephem_link = ephem # Astropy default ephem does not like .bsp in the end.
        else:
            ephem_link = l+"%s.bsp" % ephem
        if load_kernel:
            break
        try:
            coor.solar_system_ephemeris.set(ephem_link)
            load_kernel = True
        except:
            try:
                aut.data.download_file(ephem_link, timeout=50, cache=True)
                coor.solar_system_ephemeris.set(ephem_link)
                load_kernel = True
            except:
                load_kernel = False
    return load_kernel


def _load_kernel_local(ephem, path=''):
    load_kernel = False # a flag for checking if the kernel has been loaded
    if path.endswith("%s.bsp" % ephem):
        custom_path = path
    else:
        custom_path = os.path.join(path, "%s.bsp" % ephem)
    search_list = [custom_path, datapath("%s.bsp" % ephem)]
    for p in search_list:
        if load_kernel:
            break
        try:# Bipass the astropy kernel loading system.
            coor.solar_system_ephemeris._kernel = SPK.open(p)
            coor.solar_system_ephemeris._value = p
            coor.solar_system_ephemeris._kernel.origin = coor.solar_system_ephemeris._value
            load_kernel = True
        except:
            load_kernel = False
    return load_kernel


def objPosVel_wrt_SSB(objname, t, ephem, path=None, link=None):
    """This function computes a solar system object position and velocity respect
    to solar system barycenter using astropy coordinates get_body_barycentric()
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
    path: str optional
        The data directory point to a local ephemeris.
    link: str optional
        The link where to download the ephemeris.

    Returns
    -------
    PosVel object with 3-vectors for the position and velocity of the object

    Note
    ----
    If both path and link are provided. Path will be first to try.
    """
    ephem = ephem.lower()
    objname = objname.lower()
    # Use astropy to compute postion.
    # If a local path is provided, the local search will be considered first.
    # If the path is not provided, try link first, then local data file.
    if path is None:
        if link is None:
            link_str = ''
        else:
            link_str = link
        is_load = _load_kernel_link(ephem, link=link_str)
        if not is_load: # Link does not Try to load from data path.
            path_str = ''
            is_load = _load_kernel_local(ephem, path=path_str)
        if not is_load:
            raise ValueError("Can not load the ephemeris file '%s.bsp'. " % ephem)
    else:
        path_str = path
        is_load = _load_kernel_local(ephem, path=path_str)
        if not is_load:
            raise ValueError("Can not load the ephemeris file '%s.bsp' from the"
                             " local directory %s." % (ephem, path_str))
    pos, vel = coor.get_body_barycentric_posvel(objname, t)
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
    Return
    ------
    PosVel object.
        solar system obj1's position and velocity with respect to obj2 in the
        J2000 cartesian coordinate.
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
