"""Solar system ephemeris downloading and setting support."""

import os

import astropy.coordinates
import astropy.units as u
import contextlib
import numpy as np
from astropy.utils.data import download_file
from loguru import logger as log

import pint.config
from pint.utils import PosVel

__all__ = ["objPosVel_wrt_SSB", "get_tdb_tt_ephem_geocenter"]

ephemeris_mirrors = [
    # NOTE the JPL ftp site is disabled for our automatic builds. Instead,
    # we duplicated the JPL ftp site on the nanograv server.
    # Search nanograv server first, then the other two.
    "https://data.nanograv.org/static/data/ephem/",
    "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/",
    # DE440 is here, officially
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/",
]

jpl_obj_code = {
    "ssb": 0,
    "sun": 10,
    "mercury": 199,
    "venus": 299,
    "earth-moon-barycenter": 3,
    "earth": 399,
    "moon": 301,
    "mars": 499,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
    "pluto": 9,
}


def _load_kernel_link(ephem, link=None):
    if link == "":
        raise ValueError("Empty string is not a valid URL")

    mirrors = [f"{m}{ephem}.bsp" for m in ephemeris_mirrors]
    if link is not None:
        mirrors = [link] + mirrors
    astropy.coordinates.solar_system_ephemeris.set(
        download_file(mirrors[0], cache=True, sources=mirrors)
    )
    log.info(f"Set solar system ephemeris to {ephem} from download")


def _load_kernel_local(ephem, path):
    ephem_bsp = f"{ephem}.bsp"
    custom_path = os.path.join(path, ephem_bsp) if os.path.isdir(path) else path
    search_list = [custom_path]
    with contextlib.suppress(FileNotFoundError):
        search_list.append(pint.config.runtimefile(ephem_bsp))
    for p in search_list:
        if os.path.exists(p):
            # .set() can accept a path to an ephemeris
            astropy.coordinates.solar_system_ephemeris.set(ephem)
            log.info(f"Set solar system ephemeris to local file:\n\t{ephem}")
            return
    raise FileNotFoundError(f"ephemeris file {ephem} not found in any of {search_list}")


def load_kernel(ephem, path=None, link=None):
    """Load the solar system ephemeris

    Ephemeris files may be obtained through astropy's internal
    collection (which primarily downloads them from the network
    but caches them in a user-wide cache directory), from an
    additional network location via the astropy mechanism,
    or from a file on the local system.  If the ephemeris cannot
    be found a ValueError is raised.

    If a kernel must be obtained from the network, it is first looked
    for in the location specified by ``link``, then in a list of mirrors
    of the JPL ephemeris collection.

    If the ephemeris must be downloaded, it is downloaded using
    :func:`astropy.utils.data.download_file`; it is thus stored
    in the `Astropy cache <https://docs.astropy.org/en/stable/utils/data.html>`.

    Parameters
    ----------
    ephem : str
        Short name of the ephemeris, for example ``de421``. Case-insensitive.
    path : str, optional
        Load the ephemeris from the file specified in path, rather than
        requesting it from the network or astropy's collection of
        ephemerides. The file is searched for by treating path as relative
        to the current directory, or failing that, as relative to the
        data directory specified in PINT's configuration.
    link : str, optional
        Suggest the URL as a possible location astropy should search
        for the ephemeris.


    Note
    ----
    If both path and link are provided. Path will be first to try.
    """
    ephem = ephem.lower()
    # If a local path is provided, the local search will be considered first.
    if path is not None:
        try:
            _load_kernel_local(ephem, path=path)
            return
        except OSError:
            log.info(
                f"Failed to load local solar system ephemeris kernel {path}, falling back on astropy"
            )
    # Links are just suggestions, try just plain loading
    # Astropy may download something here, not from nanograv
    # Exception here means it wasn't a standard astropy ephemeris
    # or astropy can't access it (because astropy doesn't know about
    # the nanograv mirrors)
    with contextlib.suppress(ValueError, OSError):
        astropy.coordinates.solar_system_ephemeris.set(ephem)
        log.info(f"Set solar system ephemeris to {ephem} through astropy")
        return
    # If this raises an exception our last hope is gone so let it propagate
    _load_kernel_link(ephem, link=link)


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
    path : str, optional
        Local path to the ephemeris file.
    link : str, optional
        Location of path on the internet.

    Returns
    -------
    PosVel object with 3-vectors for the position and velocity of the object
    """
    objname = objname.lower()

    load_kernel(ephem, path=path, link=link)
    pos, vel = astropy.coordinates.get_body_barycentric_posvel(objname, t)
    return PosVel(pos.xyz, vel.xyz.to(u.km / u.second), origin="ssb", obj=objname)


def objPosVel(obj1, obj2, t, ephem, path=None, link=None):
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
    path : str, optional
        Local path to the ephemeris file.
    link : str, optional
        Location of path on the internet.

    Return
    ------
    PosVel object.
        solar system obj1's position and velocity with respect to obj2 in the
        J2000 cartesian coordinate.
    """
    if obj1.lower() == "ssb" and obj2.lower() != "ssb":
        return objPosVel_wrt_SSB(obj2, t, ephem, path=path, link=link)
    elif obj2.lower() == "ssb" and obj1.lower() != "ssb":
        obj1pv = objPosVel_wrt_SSB(obj1, t, ephem, path=path, link=link)
        return -obj1pv
    elif obj2.lower() != "ssb" and obj1.lower() != "ssb":
        obj1pv = objPosVel_wrt_SSB(obj1, t, ephem, path=path, link=link)
        obj2pv = objPosVel_wrt_SSB(obj2, t, ephem, path=path, link=link)
        return obj2pv - obj1pv
    else:
        # user asked for velocity between ssb and ssb
        return PosVel(
            np.zeros((3, len(t))) * u.km, np.zeros((3, len(t))) * u.km / u.second
        )


def get_tdb_tt_ephem_geocenter(tt, ephem, path=None, link=None):
    """The is a function to read the TDB_TT correction from the JPL DExxxt.bsp
    ephemeris file.

    Parameters
    ----------
    t: Astropy.time.Time object
        Observation time in Astropy.time.Time object format.
    ephem: str
        Ephemeris name.
    path : str, optional
        Local path to the ephemeris file.
    link : str, optional
        Location of path on the internet.

    Note
    ----
    Only the DEXXXt.bsp type ephemeris has the TDB-TT information, others do
    not provide it. The definition for TDB-TT column is described in the
    paper:
    https://ipnpr.jpl.nasa.gov/progress_report/42-196/196C.pdf page 6.
    """
    load_kernel(ephem, path=path, link=link)
    kernel = astropy.coordinates.solar_system_ephemeris._kernel
    try:
        # JPL ID defines this column.
        seg = kernel[1000000000, 1000000001]
    except KeyError:
        raise ValueError("Ephemeris '%s.bsp' do not provide the TDB-TT correction.")
    tdb_tt = seg.compute(tt.jd1, tt.jd2)[0]
    return tdb_tt * u.second
