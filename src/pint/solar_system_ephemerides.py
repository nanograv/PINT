"""Solar system ephemeris downloading and setting support."""

import contextlib
import os
import pathlib
from typing import Optional, Union, Tuple
import struct
from sys import byteorder

import astropy.coordinates
import astropy.units as u
import numpy as np
from astropy.utils.data import download_file
from loguru import logger as log

import pint.config
from pint.utils import PosVel

import tempfile
from astropy.utils.data import download_file

__all__ = ["objPosVel_wrt_SSB", "get_tdb_tt_ephem_geocenter"]

ephemeris_mirrors = [
    # NOTE the JPL ftp site is disabled for our automatic builds. Instead,
    # we duplicated the JPL ftp site on the nanograv server.
    # Search nanograv server first, then the other two.
    # "https://data.nanograv.org/static/data/ephem/",
    "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/",
    # DE440 is here, officially
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/",
]

_inpop_url = "https://ftp.imcce.fr/pub/ephem/planets/"

# https://github.com/marcelhesselberth/Inpop/tree/main
_inpop_obj_code = {
    "mercury": 0,
    "venus": 1,
    "earth": 2,
    "mars": 3,
    "jupiter": 4,
    "saturn": 5,
    "uranus": 6,
    "neptune": 7,
    "pluto": 8,
    "moon": 9,
    "sun": 10,
    "ssb": 11,
    "earth-moon-barycenter": 12,
}

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

loaded_ephems = {}


def clear_loaded_ephem() -> None:
    """Clear the dictionary of pre-loaded ephemeris files, to allow fresh loading"""
    log.debug("Clearing loaded ephemerides")
    global loaded_ephems
    loaded_ephems = {}


def _load_kernel_link(ephem: str, link: Optional[str] = None) -> bool:
    """Load an ephemeris file from a URL

    Parameters
    ----------
    ephem : str
        Name of ephemeris (without ``bsp`` extension)
    link : str, Optional
        URL

    Returns
    -------
    result : bool
        True if loaded successfully

    Notes
    -----
    If ``link`` is None, will still search default mirror sites
    """
    if link == "":
        raise ValueError("Empty string is not a valid URL")

    mirrors = [f"{m}{ephem}.bsp" for m in ephemeris_mirrors]
    if link is not None:
        mirrors = [link] + mirrors
    astropy.coordinates.solar_system_ephemeris.set(
        download_file(mirrors[0], cache=True, sources=mirrors)
    )
    log.info(f"Set solar system ephemeris to {ephem} from download")
    return True


def _load_kernel_local(
    ephem: str, path: Union[str, pathlib.Path]
) -> Union[str, pathlib.Path]:
    """Load an ephemeris file from a URL

    Parameters
    ----------
    ephem : str
        Name of ephemeris (without ``bsp`` extension)
    path : str or pathlib.Path
        Path to file

    Returns
    -------
    loaded_ephemeris : str or pathlib.Path

    Notes
    -----
    Will also search default PINT runtime data location in ``pint.config``
    """
    ephem_bsp = f"{ephem}.bsp"
    custom_path = os.path.join(path, ephem_bsp) if os.path.isdir(path) else path
    search_list = [custom_path]
    with contextlib.suppress(FileNotFoundError):
        search_list.append(pint.config.runtimefile(ephem_bsp))
    for p in search_list:
        if os.path.exists(p):
            # .set() can accept a path to an ephemeris
            astropy.coordinates.solar_system_ephemeris.set(p)
            log.info(f"Set solar system ephemeris to local file:\n\t{p}")
            return p
    raise FileNotFoundError(f"ephemeris file {ephem} not found in any of {search_list}")


def _inpopfile(ephem: str, short: Optional[bool] = True) -> str:
    """Return the full filename for the INPOP ephemeris

    Can return the short version (+/-100y) or long (+/-1000y)

    Parameters
    ----------
    ephem: str
        Name of inpop model
    short: bool, optional
        Whether or not to return the short version

    Returns
    -------
    str
    """
    if short:
        return f"{ephem.lower()}_TDB_m100_p100_tt.dat"
    return f"{ephem.lower()}_TDB_m1000_p1000_tt.dat"


def _inpopurl(ephem: str, short: Optional[bool] = True) -> str:
    """Return the URL for the INPOP ephemeris

    Can return the short version (+/-100y) or long (+/-1000y)

    Parameters
    ----------
    ephem: str
        Name of inpop model
    short: bool, optional
        Whether or not to return the short version

    Returns
    -------
    str
    """
    return f"{_inpop_url}{ephem.lower()}/{_inpopfile(ephem,short=short)}"


def _inpop_code_to_int(object: Union[int, np.integer, str]) -> int:
    """Return the code associated with a given object for INPOP

    Parameters
    ----------
    object : int or np.integer or str

    Returns
    -------
    int
    """
    if not isinstance(object, (int, np.integer)):
        if object.lower() in _inpop_obj_code:
            return _inpop_obj_code[object.lower()]
        raise KeyError(
            f"'{object}' is not recognized; potential objects are: [{_inpop_obj_code.keys().join(',')}]"
        )
    if object < 0 or object > 12:
        raise ValueError("Object code must be between 0 and 12")
    return object


class Inpop:

    def __init__(
        self,
        ephem: str,
        path: Optional[Union[str, pathlib.Path]] = None,
        short: Optional[bool] = True,
        cache: Optional[bool] = True,
        granulethreshold: int = 5,
    ):
        """
        Parameters
        ----------
        ephem: str
            Ephemeris version
        path : str, optional
            A location of the file on disk (will attempt to download if not supplied)
        short: bool, optional
            Whether to get the short (+/-100y) or long (+/-1000y) version
        cache: bool, optional
            Whether or not to cache the file (using astropy)
        granulethreshold: int, optional
            The threshold for the number of points in each granule (subset of interval)
            to treat using the binned method (faster when this number).
            If this is 0 will always use the binned method. If this is `np.inf` will always calculate by-item.

        Notes
        -----
        Uses :func:`astropy.utils.data.download_file` to download and cache the file.

        References
        ----------
        - Fienga et al. (2021), NSTIM, 110 [1]_

        .. [1] https://ui.adsabs.harvard.edu/abs/2021NSTIM.110.....F/abstract

        """

        # Parse the INPOP header
        # this is basically copied from
        # https://github.com/marcelhesselberth/Inpop/blob/main/inpop/inpop.py
        # there is a worry that the downloaded file will have the opposite endian-ness
        # so check and swap if needed
        if byteorder == "little":
            self.machine_byteorder = "<"
            self.opposite_byteorder = ">"
        else:
            self.machine_byteorder = ">"
            self.opposite_byteorder = "<"
        self.byteorder = self.machine_byteorder

        self.ephem = ephem
        self.granulethreshold = granulethreshold
        # use astropy caching for the download
        self.filename = (
            path
            if path is not None
            else download_file(_inpopurl(ephem, short=short), cache=cache)
        )
        self.f = open(self.filename, "rb")

        # Decode the header record
        header_spec = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
        header_struct = struct.Struct(header_spec)
        bytestr = self.f.read(header_struct.size)
        hb = header_struct.unpack(bytestr)  # header block
        self.DENUM = hb[44]  # Must be 100 for INPOP
        if self.DENUM != 100:
            self.f.seek(0)
            self.byteorder = self.opposite_byteorder
            header_spec = f"{self.byteorder}252s2400sdddidd36ii3ii3i"
            header_struct = struct.Struct(header_spec)
            bytestr = self.f.read(header_struct.size)
            hb = header_struct.unpack(bytestr)  # header block
            self.DENUM = hb[44]
            if self.DENUM != 100:
                raise (IOError("Can't determine INPOP file byteorder."))

        self.jd_struct = struct.Struct(f"{self.byteorder}dd")  # julian dates

        self.label = []  # Ephemeris label, list of 3 strings
        self.label.append(hb[0][:84].decode().strip())
        self.label.append(hb[0][84:168].decode().strip())
        self.label.append(hb[0][168:].decode().strip())

        const_names = [hb[1][6 * i : 6 * (i + 1)] for i in range(400)]

        self.jd_beg = hb[2]  # Julian start date
        self.jd_end = hb[3]  # Julian end date
        self.interval = hb[4]  # Julian interval
        self.num_const = hb[5]  # Number of constants in the second record
        self.AU = hb[6]  # Astronomical unit
        self.EMRAT = hb[7]  # Mearth / Mmoon
        self.coeff_ptr = [(hb[8 + 3 * i : 8 + 3 * i + 3]) for i in range(12)]
        self.DENUM = hb[44]  # Ephemeris ID
        self.librat_ptr = hb[45:48]  # Libration pointer
        self.recordsize = hb[48]  # Size of the record in bytes
        self.TTmTDB_ptr = hb[49:52]  # Time transformation TTmTDB or TCGmTCB

        # Location, number of coefficients and number of granules
        # for the 12 bodies.
        self.coeff_ptr = np.array(self.coeff_ptr, dtype=int)

        # Location, number of coefficients and number of granules
        # for the libration angles of the moon.
        self.librat_ptr = np.array(self.librat_ptr, dtype=int)

        # Location, number of coefficients and number of granules
        # for the mapping of  TT-TDB or TCG-TCB
        self.TTmTDB_ptr = np.array(self.TTmTDB_ptr, dtype=int)

        # Decode the constant record
        self.f.seek(self.recordsize * 8)
        const_struct = struct.Struct(f"{self.byteorder}%id" % (self.num_const))
        bytestr = self.f.read(const_struct.size)
        cb = const_struct.unpack(bytestr)
        const_names = const_names[: self.num_const]
        self.constants = {
            const_names[i].strip().decode(): cb[i] for i in range(self.num_const)
        }

        self.version = self.constants["VERSIO"]  # ephemerus version number
        self.fversion = self.constants["FVERSI"]  # file version number (0)
        self.format = self.constants["FORMAT"]  # details about file contents
        self.ksizer = int(self.constants["KSIZER"])  # numbers per record

        # Decode file format
        self.has_vel = (self.format // 1 % 10) == 0
        self.has_time = (self.format // 10) % 10 == 1
        self.has_asteroids = (self.format // 100) % 10 == 1

        # Use the following unit base and transform where necessary
        self.unit_time = "s"
        self.unit_angle = "rad"
        self.unit_pos = "au"
        self.unit_vel = "au/day"

        self.rate_factor = 2.0 / self.interval  # chain rule
        if self.constants["UNITE"] == 0:
            self.unite = 0
            self.unit_factor = 1.0
        else:
            self.unite = 1
            self.unit_factor = 1.0 / self.AU

        # If no timescale is found it is TDB (file version 1.0)
        if "TIMESC" in self.constants:
            if self.constants["TIMESC"] == 0:
                self.timescale = "TDB"
            else:
                self.timescale = "TCB"
        else:
            self.timescale = "TDB"

        self.nrecords = int((self.jd_end - self.jd_beg) / self.interval)

        self.earthfactor = -1 / (1 + self.EMRAT)
        self.moonfactor = self.EMRAT / (1 + self.EMRAT)

        # now load in the full data
        self.f.seek(0)
        self.data = np.fromfile(self.f, dtype=np.double)
        if self.byteorder != self.machine_byteorder:
            data = data.byteswap()

        # Load in the Chebyshev coefficients for each object
        # first TT-TDB
        self.CX_TTmTDB, self.CY_TTmTDB, self.CZ_TTmTDB = self.get_coeffs(
            *self.TTmTDB_ptr
        )
        # now the planets
        self.CX = []
        self.CY = []
        self.CZ = []
        for obj in range(self.coeff_ptr.shape[0]):
            CX, CY, CZ = self.get_coeffs(*self.coeff_ptr[obj])
            self.CX.append(CX)
            self.CY.append(CY)
            self.CZ.append(CZ)

    def get_coeffs(
        self, offset: int, ncoeffs: int, ngranules: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Chebyshev coefficients for a given object from the data structure

        Parameters
        ----------
        offset: int
            Offset from the start of the data structure to the given object
        ncoeffs: int
            Number of coefficients for the given object
        ngranules: int
            Number of time granules per interval

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray
            Coefficients for X,Y,Z

        """
        CX = np.zeros(
            (int((self.jd_end - self.jd_beg) // self.interval) * ngranules, ncoeffs)
        )
        CY = np.zeros(
            (int((self.jd_end - self.jd_beg) // self.interval) * ngranules, ncoeffs)
        )
        CZ = np.zeros(
            (int((self.jd_end - self.jd_beg) // self.interval) * ngranules, ncoeffs)
        )
        j = 0
        for record in np.arange(int((self.jd_end - self.jd_beg) // self.interval)):
            raddr = (record + 2) * self.recordsize
            for granule in np.arange(ngranules):
                gaddr = int(raddr + (offset - 1 + 3 * granule * ncoeffs))
                CX[j] = self.data[gaddr : gaddr + ncoeffs]
                CY[j] = self.data[gaddr + ncoeffs : gaddr + 2 * ncoeffs]
                CZ[j] = self.data[gaddr + 2 * ncoeffs : gaddr + 3 * ncoeffs]
                j += 1
        return CX, CY, CZ

    def t_to_item(
        self, t: "astropy.time.Time", ngranules: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do the indexing from an input time value to the appropriate interval/granule
        Also return the normalized time within that

        Parameters
        ----------
        t: astropy.time.Time
            (can be an array)
        ngranules: int
            Number of time granules per interval

        Returns
        -------
        item: np.ndarray
            Array of index values
        tc: np.ndarray
            Array of normalized time values [-1,1]
        """

        jd = t.jd1
        jd2 = t.jd2
        record = np.int64(((jd - self.jd_beg) + jd2) // self.interval)
        jdl = self.jd_beg + record * self.interval
        span = self.interval / ngranules
        granule = np.int64(((jd - jdl) + jd2) // span)
        jd0 = jdl + granule * span
        tc = 2 * (((jd - jd0) + jd2) / span) - 1
        return np.int64(record * self.interval // span + granule), tc

    def calc(
        self,
        t: "astropy.time.Time",
        object: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Do the baseline calculation for a given time/object.  Return the position vectors and the rate/velocity vectors

        Parameters
        ----------
        t: astropy.time.Time
            Can be array
        object: int
            Object index.  Should be -1 for TT to TDB, otherwise 0..12

        Returns
        -------
        PX,PY,PZ,VX,VY,VZ: np.ndarray
        """
        t = np.atleast_1d(t)
        if np.any(t.jd < self.jd_beg) or np.any(t.jd > self.jd_end):
            raise ValueError(
                f"All times must be between JD {self.jd_beg} and {self.jd_end}"
            )
        if object < -1 or object >= self.coeff_ptr.shape[0]:
            raise ValueError(
                f"Object pointer must be -1 or 0..{self.coeff_ptr.shape[0]}"
            )
        if object == -1:
            offset, ncoeffs, ngranules = self.TTmTDB_ptr
            CX = self.CX_TTmTDB
            CY = self.CY_TTmTDB
            CZ = self.CZ_TTmTDB
        else:
            offset, ncoeffs, ngranules = self.coeff_ptr[object]
            CX = self.CX[object]
            CY = self.CY[object]
            CZ = self.CZ[object]

        items, tcs = self.t_to_item(t, ngranules)
        PX = np.zeros(t.shape)
        PY = np.zeros(t.shape)
        PZ = np.zeros(t.shape)
        VX = np.zeros(t.shape)
        VY = np.zeros(t.shape)
        VZ = np.zeros(t.shape)

        # this threshold seems to work OK to go between the methods
        if len(items) / len(np.unique(items)) < self.granulethreshold:
            # this will work in ~constant time, but it's a little slow
            it = np.nditer(t.jd, flags=["multi_index"])
            for x in it:
                TX = np.polynomial.chebyshev.Chebyshev(CX[items[it.multi_index]])
                TY = np.polynomial.chebyshev.Chebyshev(CY[items[it.multi_index]])
                TZ = np.polynomial.chebyshev.Chebyshev(CZ[items[it.multi_index]])
                DX = TX.deriv(1)
                DY = TY.deriv(1)
                DZ = TZ.deriv(1)
                PX[it.multi_index] = TX(tcs[it.multi_index])
                PY[it.multi_index] = TY(tcs[it.multi_index])
                PZ[it.multi_index] = TZ(tcs[it.multi_index])
                VX[it.multi_index] = (
                    DX(tcs[it.multi_index]) * ngranules * self.rate_factor
                )
                VY[it.multi_index] = (
                    DY(tcs[it.multi_index]) * ngranules * self.rate_factor
                )
                VZ[it.multi_index] = (
                    DZ(tcs[it.multi_index]) * ngranules * self.rate_factor
                )
        else:
            # this will work faster when there are many items from the same granule
            for item in np.unique(items):
                index = items == item
                tc = tcs[index]
                TX = np.polynomial.chebyshev.Chebyshev(CX[item])
                TY = np.polynomial.chebyshev.Chebyshev(CY[item])
                TZ = np.polynomial.chebyshev.Chebyshev(CZ[item])
                DX = TX.deriv(1)
                DY = TY.deriv(1)
                DZ = TZ.deriv(1)
                PX[index] = TX(tc)
                PY[index] = TY(tc)
                PZ[index] = TZ(tc)
                VX[index] = DX(tc) * ngranules * self.rate_factor
                VY[index] = DY(tc) * ngranules * self.rate_factor
                VZ[index] = DZ(tc) * ngranules * self.rate_factor
        return (PX, PY, PZ, VX, VY, VZ)

    def TTmTDB(
        self, t: "astropy.time.Time", rate: Optional[bool] = False
    ) -> Tuple[u.Quantity]:
        """Return TT-TDB for a given time, optionally the rate in s/s.
        Note that this is computed for the TT timescale as input.

        Parameters
        ----------
        t: astropy.time.Time
            Can be array
        rate: optional, bool

        Returns
        -------
        TT-TDB: u.Quantity
        d(TT-TDB)/dt: u.Quantity
            Optional
        """
        if self.timescale == "TDB" and self.has_time:
            PX, PY, PZ, VX, VY, VZ = self.calc(t.tt, -1)
            return (PX * u.s) if not rate else (PX * u.s, VX * u.s / u.s)

    def PV(
        self,
        t: "astropy.time.Time",
        target: Union[int, np.integer, str],
        center: Union[int, np.integer, str],
        rate: Optional[bool] = False,
    ) -> Tuple[u.Quantity]:
        """Return the position vector X,Y,Z from center to target, optionally the velocity Vx,Vy,Vz
        Note that this is computed for the TDB timescale as input.

        Parameters
        ----------
        t: astropy.time.Time
            Can be array
        target: int or str
        center: int or str
        rate: optional, bool

        Returns
        -------
        P: u.Quantity
        V: u.Quantity
            Optional

        """
        target = _inpop_code_to_int(target)
        center = _inpop_code_to_int(center)
        gr_pos_factor = 1

        if target == 2:
            target_data = [
                x * self.earthfactor + y
                for (x, y) in zip(self.calc(t, 9), self.calc(t.tdb, 2))
            ]
        elif target == 9:
            target_data = [
                x * self.moonfactor + y
                for (x, y) in zip(self.calc(t, 9), self.calc(t.tdb, 2))
            ]
        elif target == 11:
            target_data = [np.zeros(t.shape)] * 6
        elif target == 12:
            target_data = self.calc(t.tdb, 2)
        else:
            target_data = self.calc(t.tdb, target)

        if center == 2:
            center_data = [
                x * self.earthfactor + y
                for (x, y) in zip(self.calc(t, 9), self.calc(t.tdb, 2))
            ]
        elif center == 9:
            center_data = [
                x * self.moonfactor + y
                for (x, y) in zip(self.calc(t, 9), self.calc(t.tdb, 2))
            ]
        elif center == 11:
            center_data = [np.zeros(t.shape)] * 6
        elif center == 12:
            center_data = self.calc(t.tdb, 2)
        else:
            center_data = self.calc(t.tdb, center)

        pos = (
            np.c_[target_data[0], target_data[1], target_data[2]]
            - np.c_[center_data[0], center_data[1], center_data[2]]
        )
        pos *= gr_pos_factor * self.unit_factor
        vel = (
            np.c_[target_data[3], target_data[4], target_data[5]]
            - np.c_[center_data[3], center_data[4], center_data[5]]
        )
        vel *= self.unit_factor
        return (
            (pos * u.Unit(self.unit_pos))
            if not rate
            else (pos * u.Unit(self.unit_pos), vel * u.Unit(self.unit_vel))
        )


def load_kernel(
    ephem: str,
    path: Optional[Union[str, pathlib.Path]] = None,
    link: Optional[str] = None,
) -> Union[str, pathlib.Path, bool]:
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
    path : str or pathlib.Path, optional
        Load the ephemeris from the file specified in path, rather than
        requesting it from the network or astropy's collection of
        ephemerides. The file is searched for by treating path as relative
        to the current directory, or failing that, as relative to the
        data directory specified in PINT's configuration.
    link : str, optional
        Suggest the URL as a possible location astropy should search
        for the ephemeris.

    Returns
    -------
    loaded_ephemeris : str or pathlib.Path or bool
        Can be str or pathlib.Path if loaded from a local file, or ``True``
        if loaded from URL


    Note
    ----
    If both ``path`` and ``link`` are provided, local path will be tried first.

    If ``path`` is not provided, will still search default mirror sites.

    Any local loaded ephemeris will be stored so it will not be re-requested.
    """
    ephem = ephem.lower()
    if ephem in loaded_ephems:
        log.debug(f"Using pre-loaded kernel for {ephem}: {loaded_ephems[ephem]}")
        return loaded_ephems[ephem]
    # If a local path is provided, the local search will be considered first.
    if path is not None:
        try:
            loaded_ephems[ephem] = _load_kernel_local(ephem, path=path)
            return loaded_ephems[ephem]
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
        return True
    # If this raises an exception our last hope is gone so let it propagate
    _load_kernel_link(ephem, link=link)
    return True


def objPosVel_wrt_SSB(
    objname: str,
    t: astropy.time.Time,
    ephem: str,
    path: Optional[Union[str, pathlib.Path]] = None,
    link: Optional[str] = None,
) -> PosVel:
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
        The ephem to for computing solar system object position and velocity (without bsp extension)
    path : str or pathlib.Path, optional
        Local path to the ephemeris file.
    link : str, optional
        Location of path on the internet.

    Returns
    -------
    PosVel object with 3-vectors for the position and velocity of the object
    """
    objname = objname.lower()
    if ephem.upper().startswith("DE"):
        load_kernel(ephem, path=path, link=link)
        pos, vel = astropy.coordinates.get_body_barycentric_posvel(objname, t)
        return PosVel(pos.xyz, vel.xyz.to(u.km / u.second), origin="ssb", obj=objname)
    elif ephem.upper().startswith("INPOP"):
        inpop_obj = Inpop(ephem, path=path)
        pos_inpop, vel_inpop = inpop_obj.PV(
            t, _inpop_obj_code[objname], _inpop_obj_code["ssb"], rate=True
        )
        return PosVel(
            (pos_inpop.T).to(u.km),
            (vel_inpop.T).to(u.km / u.s),
            origin="ssb",
            obj=objname,
        )


def objPosVel(
    obj1: str,
    obj2: str,
    t: astropy.time.Time,
    ephem: str,
    path: Optional[Union[str, pathlib.Path]] = None,
    link: Optional[str] = None,
) -> PosVel:
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
        The ephem to for computing solar system object position and velocity (without bsp extension)
    path : str or pathlib.Path, optional
        Local path to the ephemeris file.
    link : str, optional
        Location of path on the internet.

    Return
    ------
    PosVel object.
        solar system obj1's position and velocity with respect to obj2 in the
        J2000 cartesian coordinate.
    """
    if ephem.upper().startswith("DE"):
        if obj1.lower() == "ssb" and obj2.lower() != "ssb":
            return objPosVel_wrt_SSB(obj2, t, ephem, path=path, link=link)
        elif obj2.lower() == "ssb" and obj1.lower() != "ssb":
            obj1pv = objPosVel_wrt_SSB(obj1, t, ephem, path=path, link=link)
            return -obj1pv
        elif obj2.lower() != "ssb":
            obj1pv = objPosVel_wrt_SSB(obj1, t, ephem, path=path, link=link)
            obj2pv = objPosVel_wrt_SSB(obj2, t, ephem, path=path, link=link)
            return obj2pv - obj1pv
        else:
            # user asked for velocity between ssb and ssb
            return PosVel(
                np.zeros((3, len(t))) * u.km, np.zeros((3, len(t))) * u.km / u.second
            )
    elif ephem.upper().startswith("INPOP"):
        inpop_obj = Inpop(ephem)
        pos_inpop, vel_inpop = inpop_obj.PV(
            t, _inpop_obj_code[obj1], _inpop_obj_code[obj2], rate=True
        )
        return PosVel(
            (pos_inpop.T).to(u.km),
            (vel_inpop.T).to(u.km / u.s),
        )


def get_tdb_tt_ephem_geocenter(
    tt: astropy.time.Time,
    ephem: str,
    path: Optional[Union[str, pathlib.Path]] = None,
    link: Optional[str] = None,
) -> u.Quantity:
    """The is a function to read the TDB_TT correction from the JPL DExxxt.bsp
    ephemeris file.

    Parameters
    ----------
    tt: Astropy.time.Time object
        Observation time in Astropy.time.Time object format.
    ephem: str
        The ephem to for computing solar system object position and velocity (without bsp extension)
    path : str or pathlib.Path, optional
        Local path to the ephemeris file.
    link : str, optional
        Location of path on the internet.

    Returns
    -------
    tdb_tt_correction : u.Quantity

    Note
    ----
    Only the DEXXXt.bsp type ephemeris has the TDB-TT information, others do
    not provide it. The definition for TDB-TT column is described in the
    paper:
    https://ipnpr.jpl.nasa.gov/progress_report/42-196/196C.pdf page 6.
    """
    if ephem.upper().startswith("DE"):
        load_kernel(ephem, path=path, link=link)
        kernel = astropy.coordinates.solar_system_ephemeris._kernel
        try:
            # JPL ID defines this column.
            seg = kernel[1000000000, 1000000001]
        except KeyError:
            raise ValueError("Ephemeris '%s.bsp' do not provide the TDB-TT correction.")
        tdb_tt = seg.compute(tt.jd1, tt.jd2)[0]
        return tdb_tt * u.second
    elif ephem.upper().startswith("INPOP"):
        inpop_obj = Inpop(ephem)
        tdb_tt = inpop_obj.TTmTDB(tt)
        return tdb_tt
