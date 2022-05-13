"""Special locations that are not really observatories.

Special "site" locations (eg, barycenter) which do not need clock
corrections or much else done.
"""
import os

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import EarthLocation
import numpy as np
from loguru import logger as log

import pint.config
from pint.observatory import bipm_default
from pint.observatory.clock_file import ClockFile
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel
from . import Observatory


class SpecialLocation(Observatory):
    """Special locations that are not really observatories.

    Observatory-derived class for special sites that are not really
    observatories but sometimes are used as TOA locations (eg, solar
    system barycenter).  Currently the only feature of this class is
    that clock corrections are zero.

    Parameters
    ----------
    name : string
        The name of the observatory
    aliases : str, optional
        List of other aliases for the observatory name.
    include_gps : bool, optional
        Set False to disable UTC(GPS)->UTC clock correction.
    include_bipm : bool, optional
        Set False to disable UTC-> TT BIPM clock
        correction. If False, it only apply TAI->TT correction
        TT = TAI+32.184s, the same as TEMPO2 TT(TAI) in the
        parfile. If True, it will apply the correction from
        BIPM TT=TT(BIPMYYYY). See the link:
        https://www.bipm.org/en/bipm-services/timescales/time-ftp/ttbipm.html
    bipm_version : str, optional
        Set the version of TT BIPM clock correction file to
        use, the default is %s.  It has to be in the format
        like 'BIPM2015'
    """ % bipm_default

    def __init__(
        self,
        name,
        aliases=None,
        include_gps=True,
        include_bipm=True,
        bipm_version=bipm_default,
        overwrite=False,
    ):
        # GPS corrections not implemented yet
        self.include_gps = include_gps
        self._gps_clock = None

        # BIPM corrections not implemented yet
        self.include_bipm = include_bipm
        self.bipm_version = bipm_version
        self._bipm_clock = None

        self.origin = "Built-in special location."

        super().__init__(name, aliases=aliases)

    @property
    def gps_fullpath(self):
        """Returns full path to the GPS-UTC clock file.  Will first try PINT
        data dirs, then fall back on $TEMPO2/clock."""
        fname = "gps2utc.clk"
        try:
            fullpath = pint.config.runtimefile(fname)
            return fullpath
        except FileNotFoundError:
            log.info(
                "{} not found in PINT data dirs, falling back on TEMPO2/clock directory".format(
                    fname
                )
            )
            return os.path.join(os.getenv("TEMPO2"), "clock", fname)

    @property
    def bipm_fullpath(
        self,
    ):
        """Returns full path to the TAI TT(BIPM) clock file.  Will first try PINT
        data dirs, then fall back on $TEMPO2/clock."""
        fname = "tai2tt_" + self.bipm_version.lower() + ".clk"
        try:
            fullpath = pint.config.runtimefile(fname)
            return fullpath
        except FileNotFoundError:
            pass
        log.info(
            "{} not found in PINT data dirs, falling back on TEMPO2/clock directory".format(
                fname
            )
        )
        return os.path.join(os.getenv("TEMPO2"), "clock", fname)

    def _load_gps_clock(self):
        if self._gps_clock is None:
            log.info(
                "Observatory {0}, loading GPS clock file {1}".format(
                    self.name, self.gps_fullpath
                )
            )
            self._gps_clock = ClockFile.read(self.gps_fullpath, format="tempo2")

    def _load_bipm_clock(self):
        if self._bipm_clock is None:
            try:
                log.info(
                    "Observatory {0}, loading BIPM clock file {1}".format(
                        self.name, self.bipm_fullpath
                    )
                )
                self._bipm_clock = ClockFile.read(self.bipm_fullpath, format="tempo2")
            except:
                raise ValueError("Can not find TT BIPM file '%s'. " % self.bipm_version)

    def clock_corrections(self, t, limits="warn"):
        corr = np.zeros(t.shape) * u.s
        if self.include_gps:
            log.info("Applying GPS to UTC clock correction (~few nanoseconds)")
            self._load_gps_clock()
            corr += self._gps_clock.evaluate(t, limits=limits)
        if self.include_bipm:
            log.info("Applying TT(TAI) to TT(BIPM) clock correction (~27 us)")
            self._load_bipm_clock()
            tt2tai = 32.184 * 1e6 * u.us
            corr += self._bipm_clock.evaluate(t, limits=limits) - tt2tai
        return corr

    def last_clock_correction_mjd(self):
        """Return the MJD of the last available clock correction.

        Returns ``np.inf`` if no clock corrections are relevant.
        """
        t = np.inf
        if self.include_gps:
            self._load_gps_clock()
            t = min(t, self._gps_clock.last_correction_mjd())
        if self.include_bipm:
            self._load_bipm_clock()
            t = min(t, self._bipm_clock.last_correction_mjd())
        return t


class BarycenterObs(SpecialLocation):
    """Observatory-derived class for the solar system barycenter.  Time
    scale is assumed to be tdb."""

    @property
    def timescale(self):
        return "tdb"

    @property
    def tempo_code(self):
        return "@"

    @property
    def tempo2_code(self):
        return "bat"

    def get_gcrs(self, t, ephem=None):
        if ephem is None:
            raise ValueError("Ephemeris needed for BarycenterObs get_gcrs")
        ssb_pv = objPosVel_wrt_SSB("earth", t, ephem)
        return -1 * ssb_pv.pos

    def posvel(self, t, ephem, group=None):
        vdim = (3,) + t.shape
        return PosVel(
            np.zeros(vdim) * u.m,
            np.zeros(vdim) * u.m / u.s,
            obj=self.name,
            origin="ssb",
        )

    def clock_corrections(self, t, limits="warn"):
        log.info("Special observatory location. No clock corrections applied.")
        return np.zeros(t.shape) * u.s

    def last_clock_correction_mjd(self):
        return np.inf


class GeocenterObs(SpecialLocation):
    """Observatory-derived class for the Earth geocenter."""

    @property
    def timescale(self):
        return "utc"

    def earth_location_itrf(self, time=None):
        return EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)

    @property
    def tempo_code(self):
        return "0"

    @property
    def tempo2_code(self):
        return "coe"

    def get_gcrs(self, t, ephem=None):
        vdim = (3,) + t.shape
        return np.zeros(vdim) * u.m

    def posvel(self, t, ephem, group=None):
        return objPosVel_wrt_SSB("earth", t, ephem)


class T2SpacecraftObs(SpecialLocation):
    """An observatory with position tabulated following Tempo2 convention.

    In tempo2, it is possible to specify the GCRS position of the
    observatory via the -telx, -tely, and -telz flags in a TOA file.  This
    class is able to obtain its position in this way, i.e. by examining the
    flags in a TOA table.
    """

    @property
    def timescale(self):
        return "utc"

    @property
    def tempo_code(self):
        return None

    def get_gcrs(self, t, group, ephem=None):
        """Return spacecraft GCRS position; this assumes position flags in tim file are in km"""

        if group is None:
            raise ValueError("TOA group table needed for SpacecraftObs get_gcrs")

        try:
            x = np.array([float(flags["telx"]) for flags in group["flags"]])
            y = np.array([float(flags["tely"]) for flags in group["flags"]])
            z = np.array([float(flags["telz"]) for flags in group["flags"]])
        except:
            log.error(
                "Missing flag. TOA line should have telx,tely,telz flags for GCRS position in km."
            )
            raise ValueError(
                "Missing flag. TOA line should have telx,tely,telz flags for GCRS position in km."
            )

        pos = np.vstack((x, y, z))
        vdim = (3,) + t.shape
        if pos.shape != vdim:
            raise ValueError(
                "GCRS position vector has wrong shape: ",
                pos.shape,
                " instead of ",
                vdim.shape,
            )

        return pos * u.km

    def posvel_gcrs(self, t, group, ephem=None):
        """Return spacecraft GCRS position and velocity; this assumes position flags in tim file are in km and velocity flags are in km/s"""

        if group is None:
            raise ValueError("TOA group table needed for SpacecraftObs posvel_gcrs")

        try:
            vx = np.array([float(flags["vx"]) for flags in group["flags"]])
            vy = np.array([float(flags["vy"]) for flags in group["flags"]])
            vz = np.array([float(flags["vz"]) for flags in group["flags"]])
        except:
            log.error(
                "Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s."
            )
            raise ValueError(
                "Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s."
            )

        vel_geo = np.vstack((vx, vy, vz)) * (u.km / u.s)
        vdim = (3,) + t.shape
        if vel_geo.shape != vdim:
            raise ValueError(
                "GCRS velocity vector has wrong shape: ",
                vel_geo.shape,
                " instead of ",
                vdim.shape,
            )

        pos_geo = self.get_gcrs(t, group, ephem=None)

        stl_posvel = PosVel(pos_geo, vel_geo, origin="earth", obj="spacecraft")
        return stl_posvel

    def posvel(self, t, ephem, group=None):

        if group is None:
            raise ValueError("TOA group table needed for SpacecraftObs posvel")

        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)

        # Spacecraft posvel w.r.t. Earth
        stl_posvel = self.posvel_gcrs(t, group)

        # Vector add to geo_posvel to get full posvel vector w.r.t. SSB.
        return geo_posvel + stl_posvel


# Need to initialize one of each so that it gets added to the list
BarycenterObs("barycenter", aliases=["@", "ssb", "bary", "bat"])
GeocenterObs("geocenter", aliases=["0", "o", "coe", "geo"])
T2SpacecraftObs("stl_geo", aliases=["STL_GEO"])
# TODO -- BIPM issue
