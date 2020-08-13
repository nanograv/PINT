"""Special locations that are not really observatories.

Special "site" locations (eg, barycenter) which do not need clock
corrections or much else done.
"""
from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy
from astropy import log
from astropy.coordinates import EarthLocation

from pint.config import datapath
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
        use, the default is BIPM2015.  It has to be in the format
        like 'BIPM2015'
    """

    # def clock_corrections(self, t):
    #    log.info('Special observatory location. No clock corrections applied.')
    #    return numpy.zeros(t.shape)*u.s

    def __init__(
        self,
        name,
        aliases=None,
        include_gps=True,
        include_bipm=True,
        bipm_version="BIPM2015",
        tt2tdb_mode="pint",
    ):
        # GPS corrections not implemented yet
        self.include_gps = include_gps
        self._gps_clock = None

        # BIPM corrections not implemented yet
        self.include_bipm = include_bipm
        self.bipm_version = bipm_version
        self._bipm_clock = None

        super(SpecialLocation, self).__init__(
            name, aliases=aliases, tt2tdb_mode=tt2tdb_mode
        )

    @property
    def gps_fullpath(self):
        """Returns full path to the GPS-UTC clock file.  Will first try PINT
        data dirs, then fall back on $TEMPO2/clock."""
        fname = "gps2utc.clk"
        try:
            fullpath = datapath(fname)
            return fullpath
        except FileNotFoundError:
            log.info(
                "{} not found in PINT data dirs, falling back on TEMPO2/clock directory".format(
                    fname
                )
            )
            return os.path.join(os.getenv("TEMPO2"), "clock", fname)

    @property
    def bipm_fullpath(self,):
        """Returns full path to the TAI TT(BIPM) clock file.  Will first try PINT
        data dirs, then fall back on $TEMPO2/clock."""
        fname = "tai2tt_" + self.bipm_version.lower() + ".clk"
        try:
            fullpath = datapath(fname)
            return fullpath
        except FileNotFoundError:
            pass
        log.info(
            "{} not found in PINT data dirs, falling back on TEMPO2/clock directory".format(
                fname
            )
        )
        return os.path.join(os.getenv("TEMPO2"), "clock", fname)

    def clock_corrections(self, t):
        corr = numpy.zeros(t.shape) * u.s
        if self.include_gps:
            log.info("Applying GPS to UTC clock correction (~few nanoseconds)")
            if self._gps_clock is None:
                log.info(
                    "Observatory {0}, loading GPS clock file {1}".format(
                        self.name, self.gps_fullpath
                    )
                )
                self._gps_clock = ClockFile.read(self.gps_fullpath, format="tempo2")
            corr += self._gps_clock.evaluate(t)
        if self.include_bipm:
            log.info("Applying TT(TAI) to TT(BIPM) clock correction (~27 us)")
            tt2tai = 32.184 * 1e6 * u.us
            if self._bipm_clock is None:
                try:
                    log.info(
                        "Observatory {0}, loading BIPM clock file {1}".format(
                            self.name, self.bipm_fullpath
                        )
                    )
                    self._bipm_clock = ClockFile.read(
                        self.bipm_fullpath, format="tempo2"
                    )
                except:
                    raise ValueError(
                        "Can not find TT BIPM file '%s'. " % self.bipm_version
                    )
            corr += self._bipm_clock.evaluate(t) - tt2tai
        return corr


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

    def get_gcrs(self, t, ephem=None, grp=None):
        if ephem is None:
            raise ValueError("Ephemeris needed for BarycenterObs get_gcrs")
        ssb_pv = objPosVel_wrt_SSB("earth", t, ephem)
        return -1 * ssb_pv.pos

    def posvel(self, t, ephem):
        vdim = (3,) + t.shape
        return PosVel(
            numpy.zeros(vdim) * u.m,
            numpy.zeros(vdim) * u.m / u.s,
            obj=self.name,
            origin="ssb",
        )

    def clock_corrections(self, t):
        log.info("Special observatory location. No clock corrections applied.")
        return numpy.zeros(t.shape) * u.s


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

    def get_gcrs(self, t, ephem=None, grp=None):
        vdim = (3,) + t.shape
        return numpy.zeros(vdim) * u.m

    def posvel(self, t, ephem):
        return objPosVel_wrt_SSB("earth", t, ephem)


class SpacecraftObs(SpecialLocation):
    """Observatory-derived class for a spacecraft observatory."""

    @property
    def timescale(self):
        return "utc"

    @property
    def tempo_code(self):
        return None

    def get_gcrs(self, t, ephem=None, grp=None):
        """Return spacecraft GCRS position; this assumes position flags in tim file are in km"""

        if grp is None:
            raise ValueError("TOA group table needed for SpacecraftObs get_gcrs")

        try:
            # Is there a better way to do this?
            x = numpy.array([flags["telx"] for flags in grp["flags"]])
            y = numpy.array([flags["tely"] for flags in grp["flags"]])
            z = numpy.array([flags["telz"] for flags in grp["flags"]])
        except:
            log.error(
                "Missing flag. TOA line should have telx,tely,telz flags for GCRS position in km."
            )
            raise ValueError(
                "Missing flag. TOA line should have telx,tely,telz flags for GCRS position in km."
            )

        pos = numpy.vstack((x, y, z))
        vdim = (3,) + t.shape
        if pos.shape != vdim:
            raise ValueError(
                "GCRS position vector has wrong shape: ",
                pos.shape,
                " instead of ",
                vdim.shape,
            )

        return pos * u.km

    def posvel_gcrs(self, t, grp):
        """Return spacecraft GCRS position and velocity; this assumes position flags in tim file are in km and velocity flags are in km/s"""

        if grp is None:
            raise ValueError("TOA group table needed for SpacecraftObs posvel_gcrs")

        try:
            # Is there a better way to do this?
            vx = numpy.array([flags["vx"] for flags in grp["flags"]])
            vy = numpy.array([flags["vy"] for flags in grp["flags"]])
            vz = numpy.array([flags["vz"] for flags in grp["flags"]])
        except:
            log.error(
                "Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s."
            )
            raise ValueError(
                "Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s."
            )

        vel_geo = numpy.vstack((vx, vy, vz)) * (u.km / u.s)
        vdim = (3,) + t.shape
        if vel_geo.shape != vdim:
            raise ValueError(
                "GCRS velocity vector has wrong shape: ",
                vel.shape,
                " instead of ",
                vdim.shape,
            )

        pos_geo = self.get_gcrs(t, ephem=None, grp=grp)

        stl_posvel = PosVel(pos_geo, vel_geo, origin="earth", obj="spacecraft")
        return stl_posvel

    def posvel(self, t, ephem, grp):

        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)

        # Spacecraft posvel w.r.t. Earth
        stl_posvel = self.posvel_gcrs(t, grp)

        # Vector add to geo_posvel to get full posvel vector w.r.t. SSB.
        return geo_posvel + stl_posvel


# Need to initialize one of each so that it gets added to the list
BarycenterObs("barycenter", aliases=["@", "ssb", "bary", "bat"])
GeocenterObs("geocenter", aliases=["0", "o", "coe", "geo"])
SpacecraftObs("spacecraft", aliases=["STL_GEO"])
