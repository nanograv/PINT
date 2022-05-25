"""Ground-based fixed observatories."""
import os

import astropy.constants as c
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger as log

from pint import JD_MJD
from pint.config import runtimefile
from pint.erfautils import gcrs_posvel_from_itrf
from pint.observatory import Observatory, bipm_default
from pint.observatory.clock_file import ClockFile
from pint.pulsar_mjd import Time
from pint.solar_system_ephemerides import get_tdb_tt_ephem_geocenter, objPosVel_wrt_SSB
from pint.utils import has_astropy_unit

# These are global because they are, well, literally global
_gps_clock = None


class TopoObs(Observatory):
    """Observatories that are at a fixed location on the surface of the Earth.

    This behaves very similarly to "standard" site definitions in tempo/tempo2.  Clock
    correction files are read and computed, observatory coordinates are specified in
    ITRF XYZ, etc.

    In order for PINT to be able to actually find a clock file, you have several options:

    * Specify ``clock_file`` and ``clock_fmt=tempo2``
    * Specify ``clock_file`` and ``clock_fmt=tempo``
    * Specify ``clock_fmt=tempo2`` and ``tempo_code`` and have your clock file listed in ``time.dat`` with an ``INLCUDE`` statement

    If PINT cannot find a clock file, you will (by default) get a warning and no
    clock corrections. Calling code can request that missing clock corrections
    raise an exception.

    Parameters
    ----------

    name : str
        The name of the observatory
    itrf_xyz : astropy.units.Quantity or array-like
        IRTF site coordinates (len-3 array).  Can include
       astropy units.  If no units are given, meters are
       assumed.

    tempo_code : str, optional
        1-character tempo code for the site.  Will be
        automatically added to aliases.  Note, this is
        REQUIRED only if using TEMPO time.dat clock file.
    itoa_code : str, optional
        2-character ITOA code.  Will be added to aliases.
    aliases : list of str, optional
        List of other aliases for the observatory name.
    clock_file : str, optional
        Name of the clock correction file.
    clock_dir : str, optional
        Location of the clock file.  Special values
        'TEMPO', 'TEMPO2', or 'PINT' mean to use the
        standard directory for the package.  Otherwise
        can be set to a full path to the directory
        containing the clock_file.  Default='TEMPO'
    clock_fmt : str, optional
        Format of clock file (see ClockFile class for allowed
        values).
    include_gps : bool, optional
        Set False to disable UTC(GPS)->UTC clock correction.
    include_bipm : bool, optional
        Set False to disable UTC-> TT BIPM clock
        correction. If False, it only apply TAI->TT correction
        TT = TAI+32.184s, the same as TEMPO2 TT(TAI) in the
        parfile. If True, it will apply the correction from
        BIPM TT=TT(BIPMYYYY). See the link:
        http://www.bipm.org/en/bipm-services/timescales/time-ftp/ttbipm.html
    bipm_version : str, optional
        Set the version of TT BIPM clock correction file to
        use, the default is %s.  It has to be in the format
        like 'BIPM2015'
    origin : str, optional
        Documentation of the origin/author/date for the information
    overwrite : bool, optional
        set True to force overwriting of previous observatory definition
    bogus_last_correction : bool, optional
        Clock correction files include a bogus last correction
    """ % bipm_default

    def __init__(
        self,
        name,
        tempo_code=None,
        itoa_code=None,
        aliases=None,
        itrf_xyz=None,
        clock_file="",
        clock_dir="PINT",
        clock_fmt="tempo",
        include_gps=True,
        include_bipm=True,
        bipm_version=bipm_default,
        origin=None,
        overwrite=False,
        bogus_last_correction=False,
    ):
        # ITRF coordinates are required
        if itrf_xyz is None:
            raise ValueError("ITRF coordinates not given for observatory '%s'" % name)

        # Convert coords to standard format.  If no units are given, assume
        # meters.
        if not has_astropy_unit(itrf_xyz):
            xyz = np.array(itrf_xyz) * u.m
        else:
            xyz = itrf_xyz.to(u.m)

        # Check for correct array dims
        if xyz.shape != (3,):
            raise ValueError(
                "Incorrect coordinate dimensions for observatory '%s'" % (name)
            )

        # Convert to astropy EarthLocation, ensuring use of ITRF geocentric coordinates
        self._loc_itrf = EarthLocation.from_geocentric(*xyz)

        # Save clock file info, the data will be read only if clock
        # corrections for this site are requested.
        self.clock_file = clock_file
        self._multiple_clock_files = not isinstance(clock_file, str)
        self.clock_dir = clock_dir
        self.clock_fmt = clock_fmt
        self._clock = None  # The ClockFile object, will be read on demand

        # If using TEMPO time.dat we need to know the 1-char tempo-style
        # observatory code.
        if clock_fmt == "tempo" and clock_file == "time.dat" and tempo_code is None:
            raise ValueError("No tempo_code set for observatory '%s'" % name)

        # GPS corrections
        self.include_gps = include_gps

        # BIPM corrections
        self.include_bipm = include_bipm
        self.bipm_version = bipm_version
        self._bipm_clock = None
        self.bogus_last_correction = bogus_last_correction

        self.tempo_code = tempo_code
        if aliases is None:
            aliases = []
        for code in (tempo_code, itoa_code):
            if code is not None:
                aliases.append(code)

        self.origin = origin
        super(TopoObs, self).__init__(name, aliases=aliases)

    @property
    def clock_fullpath(self):
        """Returns the full path to the clock file."""
        if self.clock_dir == "PINT":
            if self._multiple_clock_files:
                return [(runtimefile(f) if f else "") for f in self.clock_file]
            return runtimefile(self.clock_file) if self.clock_file else ""
        elif self.clock_dir == "TEMPO":
            # Technically should read $TEMPO/tempo.cfg and get clock file
            # location from CLKDIR line...
            TEMPO_dir = os.getenv("TEMPO")
            if TEMPO_dir is None:
                log.error(
                    f"Cannot find TEMPO path from the enviroment; needed by observatory {self.name}."
                )
                dir = ""
            else:
                dir = os.path.join(TEMPO_dir, "clock")
        elif self.clock_dir == "TEMPO2":
            TEMPO2_dir = os.getenv("TEMPO2")
            if TEMPO2_dir is None:
                log.error(
                    f"Cannot find TEMPO2 path from the enviroment; needed by observatory {self.name}."
                )
                dir = ""
            else:
                dir = os.path.join(TEMPO2_dir, "clock")
        else:
            dir = self.clock_dir

        def join(dir, f):
            if f and dir:
                return os.path.join(dir, f)
            else:
                return ""

        if self._multiple_clock_files:
            return [join(dir, f) for f in self.clock_file]
        else:
            return join(dir, self.clock_file)

    @property
    def gps_fullpath(self):
        """Returns full path to the GPS-UTC clock file.  Will first try PINT
        data dirs, then fall back on $TEMPO2/clock."""
        fname = "gps2utc.clk"
        fullpath = runtimefile(fname)
        if fullpath is not None:
            return fullpath
        return os.path.join(os.getenv("TEMPO2"), "clock", fname)

    @property
    def bipm_fullpath(
        self,
    ):
        """Returns full path to the TAI TT(BIPM) clock file.

        Will first try PINT data dirs, then fall back on $TEMPO2/clock.
        """
        fname = "tai2tt_" + self.bipm_version.lower() + ".clk"
        try:
            fullpath = runtimefile(fname)
            return fullpath
        except FileNotFoundError:
            try:
                return os.path.join(os.getenv("TEMPO2"), "clock", fname)
            except OSError as e:
                if e.errno == 2:  # File not found
                    return None
                else:
                    raise

    @property
    def timescale(self):
        return "utc"

    def earth_location_itrf(self, time=None):
        return self._loc_itrf

    def _load_gps_clock(self):
        global _gps_clock
        if _gps_clock is None:
            log.info(f"Loading GPS clock file {self.gps_fullpath} for {self.name}")
            _gps_clock = ClockFile.read(
                self.gps_fullpath, format="tempo2", bogus_last_correction=True
            )

    def _load_bipm_clock(self):
        if self._bipm_clock is None:
            try:
                log.info("Loading BIPM clock file {self.bipm_fullpath} for {self.name}")
                self._bipm_clock = ClockFile.read(self.bipm_fullpath, format="tempo2")
            except Exception as e:
                raise ValueError(
                    f"Cannot find TT BIPM file for version '{self.bipm_version}'."
                ) from e

    def _load_clock_corrections(self):
        if self._clock is None:
            clock_files = (
                self.clock_fullpath
                if self._multiple_clock_files
                else [self.clock_fullpath]
            )
            self._clock = []
            for clock_file in clock_files:
                if clock_file == "":
                    continue
                log.info(f"Observatory {self.name}, loading clock file {clock_file}")
                self._clock.append(
                    ClockFile.read(
                        clock_file,
                        format=self.clock_fmt,
                        obscode=self.tempo_code,
                        bogus_last_correction=self.bogus_last_correction,
                    )
                )

    def clock_corrections(self, t, limits="warn"):
        """Compute the total clock corrections,

        Parameters
        ----------
        t : astropy.time.Time
            The time when the clock correcions are applied.
        """
        # Read clock file if necessary
        # TODO provide some method for re-reading the clock file?
        self._load_clock_corrections()
        if not self._clock:
            msg = f"No clock corrections found for observatory {self.name} taken from file {self.clock_file}"
            if limits == "warn":
                log.warning(msg)
                corr = np.zeros_like(t) * u.us
            elif limits == "error":
                raise RuntimeError(msg)
        else:
            log.info("Applying observatory clock corrections.")
            corr = self._clock[0].evaluate(t, limits=limits)
            for clock in self._clock[1:]:
                corr += clock.evaluate(t, limits=limits)

        if self.include_gps:
            log.info("Applying GPS to UTC clock correction (~few nanoseconds)")
            self._load_gps_clock()
            corr += _gps_clock.evaluate(t, limits=limits)

        if self.include_bipm:
            log.info(
                f"Applying TT(TAI) to TT({self.bipm_version}) clock correction (~27 us)"
            )
            tt2tai = 32.184 * 1e6 * u.us
            self._load_bipm_clock()
            corr += self._bipm_clock.evaluate(t, limits=limits) - tt2tai
        return corr

    def last_clock_correction_mjd(self):
        """Return the MJD of the last clock correction.

        Combines constraints based on Earth orientation parameters and on the
        available clock corrections specific to the telescope.
        """
        t = np.inf
        self._load_clock_corrections()
        if not self._clock:
            return -np.inf
        for clock in self._clock:
            t = min(t, clock.last_correction_mjd())
        if self.include_gps:
            self._load_gps_clock()
            t = min(t, _gps_clock.last_correction_mjd())
        if self.include_bipm:
            self._load_bipm_clock()
            t = min(t, self._bipm_clock.last_correction_mjd())
        return t

    def _get_TDB_ephem(self, t, ephem):
        """Read the ephem TDB-TT column.

        This column is provided by DE4XXt version of ephemeris. This function is only
        for the ground-based observatories

        """
        geo_tdb_tt = get_tdb_tt_ephem_geocenter(t.tt, ephem)
        # NOTE The earth velocity is need to compute the time correcion from
        # Topocenter to Geocenter
        # Since earth velocity is not going to change a lot in 3ms. The
        # differences between TT and TDB can be ignored.
        earth_pv = objPosVel_wrt_SSB("earth", t.tdb, ephem)
        obs_geocenter_pv = gcrs_posvel_from_itrf(
            self.earth_location_itrf(), t, obsname=self.name
        )
        # NOTE
        # Moyer (1981) and Murray (1983), with fundamental arguments adapted
        # from Simon et al. 1994.
        topo_time_corr = np.sum(earth_pv.vel / c.c * obs_geocenter_pv.pos / c.c, axis=0)
        topo_tdb_tt = geo_tdb_tt - topo_time_corr
        result = Time(
            t.tt.jd1 - JD_MJD,
            t.tt.jd2 - topo_tdb_tt.to(u.day).value,
            format="pulsar_mjd",
            scale="tdb",
            location=self.earth_location_itrf(),
        )
        return result

    def get_gcrs(self, t, ephem=None):
        """Return position vector of TopoObs in GCRS

        Parameters
        ----------
        t : astropy.time.Time or array of astropy.time.Time

        Returns
        -------
        np.array
            a 3-vector of Quantities representing the position in GCRS coordinates.
        """
        obs_geocenter_pv = gcrs_posvel_from_itrf(
            self.earth_location_itrf(), t, obsname=self.name
        )
        return obs_geocenter_pv.pos

    def posvel(self, t, ephem, group=None):
        if t.isscalar:
            t = Time([t])
        earth_pv = objPosVel_wrt_SSB("earth", t, ephem)
        obs_geocenter_pv = gcrs_posvel_from_itrf(
            self.earth_location_itrf(), t, obsname=self.name
        )
        return obs_geocenter_pv + earth_pv
