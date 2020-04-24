"""Ground-based fixed observatories."""
# topo_obs.py
# Code for dealing with "standard" ground-based observatories.
from __future__ import absolute_import, division, print_function

import os

import astropy.constants as c
import astropy.units as u
import numpy
from astropy import log
from astropy.coordinates import EarthLocation
from six import raise_from

from pint import JD_MJD
from pint.config import datapath
from pint.erfautils import gcrs_posvel_from_itrf
from pint.observatory import Observatory
from pint.observatory.clock_file import ClockFile
from pint.pulsar_mjd import Time
from pint.solar_system_ephemerides import get_tdb_tt_ephem_geocenter, objPosVel_wrt_SSB
from pint.utils import has_astropy_unit


class TopoObs(Observatory):
    """Observatories that are at a fixed location on the surface of the Earth.

    This behaves very similarly to "standard" site definitions in tempo/tempo2.  Clock
    correction files are read and computed, observatory coordinates are specified in
    ITRF XYZ, etc.

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
    bipm_version : str, optionial
        Set the version of TT BIPM clock correction file to
        use, the default is BIPM2015.  It has to be in the format
        like 'BIPM2015'
    """

    def __init__(
        self,
        name,
        tempo_code=None,
        itoa_code=None,
        aliases=None,
        itrf_xyz=None,
        clock_file="time.dat",
        clock_dir="PINT",
        clock_fmt="tempo",
        include_gps=True,
        include_bipm=True,
        bipm_version="BIPM2015",
    ):
        # ITRF coordinates are required
        if itrf_xyz is None:
            raise ValueError("ITRF coordinates not given for observatory '%s'" % name)

        # Convert coords to standard format.  If no units are given, assume
        # meters.
        if not has_astropy_unit(itrf_xyz):
            xyz = numpy.array(itrf_xyz) * u.m
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
        if clock_dir == "TEMPO" and clock_file == "time.dat" and tempo_code is None:
            raise ValueError("No tempo_code set for observatory '%s'" % name)

        # GPS corrections
        self.include_gps = include_gps
        self._gps_clock = None

        # BIPM corrections
        self.include_bipm = include_bipm
        self.bipm_version = bipm_version
        self._bipm_clock = None

        self.tempo_code = tempo_code
        if aliases is None:
            aliases = []
        for code in (tempo_code, itoa_code):
            if code is not None:
                aliases.append(code)

        super(TopoObs, self).__init__(name, aliases=aliases, tt2tdb_mode="astropy")

    @property
    def clock_fullpath(self):
        """Returns the full path to the clock file."""
        if self.clock_dir == "PINT":
            if self._multiple_clock_files:
                return [datapath(f) for f in self.clock_file]
            return datapath(self.clock_file)
        elif self.clock_dir == "TEMPO":
            # Technically should read $TEMPO/tempo.cfg and get clock file
            # location from CLKDIR line...
            TEMPO_dir = os.getenv("TEMPO")
            if TEMPO_dir is None:
                raise RuntimeError("Cannot find TEMPO path from the" " enviroment.")
            dir = os.path.join(TEMPO_dir, "clock")
        elif self.clock_dir == "TEMPO2":
            TEMPO2_dir = os.getenv("TEMPO2")
            if TEMPO2_dir is None:
                raise RuntimeError("Cannot find TEMPO2 path from the" " enviroment.")
            dir = os.path.join(TEMPO2_dir, "clock")
        else:
            dir = self.clock_dir
        if self._multiple_clock_files:
            return [os.path.join(dir, f) for f in self.clock_file]
        return os.path.join(dir, self.clock_file)

    @property
    def gps_fullpath(self):
        """Returns full path to the GPS-UTC clock file.  Will first try PINT
        data dirs, then fall back on $TEMPO2/clock."""
        fname = "gps2utc.clk"
        fullpath = datapath(fname)
        if fullpath is not None:
            return fullpath
        return os.path.join(os.getenv("TEMPO2"), "clock", fname)

    @property
    def bipm_fullpath(self,):
        """Returns full path to the TAI TT(BIPM) clock file.

        Will first try PINT data dirs, then fall back on $TEMPO2/clock.
        """
        fname = "tai2tt_" + self.bipm_version.lower() + ".clk"
        try:
            fullpath = datapath(fname)
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

    def clock_corrections(self, t):
        """Compute the total clock corrections,

        Parameters
        ----------
        t : astropy.time.Time
            The time when the clock correcions are applied.
        """
        # Read clock file if necessary
        # TODO provide some method for re-reading the clock file?
        if self._clock is None:
            clock_files = (
                self.clock_fullpath
                if self._multiple_clock_files
                else [self.clock_fullpath]
            )
            self._clock = []
            for clock_file in clock_files:
                log.info(
                    "Observatory {0}, loading clock file \n\t{1}".format(
                        self.name, clock_file
                    )
                )
                self._clock.append(
                    ClockFile.read(
                        clock_file, format=self.clock_fmt, obscode=self.tempo_code
                    )
                )
        log.info("Applying observatory clock corrections.")
        corr = self._clock[0].evaluate(t)
        for clock in self._clock[1:]:
            corr += clock.evaluate(t)

        if self.include_gps:
            log.info("Applying GPS to UTC clock correction (~few nanoseconds)")
            if self._gps_clock is None:
                log.info(
                    "Observatory {0}, loading GPS clock file \n\t{1}".format(
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
                        "Observatory {0}, loading BIPM clock file \n\t{1}".format(
                            self.name, self.bipm_fullpath
                        )
                    )
                    self._bipm_clock = ClockFile.read(
                        self.bipm_fullpath, format="tempo2"
                    )
                except Exception as e:
                    raise_from(
                        ValueError(
                            "Can not find TT BIPM file '%s'. " % self.bipm_version
                        ),
                        e,
                    )
            corr += self._bipm_clock.evaluate(t) - tt2tai
        return corr

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
        topo_time_corr = numpy.sum(
            earth_pv.vel / c.c * obs_geocenter_pv.pos / c.c, axis=0
        )
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

    def posvel(self, t, ephem):
        if t.isscalar:
            t = Time([t])
        earth_pv = objPosVel_wrt_SSB("earth", t, ephem)
        obs_geocenter_pv = gcrs_posvel_from_itrf(
            self.earth_location_itrf(), t, obsname=self.name
        )
        return obs_geocenter_pv + earth_pv
