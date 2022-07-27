"""Ground-based fixed observatories read in from ``observatories.json``.

These observatories have fixed positions that affect the data they record, but
they also often have their own reference clocks, and therefore we need to
correct for any drift in those clocks.

These observatories are registered when this file is imported.   
The standard behavior is given by :func:`pint.observatory.topo_obs.load_observatories_from_usual_locations`, which:

* Clears any existing observatories from the registry
* Loads the standard observatories 
* Loads any observatories present in ``$PINT_OBS_OVERRIDE``, overwriting those already present

This is run on import.  Otherwise it only needs to be run if :func:`pint.observatory.Observatory.clear_registry` is run.

See Also
--------
:mod:`pint.observatory.special_locations`
"""
import os
import json

import astropy.constants as c
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger as log
from pathlib import Path

from pint import JD_MJD
from pint.config import runtimefile
from pint.erfautils import gcrs_posvel_from_itrf
from pint.observatory import (
    Observatory,
    bipm_default,
    get_observatory,
    ClockCorrectionError,
    NoClockCorrections,
    ClockCorrectionOutOfRange,
)
from pint.observatory.clock_file import ClockFile, GlobalClockFile
from pint.pulsar_mjd import Time
from pint.solar_system_ephemerides import get_tdb_tt_ephem_geocenter, objPosVel_wrt_SSB
from pint.utils import has_astropy_unit, open_or_use
from pint.observatory.global_clock_corrections import Index, get_clock_correction_file

# environment variables that can override clock location and observatory location
pint_clock_env_var = "PINT_CLOCK_OVERRIDE"
pint_obs_env_var = "PINT_OBS_OVERRIDE"

# where to look for observatory data
observatories_json = runtimefile("observatories.json")


__all__ = [
    "TopoObs",
    "find_clock_file",
    "export_all_clock_files",
    "observatories_json",
    "load_observatories",
    "load_observatories_from_usual_locations",
]

# These are global because they are, well, literally global
_gps_clock = None
_bipm_clock_versions = {}


class TopoObs(Observatory):
    """Observatories that are at a fixed location on the surface of the Earth.

    This behaves very similarly to "standard" site definitions in tempo/tempo2.  Clock
    correction files are read and computed, observatory coordinates are specified in
    ITRF XYZ, etc.

    PINT can look for clock files in one of several ways, depending on how the
    ``clock_dir`` variable is set:

    * ``clock_dir="PINT"`` - clock files are looked for in ``$PINT_CLOCK_OVERRIDE``, or failing that, in a global clock correction repository
    * ``clock_dir="TEMPO"`` or ``clock_dir="TEMPO2"`` - clock files are looked for under ``$TEMPO`` or ``$TEMPO2``
    * ``clock_dir`` is a specific directory

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
    clock_fmt : str, optional
        Format of clock file (see ClockFile class for allowed
        values).
    clock_dir : str or pathlib.Path, optional
        Where to look for the clock files. "PINT", the default, means to use
        PINT's usual seach approach; "TEMPO" or "TEMPO2" mean to look in those
        programs' usual location (pointed to by their environment variables),
        while a path means to look in that specific directory.
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
        use, the default is the most recent available.
        It should be a string like 'BIPM2015' to select the file
        'tai2tt_bipm2015.clk'.
    origin : str, optional
        Documentation of the origin/author/date for the information
    overwrite : bool, optional
        set True to force overwriting of previous observatory definition
    bogus_last_correction : bool, optional
        Clock correction files include a bogus last correction
    """

    def __init__(
        self,
        name,
        *,
        tempo_code=None,
        itoa_code=None,
        aliases=None,
        itrf_xyz=None,
        lat=None,
        lon=None,
        alt=None,
        clock_file="",
        clock_fmt="tempo",
        clock_dir=None,
        include_gps=True,
        include_bipm=True,
        bipm_version=bipm_default,
        origin=None,
        overwrite=False,
        bogus_last_correction=False,
    ):
        if lat is None and lon is None and alt is None and itrf_xyz is None:
            raise ValueError(
                "ITRF coordinates or lat/lon/alt are required for observatory '%s'"
                % name
            )

        if itrf_xyz is not None and (
            lat is not None and lon is not None and alt is not None
        ):
            raise ValueError(
                f"Cannot supply both ITRF coordinates and lat/lon/alt for observatory '{name}'"
            )

        if itrf_xyz is not None:
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
        elif lat is not None and lon is not None and alt is not None:
            self._loc_itrf = EarthLocation.from_geodetic(lat=lat, lon=lon, height=alt)

        #
        # Save clock file info, the data will be read only if clock
        # corrections for this site are requested.
        self.clock_file = clock_file
        self._multiple_clock_files = not isinstance(clock_file, str)
        self.clock_fmt = clock_fmt
        self.clock_dir = clock_dir
        self._clock = None  # The ClockFile objects, will be read on demand

        # If using TEMPO time.dat we need to know the 1-char tempo-style
        # observatory code.
        if clock_fmt == "tempo" and clock_file == "time.dat" and tempo_code is None:
            raise ValueError("No tempo_code set for observatory '%s'" % name)

        # GPS corrections
        self.include_gps = include_gps

        # BIPM corrections
        # WARNING: `get_observatory` changes these after construction
        self.include_bipm = include_bipm
        self.bipm_version = bipm_version
        self.bogus_last_correction = bogus_last_correction

        self.tempo_code = tempo_code
        self.itoa_code = itoa_code
        if aliases is None:
            aliases = []
        for code in (tempo_code, itoa_code):
            if code is not None:
                aliases.append(code)

        self.origin = origin
        super().__init__(name, aliases=aliases)

    def __repr__(self):
        aliases = [f"'{x}'" for x in self.aliases]
        s = f"TopoObs('{self.name}' ({','.join(aliases)}) at [{self._loc_itrf.x}, {self._loc_itrf.y} {self._loc_itrf.z}]:\n{self.origin})"
        return s

    @property
    def timescale(self):
        return "utc"

    @property
    def x(self):
        """The X component of the geocentric coordinates."""
        return self._loc_itrf.x

    @property
    def y(self):
        """The Y component of the geocentric coordinates."""
        return self._loc_itrf.y

    @property
    def z(self):
        """The Z component of the geocentric coordinates."""
        return self._loc_itrf.z

    @property
    def geocentric(self):
        """Convert to a tuple with X, Y, and Z as quantities"""
        return self._loc_itrf.to_geocentric()

    @property
    def lat(self):
        """Latitude of the location, for the default ellipsoid."""
        return self._loc_itrf.lat

    @property
    def lon(self):
        """Longitude of the location, for the default ellipsoid."""
        return self._loc_itrf.lon

    @property
    def alt(self):
        """Altitude of the location, for the default ellipsoid."""
        return self._loc_itrf.height

    @property
    def geodetic(self):
        """Convert to geodetic coordinates for the default ellipsoid."""
        return self._loc_itrf.to_geodetic()

    @property
    def as_json(self):
        """Return a JSON string"""
        # is this better than the builtin __dict__ method and then updating some values?
        output = {}
        output["itrf_xyz"] = [x.to_value(u.m) for x in self.geocentric]
        if len(self.aliases) > 0:
            output["aliases"] = self.aliases
        if self.tempo_code is not None:
            output["tempo_code"] = self.tempo_code
        if self.itoa_code is not None:
            output["itoa_code"] = self.itoa_code
        if self.clock_file is not None and len(self.clock_file) > 0:
            output["clock_file"] = self.clock_file
        if self.clock_fmt is not None and len(self.clock_fmt) > 0:
            output["clock_fmt"] = self.clock_fmt
        if self.clock_dir is not None:
            output["clock_dir"] = self.clock_dir
        for p in [
            "include_gps",
            "include_bipm",
            "bipm_version",
            "bogus_last_correction",
        ]:
            output[p] = getattr(self, p)
        if self.origin is not None and len(self.origin) > 0:
            output["origin"] = self.origin
        return json.dumps({self.name: output})

    def earth_location_itrf(self, time=None):
        return self._loc_itrf

    def _load_gps_clock(self):
        global _gps_clock
        if _gps_clock is None:
            log.info(f"Loading global GPS clock file for {self.name}")
            _gps_clock = find_clock_file(
                "gps2utc.clk",
                format="tempo2",
                clock_dir=self.clock_dir,
            )
            if len(_gps_clock.time) == 0:
                raise NoClockCorrections(
                    "Unable to obtain GPS to UTC clock corrections"
                )

    def _load_bipm_clock(self, bipm_version):
        bipm_version = bipm_version.lower()
        if bipm_version not in _bipm_clock_versions:
            try:
                log.info(
                    f"Loading BIPM clock version {bipm_version} " f"for {self.name}"
                )
                # FIXME: error handling?
                _bipm_clock_versions[bipm_version] = find_clock_file(
                    f"tai2tt_{bipm_version}.clk",
                    format="tempo2",
                    clock_dir=self.clock_dir,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot find TT BIPM file for version '{bipm_version}'."
                ) from e
            if len(_bipm_clock_versions[bipm_version].time) == 0:
                raise NoClockCorrections(
                    f"Unable to obtain BIPM {bipm_version} clock corrections"
                )

    def _load_clock_corrections(self):
        if self._clock is None:
            # FIXME: handle other clock_dir values
            # FIXME: handle ""
            clock_files = (
                self.clock_file if self._multiple_clock_files else [self.clock_file]
            )
            self._clock = [
                find_clock_file(
                    c,
                    format=self.clock_fmt,
                    clock_dir=self.clock_dir,
                    bogus_last_correction=self.bogus_last_correction,
                )
                for c in clock_files
                if c != ""
            ]

    def clock_corrections(self, t, limits="warn"):
        """Compute the total clock corrections,

        Parameters
        ----------
        t : astropy.time.Time
            The time when the clock correcions are applied.
        """

        # Read clock file if necessary
        corr = np.zeros_like(t) * u.us
        self._load_clock_corrections()
        if not self._clock:
            if self.clock_file:
                msg = f"No clock corrections found for observatory {self.name} taken from file {self.clock_file}"
                if limits == "warn":
                    log.warning(msg)
                    corr = np.zeros_like(t) * u.us
                elif limits == "error":
                    raise NoClockCorrections(msg)
            else:
                log.info(f"Observatory {self.name} requires no clock corrections.")
        else:
            log.info("Applying observatory clock corrections.")
            for clock in self._clock:
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
            self._load_bipm_clock(self.bipm_version)
            corr += (
                _bipm_clock_versions[self.bipm_version.lower()].evaluate(
                    t, limits=limits
                )
                - tt2tai
            )
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
            self._load_bipm_clock(self.bipm_version)
            t = min(
                t, _bipm_clock_versions[self.bipm_version.lower()].last_correction_mjd()
            )
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


# FIXME: is clock_file the right place for this? The rest of the file isn't very
# PINT-specific but this is. Maybe in topo_obs - does that cover all the clock files
# we need?
def find_clock_file(
    name, format, bogus_last_correction=False, url_base=None, clock_dir=None
):
    """Locate and return a ClockFile in one of several places.

    PINT looks for clock files in three places, in order:

    1. The directory ``$PINT_CLOCK_OVERRIDE``
    2. The global clock correction repository on the Internet (or a locally cached copy)
    3. The directory ``pint.config.runtimefile('.')``

    The first place the file is found is the one use; this allows you to force PINT to
    use your own files in place of those in the global repository.

    Parameters
    ----------
    name : str
        The name of the file, for example ``time_ao.dat``.
    format : "tempo" or "tempo2"
        The format of the file; this also determines where in the global repository
        to look for it.
    bogus_last_correction : bool
        Whether the file contains a far-future value to help other programs'
        interpolation cope.
    url_base : str or None
        Override the usual location to look for global clock corrections
        (mostly useful for testing)
    clock_dir : str or pathlib.Path or None
        If None or "PINT", use the above procedure; if "TEMPO" or "TEMPO2" use
        those programs' customary locations; if a path, look there specifically.

    Returns
    -------
    ClockFile
    """
    if name == "":
        raise ValueError("No filename supplied to find_clock_file")
    if clock_dir is None or clock_dir.upper() == "PINT":
        # Don't try loading it from a specific path
        p = None
    elif clock_dir.lower() == "tempo":
        if "TEMPO" not in os.environ:
            raise NoClockCorrections(
                f"TEMPO environment variable not set but clock file {name} "
                f"is supposed to be in the directory it points to"
            )
        p = Path(os.environ["TEMPO"]) / "clock" / name
    elif clock_dir.lower() == "tempo2":
        if "TEMPO2" not in os.environ:
            raise NoClockCorrections(
                f"TEMPO2 environment variable not set but clock file {name} "
                f"is supposed to be in the directory it points to"
            )
        # Look in the TEMPO2 directory and nowhere else
        p = Path(os.environ["TEMPO2"]) / "clock" / name
    else:
        # assume it's a path and look in there
        p = Path(clock_dir) / name
    if p is not None:
        log.info(f"Loading clock file {p} from specified location")
        return ClockFile.read(
            p,
            format=format,
            bogus_last_correction=bogus_last_correction,
            friendly_name=name,
        )

    # FIXME: implement clock_dir
    env_clock = None
    global_clock = None
    local_clock = None
    if pint_clock_env_var in os.environ:
        loc = Path(os.environ[pint_clock_env_var]) / name
        if loc.exists():
            # FIXME: more arguments?
            env_clock = ClockFile.read(
                loc,
                format=format,
                bogus_last_correction=bogus_last_correction,
                friendly_name=name,
            )
            # Could just return this but we want to emit
            # a warning with an appropriate level of forcefulness
    index = Index(url_base=url_base)
    if name in index.files:
        global_clock = GlobalClockFile(
            name,
            format=format,
            bogus_last_correction=bogus_last_correction,
            url_base=url_base,
        )
    loc = Path(runtimefile(name))
    if loc.exists():
        local_clock = ClockFile.read(
            loc,
            format=format,
            bogus_last_correction=bogus_last_correction,
            friendly_name=name,
        )

    if env_clock is not None:
        if global_clock is not None:
            # FIXME: if we're not going to use the values from the global clock
            # we could have saved downloading and parsing it
            log.warning(
                f"Clock file from {env_clock.filename} overrides global clock "
                f"file {name} because of {pint_clock_env_var}"
            )
        else:
            log.info(f"Using clock file from {env_clock.filename}")
        return env_clock
    elif global_clock is not None:
        log.info(f"Using global clock file for {name} with {bogus_last_correction=}")
        return global_clock
    elif local_clock is not None:
        log.info(f"Using local clock file for {name}")
        return local_clock
    else:
        # Null clock file should return warnings/exceptions if ever you try to
        # look up a data point in it
        log.info(f"No clock file for {name}")
        return ClockFile.null(friendly_name=name)


def export_all_clock_files(directory):
    """Export all clock files PINT is using.

    This will export all the clock files PINT is using - every clock file used
    by any observatory, as well as those relating to BIPM time scales that have
    been used in this invocation of PINT. Clock files will not be updated and
    new ones will not be downloaded before this export.

    You should be able to set PINT_CLOCK_OVERRIDE to a directory constructed
    in this way in order to ensure that specifically these versions of the
    clock files are used.

    Parameters
    ----------
    directory : str or pathlib.Path
        Where to put the files.
    """
    directory = Path(directory)
    if _gps_clock is not None:
        _gps_clock.export(directory / Path(_gps_clock.filename).name)
    for version, clock in _bipm_clock_versions.items():
        clock.export(directory / Path(clock.filename).name)
    for name in Observatory.names():
        o = get_observatory(name)
        if hasattr(o, "_clock") and o._clock is not None:
            for clock in o._clock:
                if clock.filename is not None:
                    clock.export(directory / Path(clock.filename).name)


def load_observatories(filename=observatories_json, overwrite=False):
    """Load observatory definitions from JSON and create :class:`pint.observatory.topo_obs.TopoObs` objects, registering them

    Set `overwrite` to ``True`` if you want to re-read a file with updated definitions.
    If `overwrite` is ``False`` and you attempt to add an existing observatory, an exception is raised.

    Parameters
    ----------
    filename : str or file-like object, optional
    overwrite : bool, optional
        Whether a new instance of an existing observatory should overwrite the existing one.

    Raises
    ------
    ValueError
        If an attempt is made to add an existing observatory with ``overwrite=False``

    Notes
    -----
    If the ``origin`` field is a list of strings, they will be joined together with newlines.
    """
    # read in the JSON file
    with open_or_use(filename, "r") as f:
        observatories = json.load(f)

    for obsname, obsdict in observatories.items():
        if "origin" in obsdict:
            if isinstance(obsdict["origin"], list):
                obsdict["origin"] = "\n".join(obsdict["origin"])
        if overwrite:
            obsdict["overwrite"] = True
        # create the object, which will also register it
        TopoObs(name=obsname, **obsdict)


def load_observatories_from_usual_locations(clear=False):
    """Load observatories from the default JSON file as well as ``$PINT_OBS_OVERRIDE``, optionally clearing the registry

    Running with ``clear=True`` will return PINT to the state it is on import.  Running with ``clear=False`` may result in conflicting definitions if observatories have already been imported.

    Parameters
    ----------
    clear : bool, optional
        Whether or not to clear existing objects in advance
    """
    if clear:
        Observatory.clear_registry()
    # read the observatories
    load_observatories()
    # potentially override any defined here
    if pint_obs_env_var in os.environ:
        load_observatories(os.environ[pint_obs_env_var], overwrite=True)


load_observatories_from_usual_locations()
