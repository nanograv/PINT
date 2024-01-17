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
import json
import os
from pathlib import Path
import copy

import astropy.constants as c
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger as log

import pint.observatory
from pint import JD_MJD
from pint.config import runtimefile
from pint.erfautils import gcrs_posvel_from_itrf
from pint.observatory import (
    NoClockCorrections,
    Observatory,
    bipm_default,
    find_clock_file,
    get_observatory,
    earth_location_distance,
)
from pint.pulsar_mjd import Time
from pint.solar_system_ephemerides import get_tdb_tt_ephem_geocenter, objPosVel_wrt_SSB
from pint.utils import has_astropy_unit, open_or_use

# environment variables that can override clock location and observatory location
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

    Additional information can be accessed through the ``location`` attribute

    Parameters
    ----------
    name : str
        The name of the observatory
    fullname : str
        A fuller name of the observatory
    location : ~astropy.coordinates.EarthLocation, optional
    itrf_xyz : ~astropy.units.Quantity or array-like, optional
        IRTF site coordinates (len-3 array).  Can include
        astropy units.  If no units are given, meters are
        assumed.
    lat : ~astropy.units.Quantity or float, optional
        Earth East longitude.  Can be anything that initialises an
        :class:`~astropy.coordinates.Angle` object (if float, in degrees).
    lon : ~astropy.units.Quantity or float, optional
        Earth latitude.  Can be anything that initialises an
        :class:`~astropy.coordinates.Angle` object (if float, in degrees).
    height : ~astropy.units.Quantity ['length'] or float, optional
        Height above reference ellipsoid (if float, in meters; default: 0).
    tempo_code : str, optional
        1-character tempo code for the site.  Will be
        automatically added to aliases.  Note, this is
        REQUIRED only if using TEMPO time.dat clock file.
    itoa_code : str, optional
        2-character ITOA code.  Will be added to aliases.
    aliases : list of str, optional
        List of other aliases for the observatory name.
    clock_file : str or list of str or list of dict or None
        Name of the clock correction file. Can be a list of strings,
        for multiple clock files, or a list of dictionaries if it is
        desired to specify additional keyword arguments to the ClockFile objects.
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
        Set False to disable TAI-> TT BIPM clock
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

    Note
    ----
    One of ``location``, ``itrf_xyz``, or (``lat``, ``lon``, ``height``) must be specified

    """

    def __init__(
        self,
        name,
        *,
        fullname=None,
        tempo_code=None,
        itoa_code=None,
        aliases=None,
        location=None,
        itrf_xyz=None,
        lat=None,
        lon=None,
        height=None,
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
        input_values = [lat is not None, lon is not None, height is not None]
        if sum(input_values) > 0 and sum(input_values) < 3:
            raise ValueError("All of lat, lon, height are required for observatory")
        input_values = [
            location is not None,
            itrf_xyz is not None,
            (lat is not None and lon is not None and height is not None),
        ]
        if sum(input_values) == 0:
            raise ValueError(
                f"EarthLocation, ITRF coordinates, or lat/lon/height are required for observatory '{name}'"
            )
        if sum(input_values) > 1:
            raise ValueError(
                f"Cannot supply more than one of EarthLocation, ITRF coordinates, and lat/lon/height for observatory '{name}'"
            )
        if location is not None:
            self.location = location
        elif itrf_xyz is not None:
            # Convert coords to standard format.  If no units are given, assume
            # meters.
            xyz = (
                itrf_xyz.to(u.m)
                if has_astropy_unit(itrf_xyz)
                else np.array(itrf_xyz) * u.m
            )

            # Check for correct array dims
            if xyz.shape != (3,):
                raise ValueError(
                    f"Incorrect coordinate dimensions for observatory '{name}'"
                )
            # Convert to astropy EarthLocation, ensuring use of ITRF geocentric coordinates
            self.location = EarthLocation.from_geocentric(*xyz)
        elif lat is not None and lon is not None and height is not None:
            self.location = EarthLocation.from_geodetic(lat=lat, lon=lon, height=height)

        # Save clock file info, the data will be read only if clock
        # corrections for this site are requested.
        self.clock_files = [clock_file] if isinstance(clock_file, str) else clock_file
        self.clock_files = [c for c in self.clock_files if c != ""]
        self.clock_fmt = clock_fmt
        self.clock_dir = clock_dir
        self._clock = None  # The ClockFile objects, will be read on demand

        # If using TEMPO time.dat we need to know the 1-char tempo-style
        # observatory code.
        if clock_fmt == "tempo" and clock_file == "time.dat" and tempo_code is None:
            raise ValueError(f"No tempo_code set for observatory '{name}'")

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
        super().__init__(
            name,
            fullname=fullname,
            aliases=aliases,
            include_gps=include_gps,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            overwrite=overwrite,
        )

    def __repr__(self):
        aliases = [f"'{x}'" for x in self.aliases]
        origin = (
            f"{self.fullname}\n{self.origin}"
            if self.fullname != self.name
            else self.origin
        )
        return f"TopoObs('{self.name}' ({','.join(aliases)}) at [{self.location.x}, {self.location.y} {self.location.z}]:\n{origin})"

    @property
    def timescale(self):
        return "utc"

    def get_dict(self):
        """Return as a dict with limited/changed info"""
        # start with the default __dict__
        # copy some attributes to rename them and remove those that aren't needed for initialization
        output = copy.deepcopy(self.__dict__)
        output["aliases"] = output["_aliases"]
        output["clock_file"] = output["clock_files"]
        del output["_name"]
        del output["_aliases"]
        del output["_clock"]
        del output["location"]
        del output["clock_files"]
        output["itrf_xyz"] = [x.to_value(u.m) for x in self.location.geocentric]
        return {self.name: output}

    def get_json(self):
        """Return as a JSON string"""
        return json.dumps(self.get_dict())

    def separation(self, other, method="cartesian"):
        """Return separation between two TopoObs objects

        Parameters
        ----------
        other : TopoObs
        method : str, optional
            Method to compute separation.  Either "cartesian" or "geodesic"

        Returns
        -------
        astropy.quantity.Quantity

        Note
        ----
        "geodesic" method assumes a spherical Earth and ignores altitudes
        """
        assert method.lower() in ["cartesian", "geodesic"]
        assert isinstance(other, TopoObs)

        if method.lower() == "cartesian":
            return earth_location_distance(self.location, other.location)
        elif method.lower() == "geodesic":
            # this assumes a spherical Earth
            dsigma = np.arccos(
                np.sin(self.location.lat) * np.sin(other.location.lat)
                + np.cos(self.location.lat)
                * np.cos(other.location.lat)
                * np.cos(self.location.lon - other.location.lon)
            )
            return (c.R_earth * dsigma).to(u.m, equivalencies=u.dimensionless_angles())

    def earth_location_itrf(self, time=None):
        return self.location

    def _load_clock_corrections(self):
        if self._clock is not None:
            return
        self._clock = []
        for cf in self.clock_files:
            if cf == "":
                continue
            kwargs = dict(bogus_last_correction=self.bogus_last_correction)
            if isinstance(cf, dict):
                kwargs.update(cf)
                cf = kwargs.pop("name")
            self._clock.append(
                find_clock_file(
                    cf,
                    format=self.clock_fmt,
                    clock_dir=self.clock_dir,
                    **kwargs,
                )
            )

    def clock_corrections(self, t, limits="warn"):
        """Compute the total clock corrections,

        Parameters
        ----------
        t : astropy.time.Time
            The time when the clock correcions are applied.
        """

        corr = super().clock_corrections(t, limits=limits)
        # Read clock file if necessary
        self._load_clock_corrections()
        if self._clock:
            log.info(
                f"Applying observatory clock corrections for observatory='{self.name}'."
            )
            for clock in self._clock:
                corr += clock.evaluate(t, limits=limits)

        elif self.clock_files:
            msg = f"No clock corrections found for observatory {self.name} taken from file {self.clock_files}"
            if limits == "warn":
                log.warning(msg)
                corr = np.zeros_like(t) * u.us
            elif limits == "error":
                raise NoClockCorrections(msg)
        else:
            log.info(f"Observatory {self.name} requires no clock corrections.")
        return corr

    def last_clock_correction_mjd(self):
        """Return the MJD of the last clock correction.

        Combines constraints based on Earth orientation parameters and on the
        available clock corrections specific to the telescope.
        """
        t = super().last_clock_correction_mjd()
        self._load_clock_corrections()
        for clock in self._clock:
            t = min(t, clock.last_correction_mjd())
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
        return Time(
            t.tt.jd1 - JD_MJD,
            t.tt.jd2 - topo_tdb_tt.to(u.day).value,
            format="pulsar_mjd",
            scale="tdb",
            location=self.earth_location_itrf(),
        )

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
    if pint.observatory._gps_clock is not None:
        pint.observatory._gps_clock.export(
            directory / Path(pint.observatory._gps_clock.filename).name
        )
    for version, clock in pint.observatory._bipm_clock_versions.items():
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
        if "origin" in obsdict and isinstance(obsdict["origin"], list):
            obsdict["origin"] = "\n".join(obsdict["origin"])
        if overwrite:
            obsdict["overwrite"] = True
        # create the object, which will also register it
        TopoObs(name=obsname, **obsdict)


def load_observatories_from_usual_locations(clear=False):
    """Load observatories from the default JSON file as well as ``$PINT_OBS_OVERRIDE``, optionally clearing the registry

    Running with ``clear=True`` will return PINT to the state it is on import.
    Running with ``clear=False`` may result in conflicting definitions if observatories have already been imported.

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
