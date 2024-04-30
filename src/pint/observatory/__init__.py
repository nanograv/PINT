"""Machinery to support PINT's list of observatories.

This code maintains a registry of observatories known to PINT.
Observatories are added to the registry when the objects are created.
For the observatories defined in PINT, these objects are created
when the relevant module is imported.

PINT's built-in observatories are loaded when anyone imports the modules
:mod:`pint.observatory.topo_obs` and
:mod:`pint.observatory.special_locations`. This automatically happens
when you call :func:`pint.observatory.Observatory.get`,
:func:`pint.observatory.get_observatory`, or
:func:`pint.observatory.Observatory.names`
(:func:`pint.observatory.Observatory.names_and_aliases` to include aliases).
Satellite observatories are somewhat different, as they cannot be
created until the user supplies an orbit file. Once created, they will
appear in the list of known observatories.

Normal use of :func:`pint.toa.get_TOAs` will ensure that all imports have been
done, but if you are using a different subset of PINT these imports may be
necessary.
"""

from copy import deepcopy
import os
import textwrap
from collections import defaultdict
from io import StringIO
from pathlib import Path

import astropy.coordinates
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger as log

from pint.config import runtimefile
from pint.pulsar_mjd import Time
from pint.utils import interesting_lines

# Include any files that define observatories here.  This will start
# with the standard distribution files, then will read any system- or
# user-defined files.  These can override the default settings by
# redefining an Observatory with the same name.
# TODO read the files from the other locations, if they exist

__all__ = [
    "Observatory",
    "get_observatory",
    "list_last_correction_mjds",
    "update_clock_files",
    "compare_t2_observatories_dat",
    "compare_tempo_obsys_dat",
    "earth_location_distance",
    "ClockCorrectionError",
    "NoClockCorrections",
    "ClockCorrectionOutOfRange",
]

# The default BIPM to use if not explicitly specified
# FIXME: this should be auto-detected by checking the index file to see what's available
bipm_default = "BIPM2021"

pint_clock_env_var = "PINT_CLOCK_OVERRIDE"


class ClockCorrectionError(RuntimeError):
    """Unspecified error doing clock correction."""

    pass


class NoClockCorrections(ClockCorrectionError):
    """Clock corrections are expected but none are available."""

    pass


class ClockCorrectionOutOfRange(ClockCorrectionError):
    """Clock corrections are available but the requested time is not covered."""

    pass


# Global clock files shared by all observatories
_gps_clock = None
_bipm_clock_versions = {}


def _load_gps_clock():
    global _gps_clock
    if _gps_clock is None:
        log.info("Loading global GPS clock file")
        _gps_clock = find_clock_file(
            "gps2utc.clk",
            format="tempo2",
        )


def _load_bipm_clock(bipm_version):
    bipm_version = bipm_version.lower()
    if bipm_version not in _bipm_clock_versions:
        try:
            log.info(f"Loading BIPM clock version {bipm_version}")
            # FIXME: error handling?
            # FIXME: BIPM2019 and earlier come fromm the TEMPO2 repository and have a bogus last correction
            # later ones are generated and don't; how to manage this?
            bogus_last_correction = bipm_version.lower() <= "bipm2019"
            _bipm_clock_versions[bipm_version] = find_clock_file(
                f"tai2tt_{bipm_version}.clk",
                format="tempo2",
                bogus_last_correction=bogus_last_correction,
            )
        except Exception as e:
            raise ValueError(
                f"Cannot find TT BIPM file for version '{bipm_version}'."
            ) from e


class Observatory:
    """Observatory locations and related site-dependent properties

    For example, TOA time scales, clock corrections.  Any new Observatory that
    is declared will be automatically added to a registry that is keyed on
    observatory name.  Aside from their initial declaration (for examples, see
    ``pint/data/runtimefile/observatories.json``), Observatory instances should
    generally be obtained only via the :func:`pint.observatory.Observatory.get`
    function.  This will query the registry based on observatory name (and any
    defined aliases).  A list of all registered names can be returned via
    :func:`pint.observatory.Observatory.names`, or a list of names and aliases
    can be returned via :func:`pint.observatory.Observatory.names_and_aliases`.

    Observatories have names and aliases that are used in ``.tim`` and ``.par``
    files to select them. They also have positions (possibly varying, in the
    case of satellite observatories) and may have associated clock corrections
    to relate times observed at the observatory clock to global time scales.

    Terrestrial observatories are generally instances of the
    :class:`pint.observatory.topo_obs.TopoObs` class, which has a fixed
    position.
    """

    # This is a dict containing all defined Observatory instances,
    # keyed on standard observatory name.
    _registry = {}

    # This is a dict mapping any defined aliases to the corresponding
    # standard name.
    _alias_map = {}

    def __init__(
        self,
        name,
        fullname=None,
        aliases=None,
        include_gps=True,
        include_bipm=True,
        bipm_version=bipm_default,
        overwrite=False,
    ):
        self._name = name.lower()
        self._aliases = (
            list(set(map(str.lower, aliases))) if aliases is not None else []
        )
        if aliases is not None:
            Observatory._add_aliases(self, aliases)
        self.fullname = fullname if fullname is not None else name
        self.include_gps = include_gps
        self.include_bipm = include_bipm
        self.bipm_version = bipm_version

        if name.lower() in Observatory._registry:
            if not overwrite:
                raise ValueError(
                    f"Observatory {name.lower()} already present and overwrite=False"
                )
            log.warning(f"Observatory '{name.lower()}' already present; overwriting...")

        Observatory._register(self, name)

    @classmethod
    def _register(cls, obs, name):
        """Add an observatory to the registry using the specified name
        (which will be converted to lower case).  If an existing observatory
        of the same name exists, it will be replaced with the new one.
        The Observatory instance's name attribute will be updated for
        consistency."""
        cls._registry[name.lower()] = obs

    @classmethod
    def _add_aliases(cls, obs, aliases):
        """Add aliases for the specified Observatory.  Aliases
        should be given as a list.  If any of the new aliases are already in
        use, they will be replaced.  Aliases are not checked against the
        list of observatory names, but names always take precedence over
        aliases when matching.  After the new aliases are in place, the
        aliases attribute is updated for all registered observatories
        to ensure consistency."""
        for a in aliases:
            cls._alias_map[a.lower()] = obs.name

    @staticmethod
    def gps_correction(t, limits="warn"):
        """Compute the GPS clock corrections for times t."""
        log.info("Applying GPS to UTC clock correction (~few nanoseconds)")
        _load_gps_clock()
        return _gps_clock.evaluate(t, limits=limits)

    @staticmethod
    def bipm_correction(t, bipm_version=bipm_default, limits="warn"):
        """Compute the GPS clock corrections for times t."""
        log.info(f"Applying TT(TAI) to TT({bipm_version}) clock correction (~27 us)")
        tt2tai = 32.184 * 1e6 * u.us
        _load_bipm_clock(bipm_version)
        return (
            _bipm_clock_versions[bipm_version.lower()].evaluate(t, limits=limits)
            - tt2tai
        )

    @classmethod
    def clear_registry(cls):
        """Clear registry for ground-based observatories."""
        cls._registry = {}
        cls._alias_map = {}

    @classmethod
    def names(cls):
        """List all observatories known to PINT."""
        # Importing this module triggers loading all observatories
        import pint.observatory.topo_obs  # noqa
        import pint.observatory.special_locations  # noqa

        return cls._registry.keys()

    @classmethod
    def names_and_aliases(cls):
        """List all observatories and their aliases"""
        import pint.observatory.topo_obs  # noqa
        import pint.observatory.special_locations  # noqa

        return {oname: obs.aliases for oname, obs in cls._registry.items()}

    # Note, name and aliases are not currently intended to be changed
    # after initialization.  If we want to allow this, we could add
    # setter methods that update the registries appropriately.

    @property
    def name(self):
        return self._name

    @property
    def aliases(self):
        return self._aliases

    @classmethod
    def get(cls, name):
        """Returns the Observatory instance for the specified name/alias.

        If the name has not been defined, an error will be raised.  Aside
        from the initial observatory definitions, this is in general the
        only way Observatory objects should be accessed.  Name-matching
        is case-insensitive.
        """
        # Ensure that the observatory list has been read
        # We can't do this in the import section above because this class
        # needs to exist before that file is imported.
        import pint.observatory.topo_obs  # noqa
        import pint.observatory.special_locations  # noqa

        if name == "":
            raise KeyError("No observatory name or code provided")

        # Be case-insensitive
        name = name.lower()
        # First see if name matches
        if name in cls._registry:
            return cls._registry[name]
        # Then look for aliases
        if name in cls._alias_map:
            return cls._registry[cls._alias_map[name]]
        # Then look in astropy
        log.warning(
            f"Observatory name {name} is not present in PINT observatory list; searching astropy..."
        )
        # the name was not found in the list of standard PINT observatories
        # see if we can it from astropy
        try:
            site_astropy = astropy.coordinates.EarthLocation.of_site(name)
        except astropy.coordinates.errors.UnknownSiteException as e:
            # turn it into the same error type as PINT would have returned
            raise KeyError(f"Observatory name '{name}' is not defined") from e

        # we need to import this here rather than up-top because of circular import issues
        from pint.observatory.topo_obs import TopoObs

        # add in metadata from astropy
        obs = TopoObs(
            name,
            location=site_astropy,
            origin=f"""astropy: '{site_astropy.info.meta["source"]}'""",
        )
        # add to registry
        cls._register(obs, name)
        return cls._registry[name]

    # The following methods define the basic API for the Observatory class.
    # Any which raise NotImplementedError below must be implemented in
    # derived classes.

    def earth_location_itrf(self, time=None):
        """Returns observatory geocentric position as an astropy
        EarthLocation object.  For observatories where this is not
        relevant, None can be returned.

        The location is in the International Terrestrial Reference Frame (ITRF).
        The realization of the ITRF is determined by astropy,
        which uses ERFA (IAU SOFA).

        The time argument is ignored for observatories with static
        positions. For moving observatories (e.g. spacecraft), it
        should be specified (as an astropy Time) and the position
        at that time will be returned.
        """
        return None

    def get_gcrs(self, t, ephem=None):
        """Return position vector of observatory in GCRS
        t is an astropy.Time or array of astropy.Time objects
        ephem is a link to an ephemeris file. Needed for SSB observatory
        Returns a 3-vector of Quantities representing the position
        in GCRS coordinates.
        """
        raise NotImplementedError

    @property
    def timescale(self):
        """Returns the timescale that TOAs from this observatory will be in,
        once any clock corrections have been applied.  This should be a
        string suitable to be passed directly to the scale argument of
        astropy.time.Time()."""
        raise NotImplementedError

    def clock_corrections(self, t, limits="warn"):
        """Compute clock corrections for a Time array.

        Given an array-valued Time, return the clock corrections
        as a numpy array, with units.  These values are to be added to the
        raw TOAs in order to refer them to the timescale specified by
        self.timescale."""
        # TODO this and derived methods should be changed to accept a TOA
        # table in addition to Time objects.  This will allow access to extra
        # TOA metadata which may be necessary in some cases.
        corr = np.zeros_like(t) * u.us

        if self.include_gps:
            corr += self.gps_correction(t, limits=limits)

        if self.include_bipm:
            corr += self.bipm_correction(t, self.bipm_version, limits=limits)

        return corr

    def last_clock_correction_mjd(self):
        """Return the MJD of the last available clock correction.

        Returns ``np.inf`` if no clock corrections are relevant.
        """
        t = np.inf

        if self.include_gps:
            _load_gps_clock()
            t = min(t, _gps_clock.last_correction_mjd())
        if self.include_bipm:
            _load_bipm_clock(self.bipm_version)
            t = min(
                t,
                _bipm_clock_versions[self.bipm_version.lower()].last_correction_mjd(),
            )
        return t

    def get_TDBs(self, t, method="default", ephem=None, options=None):
        """This is a high level function for converting TOAs to TDB time scale.

        Different method can be applied to obtain the result. Current supported
        methods are ['default', 'ephemeris']

        Parameters
        ----------
        t: astropy.time.Time object
            The time need for converting toas
        method: str or callable, optional
            Method of computing TDB

            "default"
                Astropy time.Time object built-in converter, uses FB90.
                SpacecraftObs will include a topocentric correction term.
            "ephemeris"
                JPL ephemeris included TDB-TT correction. Not currently
                implemented.
            callable
                This callable is called with the parameter t as its first
                parameter; additional keyword arguments can be supplied
                in the options argument

        ephem: str, optional
            The ephemeris to get he TDB-TT correction. Required for the
            'ephemeris' method.
        options: dict or None
            Options to pass to a custom callable.
        """

        if t.isscalar:
            t = Time([t])
        if t.scale == "tdb":
            return t
        # Check the method. This pattern is from numpy minimize
        meth = "_custom" if callable(method) else method.lower()
        if options is None:
            options = {}
        if meth == "_custom":
            options = dict(options)
            return method(t, **options)
        if meth == "default":
            return self._get_TDB_default(t, ephem)
        elif meth == "ephemeris":
            if ephem is None:
                raise ValueError(
                    "A ephemeris file should be provided to get"
                    " the TDB-TT corrections."
                )
            return self._get_TDB_ephem(t, ephem)
        else:
            raise ValueError(f"Unknown method '{method}'.")

    def _get_TDB_default(self, t, ephem):
        return t.tdb

    def _get_TDB_ephem(self, t, ephem):
        """Read the ephem TDB-TT column.

        This column is provided by DE4XXt version of ephemeris.
        """
        raise NotImplementedError

    def posvel(self, t, ephem, group=None):
        """Return observatory position and velocity for the given times.

        Position is relative to solar system barycenter; times are
        (astropy array-valued Time objects).
        """
        # TODO this and derived methods should be changed to accept a TOA
        # table in addition to Time objects.  This will allow access to extra
        # TOA metadata which may be necessary in some cases.
        raise NotImplementedError


def get_observatory(
    name, include_gps=None, include_bipm=None, bipm_version=bipm_default
):
    """Convenience function to get observatory object with options.

    This function will simply call the ``Observatory.get`` method but
    will manually modify the global observatory object after the method is called.
    Name-matching is case-insensitive.

    If the observatory is not present in the PINT list, will fallback to astropy.

    Parameters
    ----------
    name : str
        The name of the observatory
    include_gps : bool or None, optional
        Override UTC(GPS)->UTC clock correction.
    include_bipm : bool or None, optional
        Override TAI TT(BIPM) clock correction.
    bipm_version : str, optional
        Set the version of TT BIPM clock correction files.

    .. note:: This function can and should be expanded if more clock
        file switches/options are added at a public API level.

    """
    if include_bipm is not None or include_gps is not None:
        site = deepcopy(Observatory.get(name))

        if include_gps is not None:
            site.include_gps = include_gps

        if include_bipm is not None:
            site.include_bipm = include_bipm
            site.bipm_version = bipm_version

        return site

    return Observatory.get(name)


def earth_location_distance(loc1, loc2):
    """Compute the distance between two EarthLocations."""
    return (
        sum((u.Quantity(loc1.to_geocentric()) - u.Quantity(loc2.to_geocentric())) ** 2)
    ) ** 0.5


def compare_t2_observatories_dat(t2dir=None):
    """Read a tempo2 observatories.dat file and compare with PINT

    Produces a report including lines that can be added to PINT's
    observatories.json to add any observatories unknown to PINT.

    Parameters
    ==========
    t2dir : str, optional
        Path to the TEMPO2 runtime dir; if not provided, look in the
        TEMPO2 environment variable.

    Returns
    =======
    dict
        The dictionary has two entries, under the keys "different" and "missing"; each is
        a list of observatories found in the TEMPO2 files that disagree with what PINT
        expects. Each entry in these lists is again a dict, with various properties of the
        observatory, including a line that might be suitable for starting an entry in the
        PINT observatory list.
    """
    if t2dir is None:
        t2dir = os.getenv("TEMPO2")
    if t2dir is None:
        raise ValueError(
            "TEMPO2 directory not provided and TEMPO2 environment variable not set"
        )
    filename = os.path.join(t2dir, "observatory", "observatories.dat")

    report = defaultdict(list)
    with open(filename) as f:
        for line in interesting_lines(f, comments="#"):
            try:
                x, y, z, full_name, short_name = line.split()
            except ValueError as e:
                raise ValueError(f"unrecognized line '{line}'") from e
            x, y, z = float(x), float(y), float(z)
            full_name, short_name = full_name.lower(), short_name.lower()
            topo_obs_entry = textwrap.dedent(
                f"""
                "{full_name}": {{
                    "aliases": [
                        "{short_name}"
                    ],
                    "itrf_xyz": [
                        {x},
                        {y},
                        {z}
                    ]
                }}
                """
            )
            try:
                obs = get_observatory(full_name)
            except KeyError:
                try:
                    obs = get_observatory(short_name)
                except KeyError:
                    report["missing"].append(
                        dict(name=full_name, topo_obs_entry=topo_obs_entry)
                    )
                    continue

            loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
            oloc = obs.earth_location_itrf()
            d = earth_location_distance(loc, oloc)
            if d > 1 * u.m:
                report["different"].append(
                    dict(
                        name=full_name,
                        t2_short_name=short_name,
                        t2=loc.to_geodetic(),
                        pint=oloc.to_geodetic(),
                        topo_obs_entry=topo_obs_entry,
                        pint_name=obs.name,
                        pint_tempo_code=obs.tempo_code
                        if hasattr(obs, "tempo_code")
                        else "",
                        pint_aliases=obs.aliases,
                        position_difference=d,
                        pint_origin=obs.origin,
                    )
                )

            # Check whether TEMPO alias - first two letters - works and is distinct from others?
            # Check all t2 aliases also work for PINT?
            # Check ITOA code?
            # Check time corrections?
    return report


def compare_tempo_obsys_dat(tempodir=None):
    """Read a tempo obsys.dat file and compare with PINT.

    Produces a report including lines that can be added to PINT's
    observatories.json to add any observatories unknown to PINT.

    Parameters
    ==========
    tempodir : str, optional
        Path to the TEMPO runtime dir; if not provided, look in the
        TEMPO environment variable.

    Returns
    =======
    dict
        The dictionary has two entries, under the keys "different" and "missing"; each is
        a list of observatories found in the TEMPO files that disagree with what PINT
        expects. Each entry in these lists is again a dict, with various properties of the
        observatory, including a line that might be suitable for starting an entry in the
        PINT observatory list.
    """
    if tempodir is None:
        tempodir = os.getenv("TEMPO")
        if tempodir is None:
            raise ValueError(
                "TEMPO directory not provided and TEMPO environment variable not set"
            )
    filename = os.path.join(tempodir, "obsys.dat")

    report = defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            try:
                line_io = StringIO(line)
                x = float(line_io.read(15))
                y = float(line_io.read(15))
                z = float(line_io.read(15))
                line_io.read(2)
                icoord = line_io.read(1).strip()
                icoord = int(icoord) if icoord else 0
                line_io.read(2)
                obsnam = line_io.read(20).strip().lower()
                tempo_code = line_io.read(1)
                tempo_code = tempo_code if tempo_code != "-" else ""
                line_io.read(2)
                itoa_code = line_io.read(2).strip()
            except ValueError:
                raise ValueError(f"unrecognized line '{line}'")
            if icoord:
                loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
            else:

                def convert_angle(x):
                    s = np.sign(x)
                    x = np.abs(x)
                    return s * (
                        (x // 10000) * u.deg
                        + ((x % 10000) // 100) * u.arcmin
                        + (x % 100) * u.arcsec
                    )

                loc = EarthLocation.from_geodetic(
                    -convert_angle(y), convert_angle(x), z * u.m
                )
                x, y, z = (a.to_value(u.m) for a in loc.to_geocentric())
            name = obsnam.replace(" ", "_")
            topo_obs_entry = textwrap.dedent(
                f"""
                "{name}": {{
                    "itrf_xyz": [
                        {x},
                        {y},
                        {z}
                    ],
                    "tempo_code": "{tempo_code}",
                    "itoa_code": "{itoa_code}"
                }}
                """
            )
            try:
                obs = get_observatory(itoa_code)
            except KeyError:
                try:
                    obs = get_observatory(tempo_code)
                except KeyError:
                    report["missing"].append(
                        dict(
                            name=obsnam,
                            itoa_code=itoa_code,
                            tempo_code=tempo_code,
                            topo_obs_entry=topo_obs_entry,
                        )
                    )
                    continue

            oloc = obs.earth_location_itrf()
            d = earth_location_distance(loc, oloc)
            if d > 1 * u.m:
                report["different"].append(
                    dict(
                        name=obsnam,
                        pint_name=obs.name,
                        pint_tempo_code=obs.tempo_code
                        if hasattr(obs, "tempo_code")
                        else "",
                        pint_aliases=obs.aliases,
                        itoa_code=itoa_code,
                        tempo_code=tempo_code,
                        tempo=loc.to_geodetic(),
                        pint=oloc.to_geodetic(),
                        position_difference=d,
                        pint_origin=obs.origin,
                    )
                )

            # Check whether TEMPO alias - first two letters - works and is distinct from others?
            # Check all t2 aliases also work for PINT?
            # Check ITOA code?
            # Check time corrections?
    return report


def list_last_correction_mjds():
    """Print out a list of the last MJD each clock correction is good for.

    Each observatory lists the clock files it uses and their last dates,
    and a combined last date for the observatory. The last date for the
    observatory is also limited by the date ranges covered by GPS and BIPM
    tables, if appropriate.

    Observatories for which PINT doesn't know how to find the clock corrections
    are not listed. Observatories for which PINT knows where the clock correction
    should be but can't find it are listed as MISSING.
    """
    for n in Observatory.names():
        o = get_observatory(n)
        m = o.last_clock_correction_mjd()
        try:
            print(f"{n:<24} {Time(m, format='mjd').iso}")
        except (ValueError, TypeError):
            print(f"{n:<24} MISSING")
        if not hasattr(o, "_clock"):
            continue
        for c in o._clock:
            try:
                print(
                    f"    {c.friendly_name:<20}"
                    f" {Time(c.last_correction_mjd(), format='mjd').iso}"
                )
            except (ValueError, TypeError):
                print(f"    {c.friendly_name:<20} MISSING")


def update_clock_files(bipm_versions=None):
    """Obtain an up-to-date version of all clock files.

    This up-to-date version will be stored in the Astropy cache;
    you can then export or otherwise preserve the Astropy cache
    so it can be pre-loaded on systems that might not have
    network access.

    This updates only the clock files that PINT knows how to use. To
    grab everything in the repository you can use
    :func:`pint.observatory.global_clock_corrections.update_all`.

    Parameters
    ----------
    bipm_versions : list of str or None
        Include these versions of the BIPM TAI to TT clock corrections
        in addition to whatever is in use. Typical values look like
        "BIPM2019".
    """
    # FIXME: allow forced downloads for non-expired files
    # FIXME: what to do about GPS and BIPM files?

    if bipm_versions is not None:
        o = get_observatory("arecibo")
        for v in bipm_versions:
            o._load_bipm_clock(v)

    t = Time.now()
    for n in Observatory.names():
        o = get_observatory(n)
        if not hasattr(o, "clock_file"):
            continue
        try:
            o.clock_corrections(t, limits="error")
        except ClockCorrectionOutOfRange:
            pass
        except NoClockCorrections:
            log.info(f"Observatory {n} has no clock corrections")


# Both topo_obs and special_locations need this
def find_clock_file(
    name,
    format,
    bogus_last_correction=False,
    url_base=None,
    clock_dir=None,
    valid_beyond_ends=False,
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
    valid_beyond_ends : bool
        If False, emit a warning or exception when evaluating the clock file past
        the ends of the data it contains.

    Returns
    -------
    ClockFile
    """
    # Avoid import loop
    from pint.observatory.clock_file import ClockFile, GlobalClockFile
    from pint.observatory.global_clock_corrections import Index

    if name == "":
        raise ValueError("No filename supplied to find_clock_file")
    if clock_dir is None or str(clock_dir).upper() == "PINT":
        # Don't try loading it from a specific path
        p = None
    elif str(clock_dir).lower() == "tempo":
        if "TEMPO" not in os.environ:
            raise NoClockCorrections(
                f"TEMPO environment variable not set but clock file {name} "
                f"is supposed to be in the directory it points to"
            )
        p = Path(os.environ["TEMPO"]) / "clock" / name
    elif str(clock_dir).lower() == "tempo2":
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
        log.info("Loading clock file {p} from specified location")
        return ClockFile.read(
            p,
            format=format,
            bogus_last_correction=bogus_last_correction,
            friendly_name=name,
            valid_beyond_ends=valid_beyond_ends,
        )

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
                valid_beyond_ends=valid_beyond_ends,
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
            valid_beyond_ends=valid_beyond_ends,
        )
    loc = Path(runtimefile(name))
    if loc.exists():
        local_clock = ClockFile.read(
            loc,
            format=format,
            bogus_last_correction=bogus_last_correction,
            friendly_name=name,
            valid_beyond_ends=valid_beyond_ends,
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
        raise NoClockCorrections(f"No clock file for {name}")
