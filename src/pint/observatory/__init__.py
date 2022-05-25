"""Machinery to support PINT's list of observatories."""

import sys
import os
import textwrap
from collections import defaultdict
from io import StringIO
import warnings

import astropy.constants as const
import astropy.coordinates
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger as log

import pint.solar_system_ephemerides as sse
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
    "compare_t2_observatories_dat",
    "compare_tempo_obsys_dat",
]

# The default BIPM to use if not explicitly specified
# This should be the most recent BPIM file in the datafiles directory
bipm_default = "BIPM2019"


class Observatory:
    """Observatory locations and related site-dependent properties

    For example, TOA time scales, clock corrections.
    Any new Observtory that is declared will be automatically added to
    a registry that is keyed on observatory name.  Aside from their initial
    declaration (for examples, see pint/observatory/observatories.py),
    Observatory instances should generally be accessed only via the
    Observatory.get() function.  This will query the registry based on
    observatory name (and any defined aliases).  A list of all registered
    names can be returned via Observatory.names().
    """

    # This is a dict containing all defined Observatory instances,
    # keyed on standard observatory name.
    _registry = {}

    # This is a dict mapping any defined aliases to the corresponding
    # standard name.
    _alias_map = {}

    def __new__(cls, name, *args, **kwargs):
        # Generates a new Observtory object instance, and adds it
        # it the registry, using name as the key.  Name must be unique,
        # a new instance with a given name will over-write the existing
        # one only if overwrite=True
        obs = super().__new__(cls)
        if name.lower() in cls._registry:
            if "overwrite" in kwargs and kwargs["overwrite"]:
                log.warning(
                    "Observatory '%s' already present; overwriting..." % name.lower()
                )

                cls._register(obs, name)
                return obs
            else:
                raise ValueError(
                    "Observatory '%s' already present and overwrite=False"
                    % name.lower()
                )
        cls._register(obs, name)
        return obs

    def __init__(self, name, aliases=None):
        if aliases is not None:
            Observatory._add_aliases(self, aliases)

    @classmethod
    def _register(cls, obs, name):
        """Add an observatory to the registry using the specified name
        (which will be converted to lower case).  If an existing observatory
        of the same name exists, it will be replaced with the new one.
        The Observatory instance's name attribute will be updated for
        consistency."""
        cls._registry[name.lower()] = obs
        obs._name = name.lower()

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
        for o in cls._registry.values():
            obs_aliases = []
            for alias, name in cls._alias_map.items():
                if name == o.name:
                    obs_aliases.append(alias)
            o._aliases = obs_aliases

    @classmethod
    def names(cls):
        """List all observatories known to PINT."""
        return cls._registry.keys()

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
        import pint.observatory.observatories  # noqa
        import pint.observatory.special_locations  # noqa

        if name == "":
            raise KeyError("No observatory name or code provided")

        # Be case-insensitive
        name = name.lower()
        # First see if name matches
        if name in cls._registry.keys():
            return cls._registry[name]
        # Then look for aliases
        if name in cls._alias_map.keys():
            return cls._registry[cls._alias_map[name]]
        # Then look in astropy
        log.warning(
            "Observatory name '%s' is not present in PINT observatory list; searching astropy..."
            % name
        )
        # the name was not found in the list of standard PINT observatories
        # see if we can it from astropy
        try:
            site_astropy = astropy.coordinates.EarthLocation.of_site(name)
        except astropy.coordinates.errors.UnknownSiteException:
            # turn it into the same error type as PINT would have returned
            raise KeyError("Observatory name '%s' is not defined" % name)

        # we need to import this here rather than up-top because of circular import issues
        from pint.observatory.topo_obs import TopoObs

        obs = TopoObs(
            name,
            itrf_xyz=[site_astropy.x.value, site_astropy.y.value, site_astropy.z.value],
            # add in metadata from astropy
            origin="astropy: '%s'" % site_astropy.info.meta["source"],
        )
        # add to registry
        cls._register(obs, name)
        return cls._registry[name]

        # Nothing matched, raise an error
        raise KeyError("Observatory name '%s' is not defined" % name)

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
        positions. For moving observaties (e.g. spacecraft), it
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
        """Given an array-valued Time, return the clock corrections
        as a numpy array, with units.  These values are to be added to the
        raw TOAs in order to refer them to the timescale specified by
        self.timescale."""
        # TODO this and derived methods should be changed to accept a TOA
        # table in addition to Time objects.  This will allow access to extra
        # TOA metadata which may be necessary in some cases.
        raise NotImplementedError

    def last_clock_correction_mjd(self):
        """Return the MJD of the last available clock correction.

        Returns ``np.inf`` if no clock corrections are relevant.
        """
        return np.inf

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
        # Check the method. This pattern is from numpy minize
        if callable(method):
            meth = "_custom"
        else:
            meth = method.lower()
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
            raise ValueError("Unknown method '%s'." % method)

    def _get_TDB_default(self, t, ephem):
        return t.tdb

    def _get_TDB_ephem(self, t, ephem):
        """Read the ephem TDB-TT column.

        This column is provided by DE4XXt version of ephemeris.
        """
        raise NotImplementedError

    def posvel(self, t, ephem, group=None):
        """Return observatory position and velocity for the given times.

        Postion is relative to solar system barycenter; times are
        (astropy array-valued Time objects).
        """
        # TODO this and derived methods should be changed to accept a TOA
        # table in addition to Time objects.  This will allow access to extra
        # TOA metadata which may be necessary in some cases.
        raise NotImplementedError


def get_observatory(
    name, include_gps=True, include_bipm=True, bipm_version=bipm_default
):
    """Convenience function to get observatory object with options.

    This function will simply call the ``Observatory.get`` method but
    will manually set options after the method is called.

    If the observatory is not present in the PINT list, will fallback to astropy

    Parameters
    ----------
    name : str
        The name of the observatory
    include_gps : bool, optional
        Set False to disable UTC(GPS)->UTC clock correction.
    include_bipm : bool, optional
        Set False to disable TAI TT(BIPM) clock correction.
    bipm_version : str, optional
        Set the version of TT BIPM clock correction files.

    .. note:: This function can and should be expanded if more clock
        file switches/options are added at a public API level.

    """
    site = Observatory.get(name)
    site.include_gps = include_gps
    site.include_bipm = include_bipm
    site.bipm_version = bipm_version
    return site


def earth_location_distance(loc1, loc2):
    return (
        sum((u.Quantity(loc1.to_geocentric()) - u.Quantity(loc2.to_geocentric())) ** 2)
    ) ** 0.5


def compare_t2_observatories_dat(t2dir=None):
    """Read a tempo2 observatories.dat file and compare with PINT

    Produces a report including lines that can be added to PINT's
    observatories.py to add any observatories unknown to PINT.

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
            except ValueError:
                raise ValueError(f"unrecognized line '{line}'")
            x, y, z = float(x), float(y), float(z)
            full_name, short_name = full_name.lower(), short_name.lower()
            topo_obs_entry = textwrap.dedent(
                f"""
                TopoObs(
                    name='{full_name}',
                    aliases=['{short_name}'],
                    itrf_xyz=[{x}, {y}, {z}],
                )
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
    """Read a tempo obsys.dat file and compare with PINT

    Produces a report including lines that can be added to PINT's
    observatories.py to add any observatories unknown to PINT.

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
        for line in f.readlines():
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
            topo_obs_entry = textwrap.dedent(
                f"""
                TopoObs(
                    name='{obsnam.replace(" ","_")}',
                    tempo_code='{tempo_code}',
                    itoa_code='{itoa_code}',
                    itrf_xyz=[{x}, {y}, {z}],
                )
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


def check_for_new_clock_files_in_tempo12_repos(update_download=True, show_diff=100):
    """Try to determine whether PINT's clock files are up to date.

    This iterates through all observatories for which PINT has clock corrections
    and checks the clock corrections PINT is using against those in the
    TEMPO/TEMPO2 repositories. Ones that differ are reported, along with
    a little context.

    The web versions are actually downloaded into the astropy cache, so
    optional automated updates would be possible.

    This function is still a prototype.

    Parameters
    ----------
    update_download : bool
        If True, download new versions even if there are versions in the
        astropy cache. (they might be old.)
    show_diff : int
        Show the difference between local and remote files if it is less than
        this many lines. (Huge diffs are probably just more correction data.)
    """
    import collections
    import difflib
    import astropy.utils.data

    # Importing this module triggers loading all observatories
    import pint.observatory.observatories
    import pint.observatory.topo_obs
    import pint.observatory.special_locations

    # Ensure all observatories are loaded and warnings are emitted
    clock_files = collections.defaultdict(list)
    for a, o in Observatory._registry.items():
        if not hasattr(o, "clock_file"):
            log.debug(f"Skipping bogus clock file for observatory {a}")
            continue
        try:
            # Ensure clock corrections are loaded and suppress some warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o.clock_corrections(Time(99999, format="mjd"), limits="error")
        except RuntimeError:
            pass
        if o.clock_file is None or o.clock_file == "time.dat":
            log.debug(f"Skipping bogus clock file for observatory {a}: {o.clock_file}")
            continue
        if isinstance(o.clock_fullpath, str):
            cfs = [o.clock_fullpath]
        else:
            cfs = o.clock_fullpath
        for clock_file in cfs:
            if not os.path.isfile(clock_file):
                log.debug(
                    f"Skipping unreadable clock file for observatory {a}: {clock_file}"
                )
                continue
            clock_files[clock_file].append(o)

    tempo_repo = (
        "https://sourceforge.net/p/tempo/tempo/ci/master/tree/clock/{}?format=raw"
    )
    # tempo_repo = "https://raw.githubusercontent.com/nanograv/tempo/master/clock/"
    tempo2_repo = "https://bitbucket.org/psrsoft/tempo2/raw/master/T2runtime/clock/{}"
    for clock_file, obs in clock_files.items():
        names = [o.name for o in obs]
        if len(names) == 1:
            names = names[0]
        base = os.path.basename(clock_file)
        o = obs[0]
        f = open(clock_file).read()
        if o.clock_fmt == "tempo":
            bu = tempo_repo
        elif o.clock_fmt == "tempo2":
            bu = tempo2_repo
        else:
            raise ValueError(f"Mystery format {o.clock_fmt} for observatory {a}")
        u = bu.format(base)
        log.info(f"Downloading clock file {base} for observatory {names} from {u}")
        try:
            wfn = astropy.utils.data.download_file(
                u, cache="update" if update_download else True
            )
        except IOError as e:
            log.error(f"Unable to download {base} from {u}: {e}")
            continue
        wf = open(wfn).read()
        wfl = wf.splitlines(keepends=True)
        fl = f.splitlines(keepends=True)
        if wfl != fl:
            print(
                f"Clock file {base} has changed: {len(fl)} lines in PINT, {len(wfl)} on web"
            )
            if show_diff:
                print(" " * 4 + f"Differences:")
                diff = list(difflib.unified_diff(fl, wfl))
                if len(diff) <= show_diff:
                    for l in diff:
                        sys.stdout.write(" " * 8 + l)
                else:
                    print(" " * 8 + f"{len(diff)} lines omitted")
            print(" " * 4 + f"Observatory file is in {wfn}")
            print(" " * 8 + f"downloaded from {u}")


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
    # Importing this module triggers loading all observatories
    import pint.observatory.observatories
    import pint.observatory.topo_obs
    import pint.observatory.special_locations

    for n in Observatory.names():
        o = get_observatory(n)
        if not hasattr(o, "clock_file"):
            continue
        m = o.last_clock_correction_mjd()
        if not o.clock_file:
            continue
        try:
            print(f"{n:<24} {Time(m, format='mjd').iso}")
        except (ValueError, TypeError):
            print(f"{n:<24} MISSING")
        for c in o._clock:
            try:
                print(
                    f"    {os.path.basename(c.filename):<20} {Time(c.last_correction_mjd(), format='mjd').iso}"
                )
            except (ValueError, TypeError):
                print(f"    {os.path.basename(c.filename):<20} MISSING")
