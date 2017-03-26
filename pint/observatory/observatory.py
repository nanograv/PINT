# observatory.py
# Base class for PINT observatories
import six


class Observatory(object):
    """
    The Observatory class defines observatory locations and related
    site-dependent properties (eg, TOA time scales, clock corrections).
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
        # one.
        if six.PY2:
            obs = super(Observatory,cls).__new__(cls,name,*args,**kwargs)
        else:
            obs = super().__new__(cls)
        cls._register(obs, name)
        return obs

    def __init__(self,name,aliases=None):
        if aliases is not None:
            Observatory._add_aliases(self,aliases)

    @classmethod
    def _register(cls,obs,name):
        """Add an observatory to the registry using the specified name
        (which will be converted to lower case).  If an existing observatory
        of the same name exists, it will be replaced with the new one.
        The Observatory instance's name attribute will be updated for
        consistency."""
        cls._registry[name.lower()] = obs
        obs._name = name.lower()

    @classmethod
    def _add_aliases(cls,obs,aliases):
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
        return cls._registry.keys()

    ### Note, name and aliases are not currently intended to be changed
    ### after initialization.  If we want to allow this, we could add
    ### setter methods that update the registries appropriately.

    @property
    def name(self): return self._name

    @property
    def aliases(self): return self._aliases

    @classmethod
    def get(cls,name):
        """Returns the Observatory instance for the specified name/alias.
        If the name has not been defined, an error will be raised.  Aside
        from the initial observatory definitions, this is in general the
        only way Observatory objects should be accessed.  Name-matching
        is case-insensitive."""
        # Be case-insensitive
        name = name.lower()
        # First see if name matches
        if name in cls._registry.keys():
            return cls._registry[name]
        # Then look for aliases
        if name in cls._alias_map.keys():
            return cls._registry[cls._alias_map[name]]
        # Nothing matched, raise an error
        raise KeyError("Observatory name '%s' is not defined" % name)

    ### The following methods define the basic API for the Observatory class.
    ### Any which raise NotImplementedError below must be implemented in
    ### derived classes.

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

    @property
    def timescale(self):
        """Returns the timescale that TOAs from this observatory will be in,
        once any clock corrections have been applied.  This should be a
        string suitable to be passed directly to the scale argument of
        astropy.time.Time()."""
        raise NotImplementedError

    def clock_corrections(self,t):
        """Given an array-valued Time, return the clock corrections
        as a numpy array, with units.  These values are to be added to the
        raw TOAs in order to refer them to the timescale specified by
        self.timescale."""
        # TODO this and derived methods should be changed to accept a TOA
        # table in addition to Time objects.  This will allow access to extra
        # TOA metadata which may be necessary in some cases.
        raise NotImplementedError

    def posvel(self,t,ephem):
        """Returns observatory position and velocity relative to solar system
        barycenter for the given times (astropy array-valued Time objects)."""
        # TODO this and derived methods should be changed to accept a TOA
        # table in addition to Time objects.  This will allow access to extra
        # TOA metadata which may be necessary in some cases.
        raise NotImplementedError


def get_observatory(name, include_gps=True, include_bipm=True):
    """Conviencience function to get observatory object with options.

    This function will simply call the ``Observatory.get`` method but
    will manually set options after the method is called.

    Required arguments:

        name = The name of the observatory

    Optional Arguments

        include_gps = Set False to disable UTC(GPS)->UTC clock
                      correction.
        include_bipm = Set False to disable TAI TT(BIPM) clock
                      correction.

    .. note:: This function can and should be expanded if more clock
        file switches/options are added at a public API level.

    """
    site = Observatory.get(name)
    site.include_gps = include_gps
    site.include_bipm = include_bipm
    return site
