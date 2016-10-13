# observatory.py
# Base class for PINT observatories

class Observatory(object):

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
        # one.  This could be changed to give a warning or error.
        # Will also add any aliases passed via the aliases kwarg to the
        # alias registry.  An alias can only map to a single observatory,
        # re-use of an alias will override previously defined ones.
        # Perhaps this should be checked as well.  Might also want to check
        # that no aliases are already defined as observatory names.
        obs = super(Observatory,cls).__new__(cls,name,*args,**kwargs)
        cls._registry[name] = obs
        if 'aliases' in kwargs.keys():
            for a in kwargs['aliases']: cls._alias_map[a] = name
        return obs

    def __init__(self,name,aliases=[]):
        self._name = name
        self._aliases = aliases

    ### Note, name and aliases are not currently intended to be changed
    ### after initialization.  If we want to allow this, we could add 
    ### setter methods that update the registries appropriately.

    @property
    def name(self): return self._name

    @property
    def aliases(self): return self._aliases

    @classmethod
    def get(cls,name):
        """This will return the Observatory instance for the specified name.
        If the name has not been defined, an error will be raised.  Apart
        from the initial observatory definitions, this is in general the 
        only way Observatory objects should be accessed."""
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

    def earth_location(self):
        """Returns observatory geocentric position as an astropy 
        EarthLocation object.  For observatories where this is not
        relevant, None can be returned."""
        return None

    def timescale(self):
        """Returns the timescale that TOAs from this observatory will be in,
        once any clock corrections have been applied.  This should be a 
        string suitable to be passed directly to the scale argument of
        astropy.time.Time()."""
        raise NotImplementedError

    def clock_corrections(self,t):
        """Given a Time (can be array-valued), return the clock corrections 
        as a numpy array, with units.  These values are to be added to the 
        raw TOAs in order to refer them to the timescale specified by
        self.timescale()."""
        # TODO: accept simple mjd float arrays also?
        # TODO: provide a way to accept additional metadata that may be
        # needed to calculate these (for example, different corrections
        # for different instruments at an observatory).
        # TODO: return a TimeDelta rather than a quantity?  Does it matter?
        # TODO: provide a method that will adjust the TOAs directly?  This
        # could help address the metadata question above
        raise NotImplementedError

    def posvel(self,t):
        """Returns observatory position and velocity relative to solar system
        barycenter for the given times (astropy Time objects)."""
        # TODO: accept simple float mjds?
        raise NotImplementedError

