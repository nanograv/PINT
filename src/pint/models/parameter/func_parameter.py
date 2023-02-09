from loguru import logger as log

from pint.models.parameter.parameter import floatParameter


class funcParameter(floatParameter):
    """Parameter defined as a read-only function operating on other parameters that returns a float or long double value.

    Can access the result of the function through the ``.quantity`` attribute,
    and the value without units through the ``.value`` attribute.

    On its own this parameter will not be useful,
    but when inserted into a :class:`pint.models.timing_model.Component` object
    it can operate on any parameters within that component or others in the same
    :class:`pint.models.timing_model.TimingModel`.

    Parameters
    ----------
    name : str
        The name of the parameter.
    func : function
        Returns the desired value
    params : iterable
        List or tuple of parameter names.
        Each can optionally also be a tuple including the attribute to access (default is ``quantity``)
    units : str or astropy.units.Quantity
        Parameter default unit. Parameter .value and .uncertainty_value attribute
        will associate with the default units. If unit is dimensionless, use
        "''" as its unit.
    description : str, optional
        A short description of what this parameter means.
    inpar : bool, optional
        Whether to include in par-file printouts, or to comment out
    long_double : bool, optional, default False
        A flag specifying whether value is float or long double.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Examples
    -------
    >>> import pint.models.parameter
    >>> p = pint.models.parameter.funcParameter(
            name="AGE",
            description="Spindown age",
            params=("F0", "F1"),
            func=lambda f0, f1: -f0 / 2 / f1,
            units="yr",
        )
    >>> m.components["Spindown"].add_param(p)
    >>> print(m.AGE)

    >>> import pint.models.parameter
    >>> import pint.derived_quantities
    >>> p2 = pint.models.parameter.funcParameter(
            name="PSREDOT",
            description="Spindown luminosity",
            params=("F0", "F1"),
            func=pint.derived_quantities.pulsar_edot,
            units="erg/s",
        )
    >>> m.components["Spindown"].add_param(p2)
    >>> print(m.PSREDOT)

    Notes
    -----
    Defining functions through ``lambda`` functions may result in unpickleable models

    Future versions may include derivative functions to calculate uncertainties.

    """

    def __init__(
        self,
        name=None,
        description=None,
        func=None,
        params=None,
        units=None,
        inpar=False,
        long_double=False,
        unit_scale=False,
        scale_factor=None,
        scale_threshold=None,
        aliases=None,
        **kwargs,
    ):
        self.paramType = "funcParameter"
        self.name = name
        self.description = description
        self._func = func
        if self._func.__name__ == "<lambda>":
            log.warning(
                f"May not be able to pickle function {self._func} in definition of funcParameter '{name}': use a named function if this is required"
            )
        self._set_params(params)
        self.units = "" if units is None else units
        self.long_double = long_double
        self.scale_factor = scale_factor
        self.scale_threshold = scale_threshold
        self._unit_scale = False
        self.unit_scale = unit_scale
        self.inpar = inpar
        self.aliases = [] if aliases is None else aliases
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

        # these should be fixed
        self.uncertainty = None
        self.frozen = True
        self.use_alias = None
        self.is_prefix = False
        self.continuous = True

        # for each parameter determine how many levels of parentage to check
        self._parentlevel = []
        self._parent = None

    def _set_params(self, params):
        """Split the input parameter list into tuples of parameter and attribute

        Parameters
        ----------
        params : : iterable
            List or tuple of parameter names.
            Each can optionally also be a tuple including the attribute to access (default is ``quantity``)
        """
        self._params = []
        self._attrs = []
        for p in params:
            if isinstance(p, str):
                self._params.append(p)
                # assume quantity
                self._attrs.append("quantity")
            else:
                self._params.append(p[0])
                self._attrs.append(p[1])

    def _get_parentage(self, max_level=2):
        """Determine parentage level for each parameter

        Parameters
        ----------
        max_level : int, optional
            Maximum parentage level to search

        Raises
        ------
        AttributeError :
            If the parameter cannot be located in any parent object
        """
        if self._parent is None:
            return
        self._parentlevel = []
        for i, p in enumerate(self._params):
            parent = self._parent
            for _ in range(max_level):
                if hasattr(parent, p):
                    self._parentlevel.append(parent)
                    break
                if hasattr(parent, "_parent"):
                    parent = getattr(parent, "_parent")
                else:
                    break
            if len(self._parentlevel) < i + 1:
                raise AttributeError(
                    f"Cannot find parameter '{p}' in parent objects of parameter '{self.name}'"
                )

    def _get(self):
        """Run the function and return the result

        Returns
        -------
        astropy.units.Quantity or None
            If any input value is ``None`` or if the parentage is not yet specified, will return ``None``
            Otherwise will return the result of the function

        """
        if self._parent is None:
            return None
        if self._parentlevel == []:
            self._get_parentage()
        args = []
        for l, p, a in zip(self._parentlevel, self._params, self._attrs):
            args.append(getattr(getattr(l, p), a))
            if args[-1] is None:
                return None
        return self._func(*args)

    @property
    def quantity(self):
        """The result of the function"""
        return self._get()

    @quantity.setter
    def quantity(self, value):
        raise AttributeError("Cannot set funcParameter")

    @property
    def value(self):
        """The result of the function without units."""
        return self._get().value if self._get() is not None else None

    @value.setter
    def value(self, value):
        raise AttributeError("Cannot set funcParameter")

    @property
    def params(self):
        """Return a list of tuples of parameter names and attributes"""
        return list(zip(self._params, self._attrs))

    @params.setter
    def params(self, params):
        self._set_params(params)

    def from_parfile_line(self, line):
        """Ignore reading from par file

        For :class:`~pint.models.parameter.funcParameter` ,
        it is for information only so is ignored on reading
        """
        return True

    def as_parfile_line(self, format="pint"):
        return (
            super().as_parfile_line(format=format)
            if self.inpar
            else f"# {super().as_parfile_line(format=format)}"
        )
