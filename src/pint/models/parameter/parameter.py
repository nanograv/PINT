"""Timing model parameters encapsulated as objects.

Defines Parameter class for timing model parameters. These objects keep
track of values, uncertainties, and units. They can hold a variety of
types, both numeric - python floats and numpy longdoubles - and other -
string, angles, times.

These classes also contain code to allow them to read and write values
in both exact and human-readable forms, as well as detecting when they
have occurred in ``.par`` files.

One major complication is that timing models can often have variable
numbers of parameters: for example the ``DMX`` family of parameters
can have one parameter for each group of TOAs in the input, allowing
potentially very many. These are handled in two separate ways, as "prefix
parameters" (:class:`pint.models.parameter.prefixParameter`) and
"mask parameters" (:class:`pint.models.parameter.maskParameter`)
depending on how they occur in the ``.par`` and ``.tim`` files.

See :ref:`Supported Parameters` for an overview, including a table of all the
parameters PINT understands.

"""
import numbers
from warnings import warn

import astropy.time as time
import astropy.units as u
import numpy as np
from astropy.coordinates.angles import Angle
from loguru import logger as log

from pint import pint_units
from pint.models import priors
from pint.pulsar_mjd import (
    Time,
    data2longdouble,
    fortran_float,
    quantity2longdouble_withunit,
    str2longdouble,
    time_from_longdouble,
    time_to_longdouble,
    time_to_mjd_string,
)
from pint.utils import split_prefixed_name

# potential parfile formats
# in one place for consistency
parfile_formats = ["pint", "tempo", "tempo2"]


def _identity_function(x):
    """A function to just return the input argument

    A replacement for::

        lambda x: x

    which is needed below.

    Parameters
    ----------
    x

    Returns
    -------
    x
    """

    return x


class Parameter:
    """A single timing model parameter.

    Subclasses of this class can represent parameters of various types. They
    can record units, a description of the parameter's meaning, a default value
    in some cases, whether the parameter has ever been set, and they can
    keep track of whether a parameter is to be fit or not.

    Parameters can also come in families, either in the form of numbered
    :class:`~pint.models.parameter.prefixParameter` or with associated
    selection criteria in the form of
    :class:`~pint.models.parameter.maskParameter`.

    A parameter's current value will be stored at ``.quantity``, which will
    have associated units (:class:`astropy.quantity.Quantity`) or other special
    type machinery, or can also be accessed through ``.value``, which provides
    the raw value (stripped of units if applicable). Both of these can be
    assigned to to change the parameter's value. If the parameter has units,
    they will be accessible through the ``.units`` property (an
    :class:`astropy.units.Unit`). A parameter that has not been set will have
    the value None.

    Parameters also support uncertainties; these are available including units
    through the ``.uncertainty`` attribute. Parameters can also be set as
    ``.frozen=True`` to indicate that they should not be modified as part of a
    fit.

    Parameters
    ----------
    name : str, optional
        The name of the parameter.
    value : number, str, astropy.units.Quantity, or other data type or object
        The input parameter value. Quantities are accepted here, but when the
        corresponding property is read the value will never have units.
    units : str or astropy.units.Unit, optional
        Parameter default unit. Parameter .value and .uncertainty_value attribute
        will associate with the default units.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : float
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether :class:`~pint.fitter.Fitter` objects should
        adjust the value of this parameter or leave it fixed.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.
    continuous : bool, optional
        A flag specifying whether derivatives with respect to this
        parameter exist.
    use_alias : str or None
        Alias to use on write; normally whatever alias was in the par
        file it was read from

    Attributes
    ----------
    quantity : astropy.units.Quantity or astropy.time.Time or bool or int
        The parameter's value
    """

    def __init__(
        self,
        name=None,
        value=None,
        units=None,
        description=None,
        uncertainty=None,
        frozen=True,
        aliases=None,
        continuous=True,
        prior=priors.Prior(priors.UniformUnboundedRV()),
        use_alias=None,
        parent=None,
    ):
        self.name = name  # name of the parameter
        # The input parameter from parfile, which can be an alias of the parameter
        # TODO give a better name and make it easy to access.
        self._parfile_name = name
        self.units = units  # Default unit
        self.quantity = value  # The value of parameter, internal storage
        self.prior = prior

        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.continuous = continuous
        self.aliases = [] if aliases is None else aliases
        self.is_prefix = False
        self.paramType = "Not specified"  # Type of parameter. Here is general type
        self.valueType = None
        self.special_arg = []
        self.use_alias = use_alias
        self.is_mask = False
        self._parent = parent

    @property
    def quantity(self):
        """Value including units (if appropriate)."""
        return self._quantity

    @quantity.setter
    def quantity(self, val):
        """General wrapper method to set .quantity.

        For different type of
        parameters, the setter method is stored at ._set_quantity attribute.
        """
        if val is None:
            if hasattr(self, "quantity") and self.quantity is not None:
                raise ValueError("Setting an existing value to None is not allowed.")
            self._quantity = val
            return
        self._quantity = self._set_quantity(val)

    @property
    def value(self):
        """Return the value (without units) of a parameter.

        This value is assumed to be in units of ``self.units``. Upon setting, a
        a :class:`~astropy.units.Quantity` can be provided, which will be converted
        to ``self.units``.
        """
        return None if self._quantity is None else self._get_value(self._quantity)

    @value.setter
    def value(self, val):
        if val is None:
            if (
                not isinstance(self.quantity, (str, bool))
                and self._quantity is not None
            ):
                raise ValueError(
                    "Setting .value to None will lose the parameter value."
                )
            else:
                self.value = val
        self._quantity = self._set_quantity(val)

    @property
    def units(self):
        """Units associated with this parameter.

        Should be a :class:`astropy.units.Unit` object, or None if never set.
        """
        return self._units

    @units.setter
    def units(self, unt):
        # Check if this is the first time set units and check compatibility
        if hasattr(self, "quantity") and self.units is not None:
            if unt != self.units:
                wmsg = f"Parameter {self.name} default units has been "
                wmsg += f" reset to {str(unt)} from {str(self.units)}"
                log.warning(wmsg)
            try:
                if hasattr(self.quantity, "unit"):
                    self.quantity.to(unt)
            except ValueError:
                log.warning(
                    "The value unit is not compatible with"
                    " parameter units right now."
                )

        if unt is None:
            self._units = None

        # Always compare a string to pint_units.keys()
        # If search an astropy unit object with a sting list
        # If the string does not match astropy unit, astropy will guess what
        # does the string mean. It will take a lot of time.
        elif isinstance(unt, str) and unt in pint_units.keys():
            # These are special-case unit strings in in PINT
            self._units = pint_units[unt]

        else:
            # Try to use it as an astropy unit.  If this fails,
            # ValueError will be raised.
            self._units = u.Unit(unt)

        if hasattr(self, "quantity") and hasattr(self.quantity, "unit"):
            # Change quantity unit to new unit
            self.quantity = self.quantity.to(self._units)
        if hasattr(self, "uncertainty") and hasattr(self.uncertainty, "unit"):
            # Change uncertainty unit to new unit
            self.uncertainty = self.uncertainty.to(self._units)

    @property
    def uncertainty(self):
        """Parameter uncertainty value with units."""
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, val):
        if val is None:
            if hasattr(self, "uncertainty") and self.uncertainty is not None:
                raise ValueError(
                    "Setting an existing uncertainty to None is not allowed."
                )
            self._uncertainty = self._uncertainty_value = None
            return
        val = self._set_uncertainty(val)

        if val < 0:
            raise ValueError(f"Uncertainties cannot be negative but {val} was supplied")
            # self.uncertainty_value = np.abs(self.uncertainty_value)

        self._uncertainty = val.to(self.units)

    @property
    def uncertainty_value(self):
        """Return a pure value from .uncertainty.

        This will be interpreted as having units ``self.units``.
        """
        # FIXME: is this worth having when p.uncertainty.value does the same thing?
        if self._uncertainty is None:
            return None
        else:
            return self._get_value(self._uncertainty)

    @uncertainty_value.setter
    def uncertainty_value(self, val):
        if val is None:
            if (
                not isinstance(self.uncertainty, (str, bool))
                and self._uncertainty_value is not None
            ):
                log.warning(
                    "This parameter has uncertainty value. "
                    "Change it to None will lost information."
                )
            else:
                self.uncertainty_value = val
        self._uncertainty = self._set_uncertainty(val)

    def _get_value(self, quan):
        """Extract a raw value from internal representation.

        Generally just returns the internal representation, but some subclasses
        may override this to, say, convert to the correct units and then discard
        them.
        """
        return quan

    def _set_quantity(self, val):
        """Convert value to internal representation.

        Subclasses may override this to, for example, parse Fortran-format strings into
        long doubles.
        """
        return val

    def _set_uncertainty(self, val):
        """Convert value to internal representation for use in uncertainty."""
        if val != 0:
            raise NotImplementedError()

    @property
    def repeatable(self):
        return False

    @property
    def prior(self):
        """prior distribution for this parameter.

        This should be a :class:`~pint.models.priors.Prior` object describing the prior
        distribution of the quantity, for use in Bayesian fitting.
        """
        return self._prior

    @prior.setter
    def prior(self, p):
        if not isinstance(p, priors.Prior):
            raise ValueError("prior must be an instance of Prior()")
        self._prior = p

    def prior_pdf(self, value=None, logpdf=False):
        """Return the prior probability density.

        Evaluated at the current value of the parameter, or at a proposed value.

        Parameters
        ----------
        value : array-like or float, optional
            Where to evaluate the priors; should be a unitless number.
            If not provided the prior is evaluated at ``self.value``.
        logpdf : bool
            If True, return the logarithm of the PDF instead of the PDF;
            this can help with densities too small to represent in floating-point.
        """
        if value is None:
            value = self.value
        return self.prior.logpdf(value) if logpdf else self.prior.pdf(value)

    def str_quantity(self, quan):
        """Format the argument in an appropriate way as a string."""
        return str(quan)

    def _print_uncertainty(self, uncertainty):
        """Represent uncertainty in the form of a string.

        This converts the :class:`~astropy.units.Quantity` provided to the
        appropriate units, extracts the value, and converts that to a string.
        """
        return str(uncertainty.to(self.units).value)

    def __repr__(self):
        out = "{0:16s}{1:20s}".format(f"{self.__class__.__name__}(", self.name)
        if self.quantity is None:
            out += "UNSET"
            return out
        out += "{:17s}".format(self.str_quantity(self.quantity))
        if self.units is not None:
            out += f" ({str(self.units)})"
        if self.uncertainty is not None and isinstance(self.value, numbers.Number):
            out += f" +/- {str(self.uncertainty.to(self.units))}"
        out += f" frozen={self.frozen}"
        out += ")"
        return out

    def help_line(self):
        """Return a help line containing parameter name, description and units."""
        out = "%-12s %s" % (self.name, self.description)
        if self.units is not None:
            out += f" ({str(self.units)})"
        return out

    def as_parfile_line(self, format="pint"):
        """Return a parfile line giving the current state of the parameter.

        Parameters
        ----------
        format : str, optional
             Parfile output format. PINT outputs in 'tempo', 'tempo2' and 'pint'
             formats. The default format is `pint`.

        Returns
        -------
        str

        Notes
        -----
        Format differences between tempo, tempo2, and pint at [1]_

        .. [1] https://github.com/nanograv/PINT/wiki/PINT-vs.-TEMPO%282%29-par-file-changes
        """
        assert (
            format.lower() in parfile_formats
        ), "parfile format must be one of %s" % ", ".join(
            [f'"{x}"' for x in parfile_formats]
        )

        # Don't print unset parameters
        if self.quantity is None:
            return ""
        name = self.name if self.use_alias is None else self.use_alias
        # special cases for parameter names that change depending on format
        if self.name == "CHI2" and format.lower() != "pint":
            # no CHI2 for TEMPO/TEMPO2
            return ""
        elif self.name == "SWM" and format.lower() != "pint":
            # no SWM for TEMPO/TEMPO2
            return ""
        elif self.name == "A1DOT" and format.lower() != "pint":
            # change to XDOT for TEMPO/TEMPO2
            name = "XDOT"
        elif self.name == "STIGMA" and format.lower() != "pint":
            # change to VARSIGMA for TEMPO/TEMPO2
            name = "VARSIGMA"

        # standard output formatting
        line = "%-15s %25s" % (name, self.str_quantity(self.quantity))
        # special cases for parameter values that change depending on format
        if self.name == "ECL" and format.lower() == "tempo2":
            if self.value != "IERS2003":
                log.warning(
                    f"Changing ECL from '{self.value}' to 'IERS2003'; please refit for consistent results"
                )
                # change ECL value to IERS2003 for TEMPO2
                line = "%-15s %25s" % (name, "IERS2003")
        elif self.name == "NHARMS" and format.lower() != "pint":
            # convert NHARMS value to int
            line = "%-15s %25d" % (name, self.value)
        elif self.name == "KIN" and format.lower() == "tempo":
            # convert from DT92 convention to IAU
            line = "%-15s %25s" % (name, self.str_quantity(180 * u.deg - self.quantity))
            log.warning(
                "Changing KIN from DT92 convention to IAU: this will not be readable by PINT"
            )
        elif self.name == "KOM" and format.lower() == "tempo":
            # convert from DT92 convention to IAU
            line = "%-15s %25s" % (name, self.str_quantity(90 * u.deg - self.quantity))
            log.warning(
                "Changing KOM from DT92 convention to IAU: this will not be readable by PINT"
            )
        elif self.name == "DMDATA" and format.lower() != "pint":
            line = "%-15s %d" % (self.name, int(self.value))

        if self.uncertainty is not None:
            line += " %d %s" % (
                0 if self.frozen else 1,
                self._print_uncertainty(self.uncertainty),
            )
        elif not self.frozen:
            line += " 1"

        if self.name == "T2CMETHOD" and format.lower() == "tempo2":
            # comment out T2CMETHOD for TEMPO2
            line = f"#{line}"
        return line + "\n"

    def from_parfile_line(self, line):
        """Parse a parfile line into the current state of the parameter.

        Returns True if line was successfully parsed, False otherwise.

        Note
        ----
        The accepted formats:

        * NAME value
        * NAME value fit_flag
        * NAME value fit_flag uncertainty
        * NAME value uncertainty
        """
        try:
            k = line.split()
            name = k[0]
        except IndexError:
            return False
        # Test that name matches
        if not self.name_matches(name.upper()):
            return False
        if len(k) < 2:
            return False
        self.value = k[1]
        if name != self.name:
            # FIXME: what about prefix/mask parameters?
            self.use_alias = name
        if len(k) >= 3:
            try:
                # FIXME! this is not right
                fit_flag = int(k[2])
                if fit_flag == 0:
                    self.frozen = True
                    ucty = 0.0
                elif fit_flag == 1:
                    self.frozen = False
                    ucty = 0.0
                else:
                    ucty = fit_flag
            except ValueError:
                try:
                    str2longdouble(k[2])
                    ucty = k[2]
                except ValueError as e:
                    errmsg = f"Unidentified string '{k[2]}' in"
                    errmsg += " parfile line " + " ".join(k)
                    raise ValueError(errmsg) from e

            if len(k) >= 4:
                ucty = k[3]
            self.uncertainty = self._set_uncertainty(ucty)
        return True

    def add_alias(self, alias):
        """Add a name to the list of aliases for this parameter."""
        self.aliases.append(alias)

    def name_matches(self, name):
        """Whether or not the parameter name matches the provided name"""
        return (
            (name == self.name.upper())
            or (name in [x.upper() for x in self.aliases])
            or (split_prefixed_name(name) == split_prefixed_name(self.name.upper()))
        )

    def set(self, value):
        """Deprecated - just assign to .value."""
        warn(
            "The .set() function is deprecated. Set self.value directly instead.",
            category=DeprecationWarning,
        )
        self.value = value


class floatParameter(Parameter):
    """Parameter with float or long double value.

    ``.quantity`` stores current parameter value and its unit in an
    :class:`~astropy.units.Quantity`. Upon storage in ``.quantity``
    the input is converted to ``self.units``.

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : number, str, or astropy.units.Quantity
        The input parameter float value.
    units : str or astropy.units.Quantity
        Parameter default unit. Parameter .value and .uncertainty_value attribute
        will associate with the default units. If unit is dimensionless, use
        "''" as its unit.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.
    continuous : bool, optional, default True
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    long_double : bool, optional, default False
        A flag specifying whether value is float or long double.

    Example
    -------
    >>> from parameter import floatParameter
    >>> test = floatParameter(name='test1', value=100.0, units='second')
    >>> print(test)
    test1 (s) 100.0
    """

    def __init__(
        self,
        name=None,
        value=None,
        units=None,
        description=None,
        uncertainty=None,
        frozen=True,
        aliases=None,
        continuous=True,
        long_double=False,
        unit_scale=False,
        scale_factor=None,
        scale_threshold=None,
        **kwargs,
    ):
        self.long_double = long_double
        self.scale_factor = scale_factor
        self.scale_threshold = scale_threshold
        self._unit_scale = False
        if units is None:
            units = ""
        super().__init__(
            name=name,
            value=value,
            units=units,
            frozen=frozen,
            aliases=aliases,
            continuous=continuous,
            description=description,
            uncertainty=uncertainty,
        )
        self.paramType = "floatParameter"
        self.special_arg += [
            "long_double",
            "unit_scale",
            "scale_threshold",
            "scale_factor",
        ]
        self.unit_scale = unit_scale

    @property
    def long_double(self):
        """Whether the parameter has long double precision."""
        # FIXME: why not just always keep long double precision?
        return self._long_double

    @long_double.setter
    def long_double(self, val):
        """long double setter, if a floatParameter's longdouble flag has been
        changed, `.quantity` will get reset in order to get to the right data
        type.
        """
        if not isinstance(val, bool):
            raise ValueError("long_double property can only be set as boolean" " type")
        if hasattr(self, "long_double"):
            if self.long_double != val and hasattr(self, "quantity"):
                if not val:
                    log.warning(
                        "Setting floatParameter from long double to float,"
                        " precision will be lost."
                    )
                # Reset quantity to its asked type
                self._long_double = val
                self.quantity = self.quantity
        else:
            self._long_double = val

    @property
    def unit_scale(self):
        """If True, the parameter can automatically scale some values upon assignment."""
        return self._unit_scale

    @unit_scale.setter
    def unit_scale(self, val):
        self._unit_scale = val
        if self._unit_scale:
            if self.scale_factor is None:
                raise ValueError(
                    "The scale factor should be given if unit_scale"
                    " is set to be True."
                )
            if self.scale_threshold is None:
                raise ValueError(
                    "The scale threshold should be given if unit_scale"
                    " is set to be True."
                )

    def _set_quantity(self, val):
        """Convert input to floating-point format.

        accept format

        1. Astropy quantity
        2. float
        3. string

        """
        # Check long_double
        if not self._long_double:
            setfunc_with_unit = _identity_function
            setfunc_no_unit = fortran_float
        else:
            setfunc_with_unit = quantity2longdouble_withunit
            setfunc_no_unit = data2longdouble

        # First try to use astropy unit conversion
        try:
            # If this fails, it will raise UnitConversionError
            val.to(self.units)
            result = setfunc_with_unit(val)
        except AttributeError:
            # This will happen if the input value did not have units
            num_value = setfunc_no_unit(val)
            if self.unit_scale and np.abs(num_value) > np.abs(self.scale_threshold):
                # For some parameters, if the value is above a threshold, it is assumed to be in units of scale_factor
                # e.g. "PBDOT 7.2" is interpreted as "PBDOT 7.2E-12", since the scale_factor is 1E-12 and the scale_threshold is 1E-7
                log.info(
                    f"Parameter {self.name}'s value will be scaled by {str(self.scale_factor)}"
                )
                num_value *= self.scale_factor
            result = num_value * self.units

        return result

    def _set_uncertainty(self, val):
        return self._set_quantity(val)

    def str_quantity(self, quan):
        """Quantity as a string (for floating-point values)."""
        v = quan.to(self.units).value
        if self._long_double and not isinstance(v, np.longdouble):
            raise ValueError(
                "Parameter is supposed to contain long double values but contains a float"
            )
        return str(v)

    def _get_value(self, quan):
        """Convert to appropriate units and extract value."""
        if quan is None:
            return None
        elif isinstance(quan, (float, np.longdouble)):
            return quan
        else:
            return quan.to(self.units).value


class strParameter(Parameter):
    """String-valued parameter.

    ``strParameter`` is not fittable.

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : str
        The input parameter string value.
    description : str, optional
        A short description of what this parameter means.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example
    -------
    >>> from parameter import strParameter
    >>> test = strParameter(name='test1', value='This is a test',)
    >>> print(test)
    test1 This is a test
    """

    def __init__(self, name=None, value=None, description=None, aliases=None, **kwargs):
        # FIXME: where did kwargs go?
        super().__init__(
            name=name,
            value=value,
            description=description,
            frozen=True,
            aliases=aliases,
        )

        self.paramType = "strParameter"
        self.value_type = str

    def _set_quantity(self, val):
        """Convert to string."""
        return str(val)


class boolParameter(Parameter):
    """Boolean-valued parameter.

    Boolean parameters support ``1``/``0``, ``T``/``F``, ``Y``/``N``,
    ``True``/``False``, or ``Yes``/``No`` in any combination of upper and lower
    case. They always output ``Y`` or ``N`` in a par file.

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : str, bool, [0,1]
        The input parameter boolean value.
    description : str, optional
        A short description of what this parameter means.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example
    -------
    >>> from parameter import boolParameter
    >>> test = boolParameter(name='test1', value='N')
    >>> print(test)
    test1 N
    """

    def __init__(
        self,
        name=None,
        value=None,
        description=None,
        frozen=True,
        aliases=None,
        **kwargs,
    ):
        # FIXME: where did kwargs go?
        super().__init__(
            name=name,
            value=value,
            description=description,
            frozen=True,
            aliases=aliases,
        )
        self.value_type = bool
        self.paramType = "boolParameter"

    def str_quantity(self, quan):
        return "Y" if quan else "N"

    def _set_quantity(self, val):
        """Get boolean value for boolParameter class"""
        # First try strings
        try:
            if val.upper() in ["Y", "YES", "T", "TRUE"]:
                return True
            elif val.upper() in ["N", "NO", "F", "FALSE"]:
                return False
        except AttributeError:
            # Will get here on non-string types
            pass
        else:
            # String not in the list
            return bool(float(val))
        return bool(val)


class intParameter(Parameter):
    """Integer parameter values.

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : str, bool, [0,1]
        The input parameter boolean value.
    description : str, optional
        A short description of what this parameter means.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example
    -------
    >>> from parameter import intParameter
    >>> test = intParameter(name='test1', value=7)
    >>> print(test)
    test1 7
    """

    def __init__(
        self,
        name=None,
        value=None,
        description=None,
        frozen=True,
        aliases=None,
        **kwargs,
    ):
        # FIXME: where did kwargs go?
        super().__init__(
            name=name,
            value=value,
            description=description,
            frozen=True,
            aliases=aliases,
        )
        self.value_type = int
        self.paramType = "intParameter"

    def _set_quantity(self, val):
        """Convert a string or other value to an integer."""
        if isinstance(val, str):
            try:
                ival = int(val)
            except ValueError as e:
                fval = float(val)
                ival = int(fval)
                if ival != fval and abs(fval) < 2**52:
                    raise ValueError(
                        f"Value {val} does not appear to be an integer "
                        f"but parameter {self.name} stores only integers."
                    ) from e
        else:
            ival = int(val)
            fval = float(val)
            if ival != fval and abs(fval) < 2**52:
                raise ValueError(
                    f"Value {val} does not appear to be an integer "
                    f"but parameter {self.name} stores only integers."
                )

        return ival


class MJDParameter(Parameter):
    """Parameters for MJD quantities.

    ``.quantity`` stores current parameter information in an
    :class:`astropy.time.Time` type in the format of MJD. ``.value`` returns
    the pure long double MJD value. ``.units`` is in day as default unit. Note
    that you can't make an :class:`astropy.time.Time` object just by
    multiplying a number by ``u.day``; there are complexities in constructing
    times.

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : astropy Time, str, float in mjd, str in mjd.
        The input parameter MJD value.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool, optional, default True
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.
    time_scale : str, optional, default 'tdb'
        MJD parameter time scale.

    Example
    -------
    >>> from parameter import MJDParameter
    >>> test = MJDParameter(name='test1', value='54000', time_scale='utc')
    >>> print(test)
    test1 (d) 54000.000000000000000
    """

    def __init__(
        self,
        name=None,
        value=None,
        description=None,
        uncertainty=None,
        frozen=True,
        continuous=True,
        aliases=None,
        time_scale="tdb",
        **kwargs,
    ):
        self._time_scale = time_scale
        # FIXME: where did kwargs go?
        super().__init__(
            name=name,
            value=value,
            units="MJD",
            description=description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=continuous,
            aliases=aliases,
        )
        self.value_type = time.Time
        self.paramType = "MJDParameter"
        self.special_arg += ["time_scale"]

    def str_quantity(self, quan):
        return time_to_mjd_string(quan)

    def _get_value(self, quan):
        return time_to_longdouble(quan)

    @property
    def time_scale(self):
        return self._time_scale

    @time_scale.setter
    def time_scale(self, val):
        self._time_scale = val
        mjd = self.value
        self.quantity = mjd

    @property
    def uncertainty_value(self):
        """Return a pure value from .uncertainty.

        The unit will associate with .units
        """
        if self._uncertainty is None:
            return None
        else:
            return self._uncertainty.to_value(self.units)

    @uncertainty_value.setter
    def uncertainty_value(self, val):
        if val is None:
            if (
                not isinstance(self.uncertainty, (str, bool))
                and self._uncertainty_value is not None
            ):
                log.warning(
                    "This parameter has uncertainty value. "
                    "Change it to None will lost information."
                )
            else:
                self.uncertainty_value = val
        self._uncertainty = self._set_uncertainty(val)

    def _set_quantity(self, val):
        """Value setter for MJD parameter,

        Accepted format:
        Astropy time object
        mjd float
        mjd string (in pulsar_mjd format)
        """
        if isinstance(val, numbers.Number):
            val = np.longdouble(val)
            result = time_from_longdouble(val, self.time_scale)
        elif isinstance(val, (str, bytes)):
            result = Time(val, scale=self.time_scale, format="pulsar_mjd_string")
        elif isinstance(val, time.Time):
            result = val
        else:
            raise ValueError(
                f"MJD parameter can not accept {type(val).__name__}format."
            )
        return result

    def _set_uncertainty(self, val):
        # First try to use astropy unit conversion
        try:
            # If this fails, it will raise UnitConversionError
            val.to(self.units)
            result = data2longdouble(val.value) * self.units
        except AttributeError:
            # This will happen if the input value did not have units
            result = data2longdouble(val) * self.units
        return result

    def _print_uncertainty(self, uncertainty):
        return str(self.uncertainty_value)


class AngleParameter(Parameter):
    """Parameter in angle units.

    ``.quantity`` stores current parameter information in an :class:`astropy.units.Angle` type.
    ``AngleParameter`` can accept angle format  ``{'h:m:s': u.hourangle,
    'd:m:s': u.deg, 'rad': u.rad, 'deg': u.deg}``

    Parameters
    ----------
    name : str
        The name of the parameter.
    value : angle string, float, astropy angle object
        The input parameter angle value.
    description : str, optional
        A short description of what this parameter means.
    uncertainty : number
        Current uncertainty of the value.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool, optional, default True
        A flag specifying whether phase derivatives with respect to this
        parameter exist.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Example
    -------
    >>> from parameter import AngleParameter
    >>> test = AngleParameter(name='test1', value='12:20:10', units='H:M:S')
    >>> print(test)
    test1 (hourangle) 12:20:10.00000000
    """

    def __init__(
        self,
        name=None,
        value=None,
        description=None,
        units="rad",
        uncertainty=None,
        frozen=True,
        continuous=True,
        aliases=None,
        **kwargs,
    ):
        self._str_unit = units
        self.unit_identifier = {
            "h:m:s": (u.hourangle, "h", pint_units["hourangle_second"]),
            "d:m:s": (u.deg, "d", u.arcsec),
            "rad": (u.rad, "rad", u.rad),
            "deg": (u.deg, "deg", u.deg),
        }
        # Check unit format
        if units.lower() not in self.unit_identifier.keys():
            raise ValueError(f"Unidentified unit {units}")

        self.unitsuffix = self.unit_identifier[units.lower()][1]
        self.value_type = Angle
        self.paramType = "AngleParameter"

        # FIXME: where did kwargs go?
        super().__init__(
            name=name,
            value=value,
            units=units,
            description=description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=continuous,
            aliases=aliases,
        )

    def _get_value(self, quan):
        # return Angle(x * self.unit_identifier[units.lower()][0])
        return quan.value

    def _set_quantity(self, val):
        """This function is to set value to angle parameters.

        Accepted format:
        1. Astropy angle object
        2. float
        3. number string
        """
        if isinstance(val, numbers.Number):
            result = Angle(np.longdouble(val) * self.units)
        elif isinstance(val, str):
            # FIXME: what if the user included a unit suffix?
            result = Angle(val + self.unitsuffix)
        elif hasattr(val, "unit"):
            result = Angle(val.to(self.units))
        else:
            raise ValueError(
                f"Angle parameter can not accept {type(val).__name__}format."
            )
        return result

    def _set_uncertainty(self, val):
        """This function is to set the uncertainty for an angle parameter."""
        if isinstance(val, numbers.Number):
            result = Angle(val * self.unit_identifier[self._str_unit.lower()][2])
        elif isinstance(val, str):
            result = Angle(
                str2longdouble(val) * self.unit_identifier[self._str_unit.lower()][2]
            )
        elif hasattr(val, "unit"):
            result = Angle(val.to(self.unit_identifier[self._str_unit.lower()][2]))
        else:
            raise ValueError(
                f"Angle parameter can not accept {type(val).__name__}format."
            )
        return result

    def str_quantity(self, quan):
        """This is a function to print out the angle parameter."""
        if ":" in self._str_unit:
            return quan.to_string(sep=":", precision=8)
        else:
            return quan.to_string(decimal=True, precision=15)

    def _print_uncertainty(self, unc):
        """This is a function for printing out the uncertainty"""
        if ":" not in self._str_unit:
            return unc.to_string(decimal=True, precision=20)
        angle_arcsec = unc.to(u.arcsec)
        if self.units == u.hourangle:
            # Traditionally, hourangle uncertainty is in hourangle seconds
            angle_arcsec /= 15.0
        return angle_arcsec.to_string(decimal=True, precision=20)
