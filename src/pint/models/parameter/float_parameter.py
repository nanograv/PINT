import numpy as np
from loguru import logger as log
from uncertainties import ufloat

from pint.models.parameter.param_base import Parameter
from pint.pulsar_mjd import (
    data2longdouble,
    quantity2longdouble_withunit,
    fortran_float,
)


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
            # For some parameters, if the value is above a threshold, it is assumed to be in units of scale_factor
            # e.g. "PBDOT 7.2" is interpreted as "PBDOT 7.2E-12", since the scale_factor is 1E-12 and the scale_threshold is 1E-7
            if self.unit_scale and np.abs(num_value) > np.abs(self.scale_threshold):
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

    def as_ufloat(self, units=None):
        """Return the parameter as a :class:`uncertainties.ufloat`

        Will cast to the specified units, or the default
        If the uncertainty is not set will be returned as 0

        Parameters
        ----------
        units : astropy.units.core.Unit, optional
            Units to cast the value

        Returns
        -------
        uncertainties.ufloat

        Notes
        -----
        Currently :class:`~uncertainties.ufloat` does not support double precision values,
        so some precision may be lost.
        """
        if units is None:
            units = self.units
        value = self.quantity.to_value(units) if self.quantity is not None else 0
        error = self.uncertainty.to_value(units) if self.uncertainty is not None else 0
        return ufloat(value, error)

    def from_ufloat(self, value, units=None):
        """Set the parameter from the value of a :class:`uncertainties.ufloat`

        Will cast to the specified units, or the default
        If the uncertainty is 0 it will be set to ``None``

        Parameters
        ----------
        value : uncertainties.ufloat
        units : astropy.units.core.Unit, optional
            Units to cast the value
        """
        if units is None:
            units = self.units
        self.quantity = value.n * units
        self.uncertainty = value.s * units if value.s > 0 else None
