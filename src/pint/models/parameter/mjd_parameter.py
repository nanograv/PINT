import numbers

import astropy.time as time
import astropy.units as u
import numpy as np
from loguru import logger as log
from uncertainties import ufloat

from pint.models.parameter.param_base import Parameter
from pint.pulsar_mjd import (
    Time,
    data2longdouble,
    time_from_longdouble,
    time_to_longdouble,
    time_to_mjd_string,
)


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

    def as_ufloats(self):
        """Return the parameter as a pair of :class:`uncertainties.ufloat`
        values representing the integer and fractional Julian dates.
        The uncertainty is carried by the latter.

        If the uncertainty is not set will be returned as 0

        Returns
        -------
        uncertainties.ufloat
        uncertainties.ufloat
        """
        value1 = self.quantity.jd1 if self.quantity is not None else 0
        value2 = self.quantity.jd2 if self.quantity is not None else 0
        error = self.uncertainty.to_value(u.d) if self.uncertainty is not None else 0
        return ufloat(value1, 0), ufloat(value2, error)
