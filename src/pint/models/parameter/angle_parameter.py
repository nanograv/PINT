import numbers

import numpy as np
import astropy.units as u
from astropy.coordinates.angles import Angle

from pint import pint_units
from pint.models.parameter.param_base import Parameter
from pint.pulsar_mjd import str2longdouble


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
