import numpy as np

from pint.models.parameter.parameter import floatParameter


class pairParameter(floatParameter):
    """Parameter type for parameters that need two input floats.

    One example are WAVE parameters.

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
    aliases : str, optional
        List of aliases for the current parameter

    """

    def __init__(
        self,
        name=None,
        index=None,
        value=None,
        long_double=False,
        units=None,
        description=None,
        uncertainty=None,
        frozen=True,
        continuous=False,
        aliases=[],
        **kwargs,
    ):
        self.index = index
        name_param = name
        self.origin_name = name
        self.prefix = self.origin_name

        self.prefix_aliases = aliases

        super().__init__(
            name=name_param,
            value=value,
            units=units,
            description=description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=continuous,
            aliases=aliases,
            long_double=long_double,
            **kwargs,
        )

        self.is_prefix = True

    def name_matches(self, name):
        if super().name_matches(name):
            return True
        name_idx = name + str(self.index)
        return super().name_matches(name_idx)

    def from_parfile_line(self, line):
        """Read mask parameter line (e.g. JUMP).

        Notes
        -----
        The accepted format:
            NAME value_a value_b

        """
        try:
            k = line.split()
            name = k[0].upper()
        except IndexError:
            return False
        # Test that name matches
        if not self.name_matches(name):
            return False
        try:
            self.value = (k[1], k[2])
        except IndexError:
            return False
        if name != self.name:
            # FIXME: what about prefix/mask parameters?
            self.use_alias = name

        return True

    def as_parfile_line(self, format="pint"):
        quantity = self.quantity
        if self.quantity is None:
            return ""
        name = self.name if self.use_alias is None else self.use_alias
        line = "%-15s " % name
        line += "%25s" % self.str_quantity(quantity[0])
        line += " %25s" % self.str_quantity(quantity[1])

        return line + "\n"

    def new_param(self, index):
        """Create a new but same style mask parameter."""
        return pairParameter(
            name=self.origin_name,
            index=index,
            long_double=self.long_double,
            units=self.units,
            aliases=self.prefix_aliases,
        )

    def _set_quantity(self, vals):
        vals = [floatParameter._set_quantity(self, val) for val in vals]
        return vals

    def _set_uncertainty(self, vals):
        return self._set_quantity(vals)

    @property
    def value(self):
        """Return the pure value of a parameter.

        This value will associate with parameter default value, which is .units attribute.
        """
        return None if self._quantity is None else self._get_value(self._quantity)

    @value.setter
    def value(self, val):
        """Method to set .value.

        Setting .value attribute will change the .quantity attribute other than .value attribute.
        """
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

    def str_quantity(self, quan):
        """Return quantity as a string."""
        try:
            # Maybe it's a singleton quantity
            return floatParameter.str_quantity(self, quan)
        except AttributeError as e:
            # Not a quantity, let's hope it's a list of length two?
            if len(quan) != 2:
                raise ValueError(
                    f"Don't know how to print this as a pair: {quan}"
                ) from e

        v0 = quan[0].to(self.units).value
        v1 = quan[1].to(self.units).value
        if self._long_double:
            if not isinstance(v0, np.longdouble):
                raise TypeError(
                    f"Parameter {self} is supposed to contain long doubles but contains a float"
                )
            if not isinstance(v1, np.longdouble):
                raise TypeError(
                    f"Parameter {self} is supposed to contain long doubles but contains a float"
                )
        quan0 = str(v0)
        quan1 = str(v1)
        return f"{quan0} {quan1}"
