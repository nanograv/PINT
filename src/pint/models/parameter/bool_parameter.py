from pint.models.parameter.param_base import Parameter


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
