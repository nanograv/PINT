from pint.models.parameter.param_base import Parameter


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
