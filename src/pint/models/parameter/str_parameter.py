from pint.models.parameter.param_base import Parameter


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
