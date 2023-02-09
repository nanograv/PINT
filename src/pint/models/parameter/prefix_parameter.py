from pint.models.parameter.pair_parameter import pairParameter
from pint.models.parameter.parameter import (
    AngleParameter,
    MJDParameter,
    boolParameter,
    floatParameter,
    strParameter,
)
from pint.utils import split_prefixed_name


class prefixParameter:
    """Families of parameters identified by a prefix like ``DMX_0123``.

    Creating a ``prefixParameter`` is like creating a normal parameter, except that the
    name should be in the format of prefix and index. For example, ``DMX_0001`` or
    ``F22``. Appropriate units will be inferred.

    To create a prefix parameter with the same prefix but different index, just
    use the :meth:`pint.models.parameter.prefixParameter.new_param` method. It will return a new ``prefixParameter`` with the
    same setup but a new index. Some  units and descriptions will
    be changed once the index has been changed. The new parameter will not inherit the ``frozen`` status of its parent by default.  In order to get the right units
    and description, ``.unit_template`` and ``.description_template`` should be provided. If
    not the new prefix parameter will use the same units and description with
    the old one. A typical description and units template is like::

        >>> description_template = lambda x: 'This is the description of parameter %d'%x
        >>> unit_template = lambda x: 'second^%d'%x

    Although it is best to avoid using lambda functions

    Parameters
    ----------
    parameter_type : str, optional
        Example parameter class template for quantity and value setter
    name : str optional
        The name of the parameter. It has to be in the format of prefix + index.
    value
        Initial parameter value
    units : str, optional
        Units that the value is expressed in
    unit_template : callable
        The unit template for prefixed parameter
    description : str, optional
        Description for the parameter
    description_template : callable
        Description template for prefixed parameters
    prefix_aliases : list of str, optional
        Alias for the prefix
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool
        Whether derivatives with respect to this parameter make sense.
    parameter_type : str, optional
        Example parameter class template for quantity and value setter
    long_double : bool, optional
        Set float type quantity and value in numpy long doubles.
    time_scale : str, optional
        Time scale for MJDParameter class.
    """

    def __init__(
        self,
        parameter_type="float",
        name=None,
        value=None,
        units=None,
        unit_template=None,
        description=None,
        description_template=None,
        uncertainty=None,
        frozen=True,
        continuous=True,
        prefix_aliases=None,
        long_double=False,
        unit_scale=False,
        scale_factor=None,
        scale_threshold=None,
        time_scale="utc",
        **kwargs,
    ):
        # Split prefixed name, if the name is not in the prefixed format, error
        # will be raised
        self.name = name
        self.prefix, self.idxfmt, self.index = split_prefixed_name(name)
        # Type identifier
        self.type_mapping = {
            "float": floatParameter,
            "str": strParameter,
            "bool": boolParameter,
            "mjd": MJDParameter,
            "angle": AngleParameter,
            "pair": pairParameter,
        }
        self.parameter_type = parameter_type
        try:
            self.param_class = self.type_mapping[self.parameter_type.lower()]
        except KeyError as e:
            raise ValueError(f"Unknown parameter type '{parameter_type}' ") from e

        # Set up other attributes in the wrapper class
        self.unit_template = unit_template
        self.description_template = description_template
        input_units = units
        input_description = description
        self.prefix_aliases = [] if prefix_aliases is None else prefix_aliases
        # set templates, the templates should be a named function and input is
        # the index of prefix parameter.

        # Set the description and units for the parameter composition.
        if self.unit_template is not None:
            real_units = self.unit_template(self.index)
        else:
            real_units = input_units
        if self.description_template is not None:
            real_description = self.description_template(self.index)
        else:
            real_description = input_description
        aliases = [pa + self.idxfmt for pa in self.prefix_aliases]
        self.long_double = long_double
        # initiate parameter class
        self.param_comp = self.param_class(
            name=self.name,
            value=value,
            units=real_units,
            description=real_description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=continuous,
            aliases=aliases,
            long_double=long_double,
            time_scale=time_scale,
            unit_scale=unit_scale,
            scale_factor=scale_factor,
            scale_threshold=scale_threshold,
        )
        self.is_prefix = True
        self.time_scale = time_scale

    @property
    def repeatable(self):
        return self.param_comp.repeatable

    @property
    def units(self):
        return self.param_comp.units

    @units.setter
    def units(self, unt):
        self.param_comp.units = unt

    @property
    def quantity(self):
        return self.param_comp.quantity

    @quantity.setter
    def quantity(self, qnt):
        self.param_comp.quantity = qnt

    @property
    def value(self):
        return self.param_comp.value

    @value.setter
    def value(self, val):
        self.param_comp.value = val

    @property
    def uncertainty(self):
        return self.param_comp.uncertainty

    @uncertainty.setter
    def uncertainty(self, ucty):
        self.param_comp.uncertainty = ucty

    @property
    def uncertainty_value(self):
        return self.param_comp.uncertainty_value

    @uncertainty_value.setter
    def uncertainty_value(self, val):
        self.param_comp.uncertainty_value = val

    @property
    def prior(self):
        return self.param_comp.prior

    @prior.setter
    def prior(self, p):
        self.param_comp.prior = p

    @property
    def aliases(self):
        return self.param_comp.aliases

    @aliases.setter
    def aliases(self, a):
        self.param_comp.aliases = a

    @property
    def use_alias(self):
        return self.param_comp.use_alias

    @use_alias.setter
    def use_alias(self, a):
        self.param_comp.use_alias = a

    @property
    def continuous(self):
        return self.param_comp.continuous

    @continuous.setter
    def continuous(self, val):
        self.param_comp.continuous = val

    @property
    def frozen(self):
        return self.param_comp.frozen

    @frozen.setter
    def frozen(self, val):
        self.param_comp.frozen = val

    @property
    def description(self):
        return self.param_comp.description

    @description.setter
    def description(self, val):
        self.param_comp.description = val

    @property
    def special_arg(self):
        return self.param_comp.special_arg

    def __repr__(self):
        return self.param_comp.__repr__()

    def from_parfile_line(self, line):
        return self.param_comp.from_parfile_line(line)

    def prior_pdf(self, value=None, logpdf=False):
        return self.param_comp.prior_pdf(value, logpdf)

    def str_quantity(self, quan):
        return self.param_comp.str_quantity(quan)

    def _print_uncertainty(self, uncertainty):
        return str(uncertainty.to(self.units).value)

    def name_matches(self, name):
        return self.param_comp.name_matches(name)

    def as_parfile_line(self, format="pint"):
        return self.param_comp.as_parfile_line(format=format)

    def help_line(self):
        return self.param_comp.help_line()

    def prefix_matches(self, prefix):
        return (prefix == self.prefix) or (prefix in self.prefix_aliases)

    def new_param(self, index, inheritfrozen=False):
        """Get one prefix parameter with the same type.

        Parameters
        ----------
        index : int
            index of prefixed parameter.
        inheritfrozen : bool, optional
            whether or not the parameter should inherit the "frozen" status of the base parameter

        Returns
        -------
        A prefixed parameter with the same type of instance.
        """

        new_name = self.prefix + format(index, f"0{len(self.idxfmt)}")
        kws = {
            key: getattr(self, key)
            for key in [
                "units",
                "unit_template",
                "description",
                "description_template",
                "frozen",
                "continuous",
                "prefix_aliases",
                "long_double",
                "time_scale",
                "parameter_type",
            ]
            if hasattr(self, key) and (key != "frozen" or inheritfrozen)
        }
        return prefixParameter(name=new_name, **kws)
