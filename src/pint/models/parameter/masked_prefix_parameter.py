from pint.models.parameter.mask_parameter import maskParameter
from pint.models.parameter.prefix_parameter import prefixParameter
from pint.utils import split_masked_prefixed_name


class MaskedPrefixParameter(prefixParameter):
    def __init__(
        self,
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
        (
            self.prefix,
            (self.prefix_idxfmt, self.mask_idxfmt),
            (self.prefix_index, self.mask_index),
        ) = split_masked_prefixed_name(name)
        self.parameter_type = "mask"
        self.param_class = maskParameter

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
