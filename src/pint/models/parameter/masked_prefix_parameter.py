from pint.models.parameter.mask_parameter import maskParameter
from pint.models.parameter.prefix_parameter import prefixParameter
from pint.utils import split_masked_prefixed_name


class MaskedPrefixParameter(prefixParameter):
    def __init__(
        self,
        prefix=None,
        prefix_index=1,
        mask_index=1,
        value=None,
        units=None,
        unit_template=None,
        description=None,
        description_template=None,
        uncertainty=None,
        frozen=True,
        continuous=True,
        long_double=False,
        unit_scale=False,
        **kwargs,
    ):
        # Split prefixed name, if the name is not in the prefixed format, error
        # will be raised
        self.prefix = prefix
        self.prefix_index = prefix_index
        self.mask_index = mask_index
        self.name = f"{prefix}{prefix_index}_{mask_index}"
        self.mask_name = f"{prefix}{prefix_index}"

        self.parameter_type = "mask"
        self.param_class = maskParameter

        # Set up other attributes in the wrapper class
        self.unit_template = unit_template
        self.description_template = description_template

        input_units = units
        input_description = description

        self.prefix_aliases = []
        # set templates, the templates should be a named function and input is
        # the index of prefix parameter.

        # Set the description and units for the parameter composition.
        if self.unit_template is not None:
            real_units = self.unit_template(self.prefix_index)
        else:
            real_units = input_units

        if self.description_template is not None:
            real_description = self.description_template(self.index)
        else:
            real_description = input_description

        self.long_double = long_double

        # initiate parameter class
        self.param_comp = self.param_class(
            name=self.mask_name,
            value=value,
            units=real_units,
            description=real_description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=continuous,
            long_double=long_double,
            unit_scale=unit_scale,
        )
        self.param_comp.name = self.name
        self.param_comp.origin_name = self.mask_name

        self.is_prefix = True
        self.is_mask = True
