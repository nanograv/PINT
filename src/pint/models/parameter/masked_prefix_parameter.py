import numbers
import astropy.time as time
import astropy.units as u

from pint.models.parameter.param_base import parfile_formats
from pint.models.parameter.float_parameter import floatParameter
from pint.models.parameter.mask_parameter import validate_key_value, key_identifier
from pint.pulsar_mjd import str2longdouble, time_to_mjd_string
from pint.utils import split_prefixed_name


class maskedPrefixParameter:
    def __init__(
        self,
        name=None,
        mask_index=1,
        key=None,
        key_value=None,
        value=None,
        units=None,
        unit_template=None,
        description=None,
        description_template=None,
        uncertainty=None,
        frozen=True,
        **kwargs,
    ):
        self.is_prefix = True
        self.is_mask = True

        # Parameter name and indices
        # Taken from prefixParameter
        self.origin_name = name
        self._parfile_name = self.origin_name
        self.prefix, _, self.prefix_index = split_prefixed_name(name)
        self.mask_index = mask_index
        self.name = f"{self.prefix}{self.prefix_index}_{self.mask_index}"

        # Mask key and key-value
        # Taken from maskParameter
        self.key_identifier = key_identifier
        key_value = key_value if isinstance(key_value, (list, tuple)) else [key_value]
        key_value_parser = (
            self.key_identifier[key.lower()][0]
            if key is not None and key.lower() in self.key_identifier
            else str
        )
        validate_key_value(key, key_value)
        self.key = key
        self.key_value = [
            key_value_parser(k) for k in key_value
        ]  # retains string format from .par file to ensure correct data type for comparison
        self.key_value.sort()

        # Set up other attributes in the wrapper class
        # Taken from prefixParameter
        self.unit_template = unit_template
        self.description_template = description_template
        input_units = units
        input_description = description

        # Set the description and units for the parameter composition.
        # Taken from prefixParameter
        real_units = (
            self.unit_template(self.index)
            if self.unit_template is not None
            else input_units
        )
        real_description = (
            self.description_template(self.index)
            if self.description_template is not None
            else input_description
        )

        # Setting these to simple values for the time being.
        self.long_double = False
        self.time_scale = "utc"
        self.prefix_aliases = []

        # Only support floatParameter to start with.
        # The underlying parameter is accessed via inheritance in maskParameter
        # and via composition in prefixParameter. Either will work in this case,
        # but I think composition will be easier to implement.
        self.param_class = floatParameter
        self.param_comp = self.param_class(
            name=self.name,
            value=value,
            units=real_units,
            description=real_description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=True,
            aliases=self.prefix_aliases,
            long_double=self.long_double,
            time_scale=self.time_scale,
            unit_scale=None,
            scale_factor=None,
            scale_threshold=None,
        )

    def __repr__(self):
        # Taken from maskParameter
        out = f"{self.__class__.__name__}({self.name}"
        if self.key is not None:
            out += f" {self.key}"
        if self.key_value is not None:
            for kv in self.key_value:
                out += f" {str(kv)}"
        if self.quantity is not None:
            out += f" {self.param_comp.str_quantity(self.quantity)}"
        else:
            out += " UNSET"
            return out

        if self.uncertainty is not None and isinstance(self.value, numbers.Number):
            out += f" +/- {str(self.uncertainty.to(self.units))}"
        if self.units is not None:
            out += f" ({str(self.units)})"
        out += ")"

        return out

    @property
    def repeatable(self):
        # This is False in prefixParameter and True in maskParameter.
        return True

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

    def name_matches(self, name):
        # Same as in prefixParameter
        return self.param_comp.name_matches(name)

    def from_parfile_line(self, line):
        """Read mask parameter line (e.g. FDJUMP).

        Returns
        -------
        bool
            Whether the parfile line is meaningful to this class

        Notes
        -----
        The accepted format::

            NAME key key_value parameter_value
            NAME key key_value parameter_value fit_flag
            NAME key key_value parameter_value fit_flag uncertainty
            NAME key key_value parameter_value uncertainty
            NAME key key_value1 key_value2 parameter_value
            NAME key key_value1 key_value2 parameter_value fit_flag
            NAME key key_value1 key_value2 parameter_value fit_flag uncertainty
            NAME key key_value1 key_value2 parameter_value uncertainty

        where NAME is the name for this class as reported by ``self.name_matches``.
        Note that NAME is the internal PINT representation, and need not be the same as
        the parameter name as it appears in the par file. The logic for changing the
        NAME to the PINT representation is implemented in the model builder.

        This function is the same as in maskParameter.
        """
        k = line.split()
        if not k:
            return False
        # Test that name matches
        name = k[0]
        if not self.name_matches(name):
            return False

        try:
            self.key = k[1]
        except IndexError as e:
            raise ValueError(
                "{}: No key found on timfile line {!r}".format(self.name, line)
            ) from e

        key_value_info = self.key_identifier.get(self.key.lower(), (str, 1))
        len_key_v = key_value_info[1]
        if len(k) < 3 + len_key_v:
            raise ValueError(
                "{}: Expected at least {} entries on timfile line {!r}".format(
                    self.name, 3 + len_key_v, line
                )
            )

        for ii in range(len_key_v):
            if key_value_info[0] != str:
                try:
                    kval = float(k[2 + ii])
                except ValueError:
                    kval = k[2 + ii]
            else:
                kval = k[2 + ii]
            if ii > len(self.key_value) - 1:
                self.key_value.append(key_value_info[0](kval))
            else:
                self.key_value[ii] = key_value_info[0](kval)
        if len(k) >= 3 + len_key_v:
            self.value = k[2 + len_key_v]
        if len(k) >= 4 + len_key_v:
            try:
                fit_flag = int(k[3 + len_key_v])
                if fit_flag == 0:
                    self.frozen = True
                    ucty = 0.0
                elif fit_flag == 1:
                    self.frozen = False
                    ucty = 0.0
                else:
                    ucty = fit_flag
            except ValueError:
                try:
                    str2longdouble(k[3 + len_key_v])
                    ucty = k[3 + len_key_v]
                except ValueError as exc:
                    errmsg = f"Unidentified string {k[3 + len_key_v]} in"
                    errmsg += f" parfile line {k}"
                    raise ValueError(errmsg) from exc

            if len(k) >= 5 + len_key_v:
                ucty = k[4 + len_key_v]
            self.uncertainty = self.param_comp._set_uncertainty(ucty)
        return True

    def as_parfile_line(self, format="pint"):
        # This function is the same as in maskParameter.

        assert (
            format.lower() in parfile_formats
        ), "parfile format must be one of %s" % ", ".join(
            [f'"{x}"' for x in parfile_formats]
        )

        if self.quantity is None:
            return ""

        name = self.origin_name

        line = "%-15s %s " % (name, self.key)
        for kv in self.key_value:
            if isinstance(kv, time.Time):
                line += f"{time_to_mjd_string(kv)} "
            elif isinstance(kv, u.Quantity):
                line += f"{kv.value} "
            else:
                line += f"{kv} "
        line += "%25s" % self.str_quantity(self.quantity)
        if self.uncertainty is not None:
            line += " %d %s" % (0 if self.frozen else 1, str(self.uncertainty_value))
        elif not self.frozen:
            line += " 1"
        return line + "\n"
