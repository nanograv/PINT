"""Timing model parameters encapsulated as objects.

Defines Parameter class for timing model parameters. These objects keep
track of values, uncertainties, and units. They can hold a variety of
types, both numeric - python floats and numpy longdoubles - and other -
string, angles, times.

These classes also contain code to allow them to read and write values
in both exact and human-readable forms, as well as detecting when they
have occurred in ``.par`` files.

One major complication is that timing models can often have variable
numbers of parameters: for example the ``DMX`` family of parameters
can have one parameter for each group of TOAs in the input, allowing
potentially very many. These are handled in two separate ways, as "prefix
parameters" (:class:`pint.models.parameter.prefixParameter`) and
"mask parameters" (:class:`pint.models.parameter.maskParameter`)
depending on how they occur in the ``.par`` and ``.tim`` files.

See :ref:`Supported Parameters` for an overview, including a table of all the
parameters PINT understands.

"""
import numbers

import astropy.time as time
import astropy.units as u
import numpy as np
from astropy.coordinates.angles import Angle
from uncertainties import ufloat

from loguru import logger as log

from pint import pint_units
from pint.pulsar_mjd import (
    Time,
    data2longdouble,
    str2longdouble,
    time_from_longdouble,
    time_to_longdouble,
    time_to_mjd_string,
)
from pint.toa_select import TOASelect
from pint.utils import split_prefixed_name

from pint.models.parameter.param_base import (
    parfile_formats as _parfile_formats,
    get_observatory_name as _get_observatory_name,
    return_frequency_asquantity as _return_frequency_asquantity,
    Parameter,
)
from pint.models.parameter.float_parameter import floatParameter
from pint.models.parameter.str_parameter import strParameter
from pint.models.parameter.bool_parameter import boolParameter
from pint.models.parameter.int_parameter import intParameter


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

    def as_ufloat(self, units=None):
        """Return the parameter as a :class:`uncertainties.ufloat`

        Will cast to the specified units, or the default
        If the uncertainty is not set will be returned as 0

        Parameters
        ----------
        units : astropy.units.core.Unit, optional
            Units to cast the value

        Returns
        -------
        uncertainties.ufloat
        """
        if units is None:
            units = self.units
        value = self.quantity.to_value(units) if self.quantity is not None else 0
        error = self.uncertainty.to_value(units) if self.uncertainty is not None else 0
        return ufloat(value, error)


class maskParameter(floatParameter):
    """Parameter that applies to a subset of TOAs.

    A maskParameter applies to a subset of the TOAs, for example JUMP specifies
    that their arrival times should be adjusted by the value associated with
    this JUMP. The criterion is based on either one of the standard fields
    (telescope, frequency, et cetera) or a flag; and the selection can be on an
    exact match or on a range.

    Upon creation of a maskParameter, an index part will be added, so that the
    parameters can be distinguished within the
    :class:`pint.models.timing_model.TimingModel` object. For example::

        >>> p = maskParameter(name='JUMP', index=2, key="-fe", key_value="G430")
        >>> p.name
        'JUMP2'

    The selection criterion can be one of the parameters ``mjd``, ``freq``,
    ``name``, ``tel`` representing the required columns of a ``.tim`` file, or
    the name of a flag, starting with ``-``. If the selection criterion is
    based on ``mjd`` or ``freq`` it is expected to be accompanied by a pair of
    values that define a range; other criteria are expected to be accompanied
    by a string that is matched exactly.

    Parameters
    ----------
    name : str
        The name of the parameter.
    index : int, optional
        The index number for the prefixed parameter.
    key : str, optional
        The key words/flag for the selecting TOAs
    key_value :  list/single value optional
        The value for key words/flags. Value can take one value as a flag value.
        or two value as a range.
        e.g. ``JUMP freq 430.0 1440.0``. or ``JUMP -fe G430``
    value : float or np.longdouble, optional
        Toas/phase adjust value
    long_double : bool, optional
        Set float type quantity and value in long double
    units : str, optional
        Unit for the offset value
    description : str, optional
        Description for the parameter
    uncertainty: float or np.longdouble
        uncertainty of the parameter.
    frozen : bool, optional
        A flag specifying whether "fitters" should adjust the value of this
        parameter or leave it fixed.
    continuous : bool, optional
        Whether derivatives with respect to this parameter make sense.
    aliases : list, optional
        List of aliases for parameter name.
    """

    # TODO: Is mask parameter provide some other type of parameters other then floatParameter?

    def __init__(
        self,
        name,
        index=1,
        key=None,
        key_value=[],
        value=None,
        long_double=False,
        units=None,
        description=None,
        uncertainty=None,
        frozen=True,
        continuous=False,
        aliases=[],
    ):
        self.is_mask = True
        # {key_name: (keyvalue parse function, keyvalue length)}
        # Move this to some other places.
        self.key_identifier = {
            "mjd": (float, 2),
            "freq": (_return_frequency_asquantity, 2),
            "name": (str, 1),
            "tel": (_get_observatory_name, 1),
        }

        if not isinstance(key_value, (list, tuple)):
            key_value = [key_value]

        # Check key and key value
        key_value_parser = str
        if key is not None:
            if key.lower() in self.key_identifier:
                key_info = self.key_identifier[key.lower()]
                if len(key_value) != key_info[1]:
                    errmsg = f"key {key} takes {key_info[1]} element(s)."
                    raise ValueError(errmsg)
                key_value_parser = key_info[0]
            elif not key.startswith("-"):
                raise ValueError(
                    "A key to a TOA flag requires a leading '-'."
                    " Legal keywords that don't require a leading '-' "
                    "are MJD, FREQ, NAME, TEL."
                )
        self.key = key
        self.key_value = [
            key_value_parser(k) for k in key_value
        ]  # retains string format from .par file to ensure correct data type for comparison
        self.key_value.sort()
        self.index = index
        name_param = name + str(index)
        self.origin_name = name
        self.prefix = self.origin_name
        idx_aliases = [al + str(self.index) for al in aliases]
        self.prefix_aliases = aliases
        super().__init__(
            name=name_param,
            value=value,
            units=units,
            description=description,
            uncertainty=uncertainty,
            frozen=frozen,
            continuous=continuous,
            aliases=idx_aliases + aliases,
            long_double=long_double,
        )

        # For the first mask parameter, add name to aliases for the reading
        # first mask parameter from parfile.
        if index == 1:
            self.aliases.append(name)
        self.is_prefix = True
        self._parfile_name = self.origin_name

    def __repr__(self):
        out = f"{self.__class__.__name__}({self.name}"
        if self.key is not None:
            out += f" {self.key}"
        if self.key_value is not None:
            for kv in self.key_value:
                out += f" {str(kv)}"
        if self.quantity is not None:
            out += f" {self.str_quantity(self.quantity)}"
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
        return True

    def name_matches(self, name):
        if super().name_matches(name):
            return True
        elif self.index == 1:
            name_idx = name + str(self.index)
            return super().name_matches(name_idx)

    def from_parfile_line(self, line):
        """Read mask parameter line (e.g. JUMP).

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
            self.uncertainty = self._set_uncertainty(ucty)
        return True

    def as_parfile_line(self, format="pint"):
        assert (
            format.lower() in _parfile_formats
        ), "parfile format must be one of %s" % ", ".join(
            [f'"{x}"' for x in _parfile_formats]
        )

        if self.quantity is None:
            return ""

        name = self.origin_name if self.use_alias is None else self.use_alias

        # special cases for parameter names that change depending on format
        if name == "EFAC" and format.lower() != "pint":
            # change to T2EFAC for TEMPO/TEMPO2
            name = "T2EFAC"
        elif name == "EQUAD" and format.lower() != "pint":
            # change to T2EQUAD for TEMPO/TEMPO2
            name = "T2EQUAD"

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

    def new_param(self, index, copy_all=False):
        """Create a new but same style mask parameter"""
        return (
            maskParameter(
                name=self.origin_name,
                index=index,
                key=self.key,
                key_value=self.key_value,
                value=self.value,
                long_double=self.long_double,
                units=self.units,
                description=self.description,
                uncertainty=self.uncertainty,
                frozen=self.frozen,
                continuous=self.continuous,
                aliases=self.prefix_aliases,
            )
            if copy_all
            else maskParameter(
                name=self.origin_name,
                index=index,
                long_double=self.long_double,
                units=self.units,
                aliases=self.prefix_aliases,
            )
        )

    def select_toa_mask(self, toas):
        """Select the toas that match the mask.

        Parameters
        ----------
        toas: :class:`pint.toas.TOAs`

        Returns
        -------
        array
            An array of TOA indices selected by the mask.
        """
        if len(self.key_value) == 1:
            if not hasattr(self, "toa_selector"):
                self.toa_selector = TOASelect(is_range=False, use_hash=True)
            condition = {self.name: self.key_value[0]}
        elif len(self.key_value) == 2:
            if not hasattr(self, "toa_selector"):
                self.toa_selector = TOASelect(is_range=True, use_hash=True)
            condition = {self.name: tuple(self.key_value)}
        elif len(self.key_value) == 0:
            return np.array([], dtype=int)
        else:
            raise ValueError(
                f"Parameter {self.name} has more key values than expected.(Expect 1 or 2 key values)"
            )
        # get the table columns
        # TODO Right now it is only supports mjd, freq, tel, and flagkeys,
        # We need to consider some more complicated situation
        key = self.key[1::] if self.key.startswith("-") else self.key

        tbl = toas.table
        column_match = {"mjd": "mjd_float", "freq": "freq", "tel": "obs"}
        if (
            self.key.lower() not in column_match
        ):  # This only works for the one with flags.
            # The flags are recomputed every time. If don't
            # recompute, flags can only be added to the toa table once and then never update,
            # making it impossible to add additional jump parameters after the par file is read in (pintk)
            flag_col = [x.get(key, None) for x in tbl["flags"]]
            tbl[key] = flag_col
            col = tbl[key]
        else:
            col = tbl[column_match[key.lower()]]
        select_idx = self.toa_selector.get_select_index(condition, col)
        return select_idx[self.name]

    def compare_key_value(self, other_param):
        """Compare if the key and value are the same with the other parameter.

        Parameters
        ----------
        other_param: maskParameter
            The parameter to compare.

        Returns
        -------
        bool:
            If the key and value are the same, return True, otherwise False.

        Raises
        ------
        ValueError:
            If the parameter to compare does not have 'key' or 'key_value'.
        """
        if not hasattr(other_param, "key") and not hasattr(other_param, "key_value"):
            raise ValueError("Parameter to compare does not have `key` or `key_value`.")
        if self.key != other_param.key:
            return False
        return self.key_value == other_param.key_value


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
        except AttributeError:
            # Not a quantity, let's hope it's a list of length two?
            if len(quan) != 2:
                raise ValueError(f"Don't know how to print this as a pair: {quan}")

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


class funcParameter(floatParameter):
    """Parameter defined as a read-only function operating on other parameters that returns a float or long double value.

    Can access the result of the function through the ``.quantity`` attribute,
    and the value without units through the ``.value`` attribute.

    On its own this parameter will not be useful,
    but when inserted into a :class:`pint.models.timing_model.Component` object
    it can operate on any parameters within that component or others in the same
    :class:`pint.models.timing_model.TimingModel`.

    Parameters
    ----------
    name : str
        The name of the parameter.
    func : function
        Returns the desired value
    params : iterable
        List or tuple of parameter names.
        Each can optionally also be a tuple including the attribute to access (default is ``quantity``)
    units : str or astropy.units.Quantity
        Parameter default unit. Parameter .value and .uncertainty_value attribute
        will associate with the default units. If unit is dimensionless, use
        "''" as its unit.
    description : str, optional
        A short description of what this parameter means.
    inpar : bool, optional
        Whether to include in par-file printouts, or to comment out
    long_double : bool, optional, default False
        A flag specifying whether value is float or long double.
    aliases : list, optional
        An optional list of strings specifying alternate names that can also
        be accepted for this parameter.

    Examples
    -------
    >>> import pint.models.parameter
    >>> p = pint.models.parameter.funcParameter(
            name="AGE",
            description="Spindown age",
            params=("F0", "F1"),
            func=lambda f0, f1: -f0 / 2 / f1,
            units="yr",
        )
    >>> m.components["Spindown"].add_param(p)
    >>> print(m.AGE)

    >>> import pint.models.parameter
    >>> import pint.derived_quantities
    >>> p2 = pint.models.parameter.funcParameter(
            name="PSREDOT",
            description="Spindown luminosity",
            params=("F0", "F1"),
            func=pint.derived_quantities.pulsar_edot,
            units="erg/s",
        )
    >>> m.components["Spindown"].add_param(p2)
    >>> print(m.PSREDOT)

    Notes
    -----
    Defining functions through ``lambda`` functions may result in unpickleable models

    Future versions may include derivative functions to calculate uncertainties.

    """

    def __init__(
        self,
        name=None,
        description=None,
        func=None,
        params=None,
        units=None,
        inpar=False,
        long_double=False,
        unit_scale=False,
        scale_factor=None,
        scale_threshold=None,
        aliases=None,
        **kwargs,
    ):
        self.paramType = "funcParameter"
        self.name = name
        self.description = description
        self._func = func
        if self._func.__name__ == "<lambda>":
            log.warning(
                f"May not be able to pickle function {self._func} in definition of funcParameter '{name}': use a named function if this is required"
            )
        self._set_params(params)
        self.units = "" if units is None else units
        self.long_double = long_double
        self.scale_factor = scale_factor
        self.scale_threshold = scale_threshold
        self._unit_scale = False
        self.unit_scale = unit_scale
        self.inpar = inpar
        self.aliases = [] if aliases is None else aliases
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

        # these should be fixed
        self.uncertainty = None
        self.frozen = True
        self.use_alias = None
        self.is_prefix = False
        self.continuous = True

        # for each parameter determine how many levels of parentage to check
        self._parentlevel = []
        self._parent = None

    def _set_params(self, params):
        """Split the input parameter list into tuples of parameter and attribute

        Parameters
        ----------
        params : : iterable
            List or tuple of parameter names.
            Each can optionally also be a tuple including the attribute to access (default is ``quantity``)
        """
        self._params = []
        self._attrs = []
        for p in params:
            if isinstance(p, str):
                self._params.append(p)
                # assume quantity
                self._attrs.append("quantity")
            else:
                self._params.append(p[0])
                self._attrs.append(p[1])

    def _get_parentage(self, max_level=2):
        """Determine parentage level for each parameter

        Parameters
        ----------
        max_level : int, optional
            Maximum parentage level to search

        Raises
        ------
        AttributeError :
            If the parameter cannot be located in any parent object
        """
        if self._parent is None:
            return
        self._parentlevel = []
        for i, p in enumerate(self._params):
            parent = self._parent
            for _ in range(max_level):
                if hasattr(parent, p):
                    self._parentlevel.append(parent)
                    break
                if hasattr(parent, "_parent"):
                    parent = getattr(parent, "_parent")
                else:
                    break
            if len(self._parentlevel) < i + 1:
                raise AttributeError(
                    f"Cannot find parameter '{p}' in parent objects of parameter '{self.name}'"
                )

    def _get(self):
        """Run the function and return the result

        Returns
        -------
        astropy.units.Quantity or None
            If any input value is ``None`` or if the parentage is not yet specified, will return ``None``
            Otherwise will return the result of the function

        """
        if self._parent is None:
            return None
        if self._parentlevel == []:
            self._get_parentage()
        args = []
        for l, p, a in zip(self._parentlevel, self._params, self._attrs):
            args.append(getattr(getattr(l, p), a))
            if args[-1] is None:
                return None
        return self._func(*args)

    @property
    def quantity(self):
        """The result of the function"""
        return self._get()

    @quantity.setter
    def quantity(self, value):
        raise AttributeError("Cannot set funcParameter")

    @property
    def value(self):
        """The result of the function without units."""
        return self._get().value if self._get() is not None else None

    @value.setter
    def value(self, value):
        raise AttributeError("Cannot set funcParameter")

    @property
    def params(self):
        """Return a list of tuples of parameter names and attributes"""
        return list(zip(self._params, self._attrs))

    @params.setter
    def params(self, params):
        self._set_params(params)

    def from_parfile_line(self, line):
        """Ignore reading from par file

        For :class:`~pint.models.parameter.funcParameter` ,
        it is for information only so is ignored on reading
        """
        return True

    def as_parfile_line(self, format="pint"):
        return (
            super().as_parfile_line(format=format)
            if self.inpar
            else f"# {super().as_parfile_line(format=format)}"
        )
