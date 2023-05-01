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
from loguru import logger as log
from uncertainties import ufloat

from pint.models.parameter.bool_parameter import boolParameter
from pint.models.parameter.float_parameter import floatParameter
from pint.models.parameter.int_parameter import intParameter
from pint.models.parameter.mjd_parameter import MJDParameter
from pint.models.parameter.param_base import (
    Parameter,
    get_observatory_name as _get_observatory_name,
    parfile_formats as _parfile_formats,
    return_frequency_asquantity as _return_frequency_asquantity,
)
from pint.models.parameter.str_parameter import strParameter
from pint.models.parameter.angle_parameter import AngleParameter
from pint.models.parameter.pair_parameter import pairParameter
from pint.models.parameter.prefix_parameter import prefixParameter
from pint.pulsar_mjd import str2longdouble, time_to_mjd_string
from pint.toa_select import TOASelect


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
