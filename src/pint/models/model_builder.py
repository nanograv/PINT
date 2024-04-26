"""Building a timing model from a par file."""

import copy
import warnings
from io import StringIO
from collections import Counter, defaultdict
from pathlib import Path
from astropy import units as u
from loguru import logger as log
import re

from pint.models.astrometry import Astrometry
from pint.models.parameter import maskParameter
from pint.models.timing_model import (
    DEFAULT_ORDER,
    Component,
    AllComponents,
    TimingModel,
    ignore_prefix,
    AliasConflict,
    UnknownBinaryModel,
    UnknownParameter,
    TimingModelError,
    MissingBinaryError,
    ignore_params,
    ignore_prefix,
)
from pint.toa import get_TOAs
from pint.utils import (
    PrefixError,
    interesting_lines,
    lines_of,
    split_prefixed_name,
    get_unit,
)
from pint.models.tcb_conversion import convert_tcb_tdb
from pint.models.binary_ddk import _convert_kin, _convert_kom

__all__ = ["ModelBuilder", "get_model", "get_model_and_toas"]

default_models = ["StandardTimingModel"]
_binary_model_priority = [
    "Isolated",
    "BT",
    "BT_piecewise",
    "ELL1",
    "ELL1H",
    "ELL1k",
    "DD",
    "DDK",
    "DDGR",
    "DDS",
    "DDH",
]


class ComponentConflict(ValueError):
    """Error for multiple components can be select but no other indications."""


def parse_parfile(parfile):
    """Function for parsing .par file or .par style StringIO.

    Parameters
    ----------
    parfile: str or file-like object
        Input .par file name or string contents.

    Returns
    -------
    dict:
        Parameter and its associated lines. The key is the parameter name and
        the value is a list of the lines associated to the parameter name.
    """
    parfile_dict = defaultdict(list)
    for l in interesting_lines(lines_of(parfile), comments=("#", "C ")):
        k = l.split()
        parfile_dict[k[0].upper()].append(" ".join(k[1:]))
    return parfile_dict


def _replace_fdjump_in_parfile_dict(pardict):
    """Replace parameter names s of the form "FDJUMPp" by "FDpJUMP"
    while reading the par file, where p is the prefix index.

    Ideally, this should have been done using the parameter alias
    mechanism, but there is no easy way to do this currently due to the
    mask and prefix indices being treated in an identical manner.

    See :class:`~pint.models.fdjump.FDJump` for more details."""
    fdjumpn_regex = re.compile("^FDJUMP(\\d+)")
    pardict_new = {}
    for key, value in pardict.items():
        if m := fdjumpn_regex.match(key):
            j = int(m.groups()[0])
            new_key = f"FD{j}JUMP"
            pardict_new[new_key] = value
        else:
            pardict_new[key] = value

    return pardict_new


class ModelBuilder:
    """Class for building a `TimingModel` object from a parameter file.

    The ModelBuilder class helps building a TimingModel from a parameter file
    (i.e., pulsar ephemeris or '.par' file).
    It first maps the provided parameter names to the PINT defined parameter
    names, if they are in the PINT parameter aliases list. Then, the
    ModelBuilder selects model components based on the following rules:

        * The components in the :py:attr:`~default_components` list will be selected.
        * When a component get mapped uniquely by the given parameters.
        * The pulsar binary component will be selected by the 'BINARY' parameter.
    """

    def __init__(self):
        # Validate the components
        self.all_components = AllComponents()
        self._validate_components()
        self.default_components = []

    def __call__(
        self,
        parfile,
        allow_name_mixing=False,
        allow_tcb=False,
        allow_T2=False,
        force_binary_model=None,
        toas_for_tzr=None,
        **kwargs,
    ):
        """Callable object for making a timing model from .par file.

        Parameters
        ----------
        parfile: str or file-like object
            Input .par file name or string contents

        allow_name_mixing : bool, optional
            Flag for allowing the input to have mixing aliases names for the
            same parameter. For example, if this flag is true, one can have
            T2EFAC and EFAC, both of them maps to PINT parameter EFAC, present
            in the parfile at the same time.

        allow_tcb : True, False, or "raw", optional
            Whether to read TCB par files. Default is False, and will throw an
            error upon encountering TCB par files. If True, the par file will be
            converted to TDB upon read. If "raw", an unconverted malformed TCB
            TimingModel object will be returned.

        allow_T2 : bool, optional
            Whether to convert a T2 binary model to an appropriate underlying
            binary model. Default is False, and will throw an error upon
            encountering the T2 binary model. If True, the binary model will be
            converted to the most appropriate PINT-compatible binary model.

        force_binary_model : str, optional
            When set to some binary model, like force_binary_model="DD", this
            will override the binary model set in the parfile. Defaults to None

        toas_for_tzr : TOAs or None, optional
            If this is not None, a TZR TOA (AbsPhase) will be created using the
            given TOAs object.

        kwargs : dict
            Any additional parameter/value pairs that will add to or override those in the parfile.

        Returns
        -------
        pint.models.timing_model.TimingModel
            The result timing model based on the input .parfile or file object.
        """

        assert allow_tcb in [True, False, "raw"]
        convert_tcb = allow_tcb == True
        allow_tcb_ = allow_tcb in [True, "raw"]

        assert isinstance(allow_T2, bool)

        pint_param_dict, original_name, unknown_param = self._pintify_parfile(
            parfile, allow_name_mixing
        )
        remaining_args = {}
        for k, v in kwargs.items():
            if k not in pint_param_dict:
                if isinstance(v, u.Quantity):
                    pint_param_dict[k] = [
                        str(v.to_value(get_unit(k))),
                    ]
                else:
                    pint_param_dict[k] = [
                        str(v),
                    ]
                original_name[k] = k
            else:
                remaining_args[k] = v
        selected, conflict, param_not_in_pint = self.choose_model(
            pint_param_dict, force_binary_model=force_binary_model, allow_T2=allow_T2
        )
        selected.update(set(self.default_components))

        # Add SolarSystemShapiro only if an Astrometry component is present.
        if any(
            isinstance(self.all_components.components[sc], Astrometry)
            for sc in selected
        ):
            selected.add("SolarSystemShapiro")

        # Report conflict
        if len(conflict) != 0:
            self._report_conflict(conflict)
        # Make timing model
        cps = [self.all_components.components[c] for c in selected]
        tm = TimingModel(components=cps)
        self._setup_model(
            tm,
            pint_param_dict,
            original_name,
            setup=True,
            validate=True,
            allow_tcb=allow_tcb_,
        )
        # Report unknown line
        for k, v in unknown_param.items():
            p_line = " ".join([k] + v)
            warnings.warn(f"Unrecognized parfile line '{p_line}'", UserWarning)
            # log.warning(f"Unrecognized parfile line '{p_line}'")

        if tm.UNITS.value is None or tm.UNITS.value == "":
            log.warning("UNITS is not specified. Assuming TDB...")
            tm.UNITS.value = "TDB"

        if tm.UNITS.value == "TCB" and convert_tcb:
            convert_tcb_tdb(tm)

        for k, v in remaining_args.items():
            if not hasattr(tm, k):
                raise ValueError(f"Model does not have parameter '{k}'")
            log.debug(f"Overriding '{k}' to '{v}'")
            if isinstance(v, u.Quantity):
                getattr(tm, k).quantity = v
            else:
                getattr(tm, k).value = v

        # Explicitly add a TZR TOA from a given TOAs object.
        if "AbsPhase" not in tm.components and toas_for_tzr is not None:
            log.info("Creating a TZR TOA (AbsPhase) using the given TOAs object.")
            tm.add_tzr_toa(toas_for_tzr)

        if not hasattr(tm, "DelayComponent_list"):
            setattr(tm, "DelayComponent_list", [])
        if not hasattr(tm, "NoiseComponent_list"):
            setattr(tm, "NoiseComponent_list", [])

        return tm

    def _validate_components(self):
        """Validate the built-in component.

        This function validates if there is a subset parameter conflict in the
        Components. Normally, one component's parameter should not be a subset
        of another component's parameter list. Otherwise, the model builder does
        not have an unique choice of component. Currently, Pulsar binary does
        not follow the same rule. They are specified by the `BINARY` parameter.
        """
        for k, v in self.all_components.components.items():
            superset = self._is_subset_component(v)
            if superset is not None:
                m = (
                    f"Component {k}'s parameter is a subset of component"
                    f" {superset}. Module builder will have trouble to "
                    f" select the component. If component {k} is a base"
                    f" class, please set register to 'False' in the class"
                    f" of component {k}."
                )
                if v.category == "pulsar_system":
                    # The pulsar system will be selected by parameter BINARY
                    continue
                else:
                    raise ComponentConflict(m)

    def _get_component_param_overlap(self, component):
        """Check the parameter overlapping between two components.

        Check if one component's parameters are overlapped with another
        component.

        Parameters
        ----------
        components: pint.models.timing_model.Component
            The component to be checked.

        Returns
        -------
        overlap : dict
            The component has overlap parameters and the overlapping parameters
            in the format of {overlap component name: (overlap parameter,
            number of non-overlap param in test component,
            number of non-overlap param in overlap component) }
        """
        overlap_entries = {}
        for k, cp in self.all_components.components.items():
            # Check name is a safer way to avoid the component compares to itself
            if component.__class__.__name__ == k:
                continue
            # We assume parameters are unique in one component
            in_param = set(component.aliases_map)
            cpm_param = set(cp.aliases_map)
            # Add aliases compare
            overlap = in_param & cpm_param
            # translate to PINT parameter
            overlap_pint_par = {
                self.all_components.alias_to_pint_param(ovlp)[0] for ovlp in overlap
            }
            # The degree of overlapping for input component and compared component
            overlap_deg_in = len(component.params) - len(overlap_pint_par)
            overlap_deg_cpm = len(cp.params) - len(overlap_pint_par)
            overlap_entries[k] = (overlap_pint_par, overlap_deg_in, overlap_deg_cpm)
        return overlap_entries

    def _is_subset_component(self, component):
        """Is the component's parameters a subset of another component's parameters.

        Parameters
        ----------
        component: pint.models.timing_model.Component
            The component to be checked.
        Returns
        -------
        str
            The superset component name, or None
        """
        overlap = self._get_component_param_overlap(component)
        for k, v in overlap.items():
            if v[1] == 0:
                return k
        return None

    def _pintify_parfile(self, parfile, allow_name_mixing=False):
        """Translate parfile parameter name to PINT style name.

        This function converts the parfile information to PINT understandable
        parameter name. It also returns the PINT unrecognized parameters and
        check if the parfile has illegal repeating lines.

        Parameters
        ----------
        parfile : str, file-like object, or parfile dictionary
            Parfile name, parfile StringIO, or the parfile dictionary returned
            by :func:`parse_parfile`.

        allow_name_mixing : bool, optional
            Flag for allowing the input to have mixing aliases names for the
            same parameter. For example, if this flag is true, one can have
            T2EFAC and EFAC, both of them maps to PINT parameter EFAC, present
            in the parfile at the same time.

        Returns
        -------
            pint_param_dict : dict
                Pintified parameter dictionary with the PINT name as key and list of
                parameter value-uncertainty lines as value. For the
                repeating parameters in the parfile, the value will contain
                multiple lines.

            original_name_map : dict
                PINT name maps to the original .par file input names. PINT name
                is the key and the original name is in the value.

            unknown_param : dict
                The PINT unrecognized parameters in the format of a dictionary.
                The key is the unknown parameter name and the value is the
                parfile value lines.

        Raises
        ------
        TimingModelError
            If the parfile has multiple line with non-repeating parameters.
        """
        pint_param_dict = defaultdict(list)
        original_name_map = defaultdict(list)
        unknown_param = defaultdict(list)
        repeating = Counter()
        if isinstance(parfile, (str, StringIO, Path)):
            parfile_dict = parse_parfile(parfile)
        else:
            parfile_dict = parfile

        # This is a special-case-hack to deal with FDJUMP parameters.
        # @TODO: Implement a general mechanism to deal with cases like this.
        parfile_dict = _replace_fdjump_in_parfile_dict(parfile_dict)

        for k, v in parfile_dict.items():
            try:
                pint_name, init0 = self.all_components.alias_to_pint_param(k)
            except UnknownParameter:
                if k in ignore_params:
                    # Parameter is known but in the ignore list
                    continue
                # Check ignored prefix
                try:
                    pfx, idxs, idx = split_prefixed_name(k)
                    if pfx in ignore_prefix:  # It is an ignored prefix.
                        continue
                    else:
                        unknown_param[k] += v
                except PrefixError:
                    unknown_param[k] += v
                continue
            pint_param_dict[pint_name] += v
            original_name_map[pint_name].append(k)
            repeating[pint_name] += len(v)
            # Check if this parameter is allowed to be repeated by PINT
            if (
                len(pint_param_dict[pint_name]) > 1
                and pint_name not in self.all_components.repeatable_param
            ):
                raise TimingModelError(
                    f"Parameter {pint_name} is not a repeatable parameter. "
                    f"However, multiple line use it."
                )
        # Check if the name is mixed
        for p_n, o_n in original_name_map.items():
            if len(o_n) > 1 and not allow_name_mixing:
                raise TimingModelError(
                    f"Parameter {p_n} have mixed input names/alias "
                    f"{o_n}. If you want to have mixing names, please use"
                    f" 'allow_name_mixing=True', and the output .par file "
                    f"will use '{original_name_map[pint_name][0]}'."
                )
            original_name_map[p_n] = o_n[0]

        return pint_param_dict, original_name_map, unknown_param

    def choose_model(self, param_inpar, force_binary_model=None, allow_T2=False):
        """Choose the model components based on the parfile.

        Parameters
        ----------
        param_inpar: dict
            Dictionary of the unique parameters in .par file with the key is the
            parfile line. :func:`parse_parfile` returns this dictionary.

        allow_T2 : bool, optional
            Whether to convert a T2 binary model to an appropriate underlying
            binary model. Default is False, and will throw an error upon
            encountering the T2 binary model. If True, the binary model will be
            converted to the most appropriate PINT-compatible binary model.

        force_binary_model : str, optional
            When set to some binary model, like force_binary_model="DD", this
            will override the binary model set in the parfile. Defaults to None

        Returns
        -------
        list
            List of selected components.
        dict
            Conflict components dictionary, where the key the component name,
            the value is a list of component names that are conflicted with the
            components in the key.
        list
            A list of parameters that are in the .parfile but not in the PINT
            defined parameters.

        Note
        ----
        The selection algorithm:
            #. Look at the BINARY in the par file and cache the indicated binary model
            #. Translate para file parameters to the pint parameter name
            #. Go over the parameter-component map and pick up the components based
               on the parameters in parfile.

                #. Select the components that have its unique parameters in the parfile.
                   In other words, select the components that have one parameter to
                   on component mapping.
                #. Log the conflict components, one parameter to multiple components mapping.
        """
        selected_components = set()
        param_count = Counter()
        # 1. iteration read parfile with a no component timing_model to get
        # the overall control parameters. This will get us the binary model name
        # build the base fo the timing model
        # pint_param_dict, unknown_param = self._pintify_parfile(param_inpar)
        binary = param_inpar.get("BINARY", None)

        if binary:
            binary = binary[0]
            selected_components.add(
                self.choose_binary_model(param_inpar, force_binary_model, allow_T2)
            )

        # 2. Get the component list from the parameters in the parfile.
        # 2.1 Check the aliases of input parameters.
        # This does not include the repeating parameters, but it should not
        # matter in the component selection.
        param_not_in_pint = []  # For parameters not initialized in PINT yet.
        param_components_inpar = {}
        for pp in param_inpar.keys():
            try:
                p_name, first_init = self.all_components.alias_to_pint_param(pp)
            except UnknownParameter:
                param_not_in_pint.append(pp)
                continue

            param_count[p_name] += 1

            # For the case that the indexed parameter maps to a component, but
            # the parameter with the provided index is not initialized yet.
            if p_name != first_init:
                param_not_in_pint.append(pp)

            if p_cp := self.all_components.param_component_map.get(first_init, None):
                param_components_inpar[p_name] = p_cp
        # Back map the possible_components and the parameters in the parfile
        # This will remove the duplicate components.
        conflict_components = defaultdict(set)  # graph for conflict
        for k, cps in param_components_inpar.items():
            # If `timing_model` in param --> component mapping skip
            # Timing model is the base.
            if "timing_model" in cps:
                continue
            # Check if it is a binary component, if yes, skip. It is controlled
            # by the BINARY tag
            if self.all_components.components[cps[0]].category == "pulsar_system":
                if binary is None:
                    raise MissingBinaryError(
                        f"Pulsar binary/pulsar system model is"
                        f" decided by the parameter 'BINARY'. "
                        f" Please indicate the binary model "
                        f" before using parameter {k}, which is"
                        f" a binary model parameter."
                    )
                else:
                    continue

            if len(cps) == 1:  # No conflict, parameter only shows in one component.
                selected_components.add(cps[0])
                continue
            # Has conflict, same parameter shows in different components
            # Only record the conflict here and do nothing, if there is any
            # component unique parameter show in the parfile, the component will
            # be selected.
            if len(cps) > 1:
                # Add conflict to the conflict graph
                for cp in cps:
                    temp_cf_cp = copy.deepcopy(cps)
                    temp_cf_cp.remove(cp)
                    conflict_components[cp].update(set(temp_cf_cp))
                continue
        # Check if the selected component in the conflict graph. If it is
        # remove the selected components with its conflict components.
        for ps_cp in selected_components:
            cf_cps = conflict_components.get(ps_cp)
            if cf_cps is not None:  # Had conflict, but resolved.
                for cf_cp in cf_cps:
                    del conflict_components[cf_cp]
                del conflict_components[ps_cp]
        # Check if there are components from the same category
        selected_cates = {}
        for cp in selected_components:
            cate = self.all_components.component_category_map[cp]
            if cate in selected_cates:
                exist_cp = selected_cates[cate]
                raise TimingModelError(
                    f"Component '{cp}' and '{exist_cp}' belong to the"
                    f" same category '{cate}'. Only one component from"
                    f" the same category can be used for a timing model."
                    f" Please check your input (e.g., .par file)."
                )

            else:
                selected_cates[cate] = cp
        return selected_components, conflict_components, param_not_in_pint

    def choose_binary_model(self, param_inpar, force_binary_model=None, allow_T2=False):
        """Choose the BINARY model based on the parfile.

        Parameters
        ----------
        param_inpar: dict
            Dictionary of the unique parameters in .par file with the key is the
            parfile line. :func:`parse_parfile` returns this dictionary.

        force_binary_model : str, optional
            When set to some binary model, like force_binary_model="DD", this
            will override the binary model set in the parfile. Defaults to None

        allow_T2 : bool, optional
            Whether to convert a T2 binary model to an appropriate underlying
            binary model. Default is False, and will throw an error upon
            encountering the T2 binary model. If True, the binary model will be
            converted to the most appropriate PINT-compatible binary model.

        Returns
        -------
        str
            Name of the binary component

        Note
        ----
        If the binary model does not have a PINT model (e.g. the T2 model), an
        error is thrown with the suggested model that could replace it. If
        allow_T2 is set to True, the most appropriate binary model is guessed
        and used. If an appropriate model cannot be found, no suggestion is
        given and an error is thrown.
        """
        binary = param_inpar["BINARY"][0]

        # Guess what the binary model should be, regardless of BINARY parameter
        try:
            binary_model_guesses = guess_binary_model(param_inpar)
        except UnknownBinaryModel as e:
            log.error(
                "Unable to find suitable binary model that has all the"
                "parameters in the parfile. Please fix the par file."
            )

        # Allow for T2 model, gracefully
        if force_binary_model is not None and binary != "T2":
            binary = force_binary_model
        elif binary == "T2" and allow_T2:
            binary = binary_model_guesses[0]
            log.warning(
                f"Found T2 binary model. Gracefully converting T2 to: {binary}."
            )

            # Make sure that DDK parameters are properly converted
            convert_binary_params_dict(param_inpar, force_binary_model=binary)

        try:
            binary_cp = self.all_components.search_binary_components(binary)

        except UnknownBinaryModel as e:
            log.error(f"Could not find binary model {binary}")

            log.info(
                f"Compatible models with these parameters: {', '.join(binary_model_guesses)}."
            )

            # Re-raise the error, with an added guess for the binary model if we have one
            if binary_model_guesses:
                raise UnknownBinaryModel(
                    str(e), suggestion=binary_model_guesses[0]
                ) from None
            raise

        return binary_cp.__class__.__name__

    def _setup_model(
        self,
        timing_model,
        pint_param_dict,
        original_name=None,
        setup=True,
        validate=True,
        allow_tcb=False,
    ):
        """Fill up a timing model with parameter values and then setup the model.

        This function fills up the timing model parameter values from the input
        pintified parameter dictionary. If the parameter has not initialized yet,
        it will add the parameter to the timing model. For the repeatable parameters,
        it will search matching key value pair first. If the input parameter line's
        key-value matches the existing parameter, the parameter value and uncertainty
        will copy to the existing parameter. If there is no match, it will find an
        empty existing parameter, whose `key` is `None`, and fill it up. If no empty
        parameter left, it will add a new parameter to it.

        Parameters
        ----------
        timing_model : pint.models.TimingModel
            Timing model to get setup.
        pint_param_dict: dict
            Pintified parfile dictionary which can be acquired by
            :meth:`ModelBuilder._pintify_parfile`
        original_name : dict, optional
            A map from PINT name to the original input name.
        setup : bool, optional
            Whether to run the setup function in the timing model.
        validate : bool, optional
            Whether to run the validate function in the timing model.
        allow_tcb : bool, optional
            Whether to allow reading TCB par files
        """
        use_alias = original_name is not None
        for pp, v in pint_param_dict.items():
            try:
                par = getattr(timing_model, pp)
            except AttributeError:
                # since the input is pintfied, it should be an uninitialized indexed parameter
                # double check if the missing parameter an indexed parameter.
                pint_par, first_init = self.all_components.alias_to_pint_param(pp)
                try:
                    prefix, _, index = split_prefixed_name(pint_par)
                except PrefixError as e:
                    par_hosts = self.all_components.param_component_map[pint_par]
                    current_cp = timing_model.components.keys()
                    raise TimingModelError(
                        f"Parameter {pint_par} is recognized"
                        f" by PINT, but not used in the current"
                        f" timing model. It is used in {par_hosts},"
                        f" but the current timing model uses {current_cp}."
                    ) from e
                # TODO need to create a better API for _locate_param_host
                host_component = timing_model._locate_param_host(first_init)
                timing_model.add_param_from_top(
                    getattr(timing_model, first_init).new_param(index),
                    host_component[0][0],
                )
                par = getattr(timing_model, pint_par)

            # Fill up the values
            param_line = len(v)
            if param_line < 2:
                name = original_name[pp] if use_alias else pp
                par.from_parfile_line(" ".join([name] + v))
            else:  # For the repeatable parameters
                lines = copy.deepcopy(v)  # Line queue.
                # Check how many repeatable parameters in the model.
                example_par = getattr(timing_model, pp)
                prefix, _, index = split_prefixed_name(pp)
                for li in lines:
                    # Create a temp parameter with the idx bigger than all the existing indices
                    repeatable_map = timing_model.get_prefix_mapping(prefix)
                    new_max_idx = max(repeatable_map.keys()) + 1
                    temp_par = example_par.new_param(new_max_idx)
                    temp_par.from_parfile_line(
                        " ".join([prefix + str(new_max_idx), li])
                    )
                    if use_alias:  # Use the input alias as input
                        temp_par.use_alias = original_name[pp]
                    # Check current repeatable's key and value
                    # TODO need to change here when maskParameter name changes to name_key_value
                    empty_repeat_param = []
                    for idx, rp in repeatable_map.items():
                        rp_par = getattr(timing_model, rp)
                        if rp_par.compare_key_value(temp_par):
                            # Key and key value match, copy the new line to it
                            # and exit
                            rp_par.from_parfile_line(" ".join([rp, li]))
                            if use_alias:  # Use the input alias as input
                                rp_par.use_alias = original_name[pp]
                            break

                        if rp_par.key is None:
                            # Empty space for new repeatable parameter
                            empty_repeat_param.append(rp_par)

                    # There is no current repeatable parameter matching the new line
                    # First try to fill up an empty space.
                    if not empty_repeat_param:
                        # No empty space, add a new parameter to the timing model.
                        host_component = timing_model._locate_param_host(pp)
                        timing_model.add_param_from_top(temp_par, host_component[0][0])

                    else:
                        emt_par = empty_repeat_param.pop(0)
                        emt_par.from_parfile_line(" ".join([emt_par.name, li]))
                        if use_alias:  # Use the input alias as input
                            emt_par.use_alias = original_name[pp]
        if setup:
            timing_model.setup()
        if validate:
            timing_model.validate(allow_tcb=allow_tcb)
        return timing_model

    def _report_conflict(self, conflict_graph):
        """Report conflict components"""
        for k, v in conflict_graph.items():
            # Put all the conflict components together from the graph
            cf_cps = list(v)
            cf_cps.append(k)
            raise ComponentConflict(f"Can not decide the one component from: {cf_cps}")


def get_model(
    parfile,
    allow_name_mixing=False,
    allow_tcb=False,
    allow_T2=False,
    force_binary_model=None,
    toas_for_tzr=None,
    **kwargs,
):
    """A one step function to build model from a parfile.

    Parameters
    ----------
    parfile : str
        The parfile name, or a file-like object to read the parfile contents from

    allow_name_mixing : bool, optional
        Flag for allowing the input to have mixing aliases names for the
        same parameter. For example, if this flag is true, one can have
        T2EFAC and EFAC, both of them maps to PINT parameter EFAC, present
        in the parfile at the same time.

    allow_tcb : True, False, or "raw", optional
        Whether to read TCB par files. Default is False, and will throw an
        error upon encountering TCB par files. If True, the par file will be
        converted to TDB upon read. If "raw", an unconverted malformed TCB
        TimingModel object will be returned.

    allow_T2 : bool, optional
        Whether to convert a T2 binary model to an appropriate underlying
        binary model. Default is False, and will throw an error upon
        encountering the T2 binary model. If True, the binary model will be
        converted to the most appropriate PINT-compatible binary model.

    force_binary_model : str, optional
        When set to some binary model, like force_binary_model="DD", this will
        override the binary model set in the parfile. Defaults to None

    toas_for_tzr : TOAs or None, optional
        If this is not None, a TZR TOA (AbsPhase) will be created using the
        given TOAs object.

    kwargs : dict
        Any additional parameter/value pairs that will add to or override those in the parfile.

    Returns
    -------
    Model instance get from parfile.
    """
    model_builder = ModelBuilder()
    try:
        contents = parfile.read()
    except AttributeError:
        contents = None
    if contents is not None:
        return model_builder(
            StringIO(contents),
            allow_name_mixing,
            allow_tcb=allow_tcb,
            allow_T2=allow_T2,
            force_binary_model=force_binary_model,
            toas_for_tzr=toas_for_tzr,
            **kwargs,
        )

    # # parfile is a filename and can be handled by ModelBuilder
    # if _model_builder is None:
    #     _model_builder = ModelBuilder()
    model = model_builder(
        parfile,
        allow_name_mixing,
        allow_tcb=allow_tcb,
        allow_T2=allow_T2,
        force_binary_model=force_binary_model,
        toas_for_tzr=toas_for_tzr,
        **kwargs,
    )
    model.name = parfile

    return model


def get_model_and_toas(
    parfile,
    timfile,
    ephem=None,
    include_bipm=None,
    bipm_version=None,
    include_gps=None,
    planets=None,
    usepickle=False,
    tdb_method="default",
    include_pn=True,
    picklefilename=None,
    allow_name_mixing=False,
    limits="warn",
    allow_tcb=False,
    allow_T2=False,
    force_binary_model=None,
    add_tzr_to_model=True,
    **kwargs,
):
    """Load a timing model and a related TOAs, using model commands as needed

    Parameters
    ----------
    parfile : str
        The parfile name, or a file-like object to read the parfile contents from
    timfile : str
        The timfile name, or a file-like object to read the timfile contents from
    ephem : str, optional
        If not None (default), this ephemeris will be used to create the TOAs object.
        Default is to use the EPHEM parameter from the timing model.
    include_bipm : bool or None
        Whether to apply the BIPM clock correction. Defaults to True.
    bipm_version : string or None
        Which version of the BIPM tables to use for the clock correction.
        The format must be 'BIPMXXXX' where XXXX is a year.
    include_gps : bool or None
        Whether to include the GPS clock correction. Defaults to True.
    planets : bool or None
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.
    usepickle : bool
        Whether to try to use pickle-based caching of loaded clock-corrected TOAs objects.
    tdb_method : str
        Which method to use for the clock correction to TDB. See
        :func:`pint.observatory.Observatory.get_TDBs` for details.
    include_pn : bool, optional
        Whether or not to read in the 'pn' column (``pulse_number``)
    picklefilename : str or None
        Filename to use for caching loaded file. Defaults to adding ``.pickle.gz`` to the
        filename of the timfile, if there is one and only one. If no filename is available,
        or multiple filenames are provided, a specific filename must be provided.
    allow_name_mixing : bool, optional
        Flag for allowing the input to have mixing aliases names for the
        same parameter. For example, if this flag is true, one can have
        T2EFAC and EFAC, both of them maps to PINT parameter EFAC, present
        in the parfile at the same time.
    limits : "warn" or "error"
        What to do when encountering TOAs for which clock corrections are not available.
    allow_tcb : True, False, or "raw", optional
        Whether to read TCB par files. Default is False, and will throw an
        error upon encountering TCB par files. If True, the par file will be
        converted to TDB upon read. If "raw", an unconverted malformed TCB
        TimingModel object will be returned.
    allow_T2 : bool, optional
        Whether to convert a T2 binary model to an appropriate underlying
        binary model. Default is False, and will throw an error upon
        encountering the T2 binary model. If True, the binary model will be
        converted to the most appropriate PINT-compatible binary model.
    force_binary_model : str, optional
        When set to some binary model, like force_binary_model="DD", this
        will override the binary model set in the parfile. Defaults to None
    add_tzr_to_model : bool, optional
        Create a TZR TOA in the timing model using the created TOAs object. Default is
        True.
    kwargs : dict
        Any additional parameter/value pairs that will add to or override those in the parfile.

    Returns
    -------
    A tuple with (model instance, TOAs instance)
    """

    mm = get_model(
        parfile,
        allow_name_mixing,
        allow_tcb=allow_tcb,
        allow_T2=allow_T2,
        force_binary_model=force_binary_model,
        **kwargs,
    )

    tt = get_TOAs(
        timfile,
        include_pn=include_pn,
        model=mm,
        ephem=ephem,
        include_bipm=include_bipm,
        bipm_version=bipm_version,
        include_gps=include_gps,
        planets=planets,
        usepickle=usepickle,
        tdb_method=tdb_method,
        picklefilename=picklefilename,
        limits=limits,
    )

    if "AbsPhase" not in mm.components and add_tzr_to_model:
        log.info("Creating a TZR TOA (AbsPhase) using the given TOAs object.")
        mm.add_tzr_toa(tt)

    return mm, tt


def guess_binary_model(parfile_dict):
    """Based on the PINT parameter dictionary, guess the binary model

    Parameters
    ----------
    parfile_dict
        The parameter dictionary as read-in by parse_parfile

    Returns
    -------
    list:
        A priority-ordered list of possible binary models. The first one is the
        best-guess

    """

    def add_sini(parameters):
        """If 'KIN' is a model parameter, Tempo2 doesn't really use SINI"""
        if "KIN" in parameters:
            return list(parameters) + ["SINI"]
        else:
            return list(parameters)

    all_components = AllComponents()
    binary_models = all_components.category_component_map["pulsar_system"]

    # Find all binary parameters
    binary_parameters_map = {
        all_components.components[binary_model].binary_model_name: add_sini(
            all_components.search_binary_components(binary_model).aliases_map.keys()
        )
        for binary_model in binary_models
    }
    binary_parameters_map.update({"Isolated": []})
    all_binary_parameters = {
        parname for parnames in binary_parameters_map.values() for parname in parnames
    }

    # Find all parfile parameters
    all_parfile_parameters = set(parfile_dict.keys())

    # All binary parameters in the parfile
    parfile_binary_parameters = all_parfile_parameters & all_binary_parameters

    # Find which binary models include those
    allowed_binary_models = {
        binary_model
        for (binary_model, bmc) in binary_parameters_map.items()
        if len(parfile_binary_parameters - set(bmc)) == 0
    }

    # Now select the best-guess binary model
    priority = [bm for bm in _binary_model_priority if bm in allowed_binary_models]
    omitted = allowed_binary_models - set(priority)

    return priority + list(omitted)


def convert_binary_params_dict(
    parfile_dict, convert_komkin=True, drop_ddk_sini=True, force_binary_model=None
):
    """Convert the PINT parameter dictionary to include the best-guess binary

    Parameters
    ----------
    parfile_dict
        The parameter dictionary as read-in by parse_parfile
    convert_komkin
        Whether or not to convert the KOM and KIN parameters
    drop_ddk_sini
        Whether to drop SINI when converting to the DDK model

    force_binary_model : str, optional
        When set to some binary model, like force_binary_model="DD", this will
        override the binary model set in the parfile. Defaults to None

    Returns
    -------
    A new parfile dictionary with the binary model replaced with the best-guess
    model. For a conversion to DDK, this function also converts the KOM/KIN
    parameters if they exist.
    """
    binary = parfile_dict.get("BINARY", None)
    binary = binary if not binary else binary[0]
    log.debug(f"Requested to convert binary model for BINARY model: {binary}")

    if binary:
        if not force_binary_model:
            binary_model_guesses = guess_binary_model(parfile_dict)
            log.info(
                f"Compatible models with these parameters: {', '.join(binary_model_guesses)}. Using {binary_model_guesses[0]}"
            )

            if not binary_model_guesses:
                error_message = f"Unable to determine binary model for this par file"
                log_message = (
                    f"Unable to determine the binary model based"
                    f"on the model parameters in the par file."
                )

                log.error(log_message)
                raise UnknownBinaryModel(error_message)

        else:
            binary_model_guesses = [force_binary_model]

        # Select the best-guess binary model
        parfile_dict["BINARY"] = [binary_model_guesses[0]]

        # Convert KIN if requested
        if convert_komkin and "KIN" in parfile_dict:
            log.info(f"Converting KOM to/from IAU <--> DT96: {parfile_dict['KIN']}")
            log.debug(f"Converting KIN to/from IAU <--> DT96")
            entries = parfile_dict["KIN"][0].split()
            new_value = _convert_kin(float(entries[0]) * u.deg).value
            parfile_dict["KIN"] = [" ".join([repr(new_value)] + entries[1:])]

        # Convert KOM if requested
        if convert_komkin and "KOM" in parfile_dict:
            log.debug(f"Converting KOM to/from IAU <--> DT96")
            entries = parfile_dict["KOM"][0].split()
            new_value = _convert_kom(float(entries[0]) * u.deg).value
            parfile_dict["KOM"] = [" ".join([repr(new_value)] + entries[1:])]

        # Drop SINI if requested
        if drop_ddk_sini and binary_model_guesses[0] == "DDK":
            log.debug(f"Dropping SINI from DDK model")
            parfile_dict.pop("SINI", None)

    return parfile_dict
