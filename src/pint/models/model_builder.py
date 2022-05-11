import os
import copy
from io import StringIO
from collections import Counter, defaultdict
import warnings
from pint.models.parameter import maskParameter
from astropy import log
from astropy.utils.decorators import lazyproperty
from loguru import logger as log
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
from pint.utils import PrefixError, interesting_lines, lines_of, split_prefixed_name


__all__ = ["ModelBuilder", "get_model", "get_model_and_toas"]

default_models = ["StandardTimingModel"]


class ComponentConflict(ValueError):
    """Error for mulitple components can be select but no other indications."""


def parse_parfile(parfile):
    """Function for parsing .par file or .par style StringIO.

    Parameter
    ---------
    parfile: str or file-like object
        Input .par file name or string contents.

    Return
    ------
    dict:
        Parameter and its associated lines. The key is the parameter name and
        the value is a list of the lines associated to the parameter name.
    """
    parfile_dict = defaultdict(list)
    for l in interesting_lines(lines_of(parfile), comments=("#", "C ")):
        k = l.split()
        parfile_dict[k[0].upper()].append(" ".join(k[1:]))
    return parfile_dict


class ModelBuilder:
    """Class for building a `TimingModel` object from a parameter file.

    The ModelBuilder class helps building a TimingModel from a parameter file
    (i.e., pulsar ephemerise or '.par' file).
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
        self.default_components = ["SolarSystemShapiro"]

    def __call__(self, parfile, allow_name_mixing=False):
        """Callable object for making a timing model from .par file.

        Parameter
        ---------
        parfile: str or file-like object
            Input .par file name or string contents

        allow_name_mixing : bool, optional
            Flag for allowing the input to have mixing aliases names for the
            same parameter. For example, if this flag is true, one can have
            T2EFAC and EFAC, both of them maps to PINT parameter EFAC, present
            in the parfile at the same time.

        Return
        ------
        pint.models.timing_model.TimingModel
            The result timing model based on the input .parfile or file object.
        """
        pint_param_dict, original_name, unknown_param = self._pintify_parfile(
            parfile, allow_name_mixing
        )
        selected, conflict, param_not_in_pint = self.choose_model(pint_param_dict)
        selected.update(set(self.default_components))
        # Report conflict
        if len(conflict) != 0:
            self._report_conflict(conflict)
        # Make timing model
        cps = [self.all_components.components[c] for c in selected]
        tm = TimingModel(components=cps)
        self._setup_model(tm, pint_param_dict, original_name, setup=True, validate=True)
        # Report unknown line
        for k, v in unknown_param.items():
            p_line = " ".join([k] + v)
            warnings.warn(f"Unrecognized parfile line '{p_line}'", UserWarning)
            # log.warning(f"Unrecognized parfile line '{p_line}'")
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
        """Check the parameter overlaping between two components.

        Check if one component's parameters are overlaped with another
        component.

        Parameters
        ----------
        components: pint.models.timing_model.Component
            The component to be checked.
        Returns
        -------
        overlap : dict
            The component has overlap parameters and the overlaping parameters
            in the format of {overlap compnent name: (overlap parameter,
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
            overlap_pint_par = set(
                [self.all_components.alias_to_pint_param(ovlp)[0] for ovlp in overlap]
            )
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
                mulitple lines.

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
            If the parfile has mulitple line with non-repeating parameters.
        """
        pint_param_dict = defaultdict(list)
        original_name_map = defaultdict(list)
        unknown_param = defaultdict(list)
        repeating = Counter()
        if isinstance(parfile, (str, StringIO)):
            parfile_dict = parse_parfile(parfile)
        else:
            parfile_dict = parfile
        for k, v in parfile_dict.items():
            try:
                pint_name, init0 = self.all_components.alias_to_pint_param(k)
            except UnknownParameter:
                if k in ignore_params:  # Parameter is known but in the ingore list
                    continue
                else:  # Check ignored prefix
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
            if len(pint_param_dict[pint_name]) > 1:
                if pint_name not in self.all_components.repeatable_param:
                    raise TimingModelError(
                        f"Parameter {pint_name} is not a repeatable parameter. "
                        f"However, mulitple line use it."
                    )
        # Check if the name is mixed
        for p_n, o_n in original_name_map.items():
            if len(o_n) > 1:
                if not allow_name_mixing:
                    raise TimingModelError(
                        f"Parameter {p_n} have mixed input names/alias "
                        f"{o_n}. If you want to have mixing names, please use"
                        f" 'allow_name_mixing=True', and the output .par file "
                        f"will use '{original_name_map[pint_name][0]}'."
                    )
            original_name_map[p_n] = o_n[0]

        return pint_param_dict, original_name_map, unknown_param

    def choose_model(self, param_inpar):
        """Choose the model components based on the parfile.

        Parameter
        ---------
        param_inpar: dict
            Dictionary of the unique parameters in .par file with the key is the
        parfile line. :func:`parse_parfile` returns this dictionary.

        Return
        ------
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
            #. Look at the BINARY in the par file and catche the indicated binary model
            #. Translate para file parameters to the pint parameter name
            #. Go over the parameter-component map and pick up the components based
               on the parameters in parfile.
                #. Select the components that have its unique parameters in the parfile.
                   In other words, select the components that have one parameter to
                   on component mapping.
                #. Log the conflict components, one parameter to mulitple components mapping.
        """
        selected_components = set()
        param_count = Counter()
        # 1. iteration read parfile with a no component timing_model to get
        # the overall control parameters. This will get us the binary model name
        # build the base fo the timing model
        # pint_param_dict, unknown_param = self._pintify_parfile(param_inpar)
        binary = param_inpar.get("BINARY", None)
        if binary is not None:
            binary = binary[0]
            binary_cp = self.all_components.search_binary_components(binary)
            selected_components.add(binary_cp.__class__.__name__)
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

            p_cp = self.all_components.param_component_map.get(first_init, None)
            if p_cp:
                param_components_inpar[p_name] = p_cp
        # Back map the possible_components and the parameters in the parfile
        # This will remove the duplicate components.
        conflict_components = defaultdict(set)  # graph for confilict
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
        # Check if the selected component in the confilict graph. If it is
        # remove the selected componens with its conflict components.
        for ps_cp in selected_components:
            cf_cps = conflict_components.get(ps_cp, None)
            if cf_cps is not None:  # Had conflict, but resolved.
                for cf_cp in cf_cps:
                    del conflict_components[cf_cp]
                del conflict_components[ps_cp]
        # Check if there are components from the same category
        selected_cates = {}
        for cp in selected_components:
            cate = self.all_components.component_category_map[cp]
            if cate not in selected_cates.keys():
                selected_cates[cate] = cp
            else:
                exisit_cp = selected_cates[cate]
                raise TimingModelError(
                    f"Component '{cp}' and '{exisit_cp}' belong to the"
                    f" same category '{cate}'. Only one component from"
                    f" the same category can be used for a timing model."
                    f" Please check your input (e.g., .par file)."
                )

        return selected_components, conflict_components, param_not_in_pint

    def _setup_model(
        self,
        timing_model,
        pint_param_dict,
        original_name=None,
        setup=True,
        validate=True,
    ):
        """Fill up a timing model with parameter values and then setup the model.

        This function fills up the timing model parameter values from the input
        pintified parameter dictionary. If the parameter has not initialized yet,
        it will add the parameter to the timing model. For the repeatable parameters,
        it will search matching key value pair first. If the input parameter line's
        key-value matches the existing parameter, the parameter value and uncertainty
        will copy to the existing parameter. If there is no match, it will find an
        empty existing parameter, whose `key` is `None`, and fill it up. If no empyt
        parameter left, it will add a new parameter to it.

        Parameters
        ----------
        timing_model : pint.models.TimingModel
            Timing model to get setup.
        pint_param_dict: dict
            Pintified parfile dictionary which can be aquired by
            :meth:`ModelBuilder._pintify_parfile`
        origin_name : dict, optional
            A map from PINT name to the original input name.
        setup : bool, optional
            Whether to run the setup function in the timing model.
        validate : bool, optional
            Whether to run the validate funciotn in the timing model.
        """
        if original_name is not None:
            use_alias = True
        else:
            use_alias = False

        for pp, v in pint_param_dict.items():
            try:
                par = getattr(timing_model, pp)
            except AttributeError:
                # since the input is pintfied, it should be an uninitized indexed parameter
                # double check if the missing parameter an indexed parameter.
                pint_par, first_init = self.all_components.alias_to_pint_param(pp)
                try:
                    prefix, _, index = split_prefixed_name(pint_par)
                except PrefixError:
                    par_hosts = self.all_components.param_component_map[pint_par]
                    currnt_cp = timing_model.components.keys()
                    raise TimingModelError(
                        f"Parameter {pint_par} is recognized"
                        f" by PINT, but not used in the current"
                        f" timing model. It is used in {par_hosts},"
                        f" but the current timing model uses {currnt_cp}."
                    )
                # TODO need to create a beeter API for _loacte_param_host
                host_component = timing_model._locate_param_host(first_init)
                timing_model.add_param_from_top(
                    getattr(timing_model, first_init).new_param(index),
                    host_component[0][0],
                )
                par = getattr(timing_model, pint_par)

            # Fill up the values
            param_line = len(v)
            if param_line < 2:
                if use_alias:  # Use the input alias as input
                    name = original_name[pp]
                else:
                    name = pp
                par.from_parfile_line(" ".join([name] + v))
            else:  # For the repeatable parameters
                lines = copy.deepcopy(v)  # Line queue.
                # Check how many repeatable parameters in the model.
                example_par = getattr(timing_model, pp)
                prefix, _, index = split_prefixed_name(pp)
                for li in lines:
                    # Creat a temp parameter with the idx bigger than all the existing indices
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
                    if empty_repeat_param != []:
                        emt_par = empty_repeat_param.pop(0)
                        emt_par.from_parfile_line(" ".join([emt_par.name, li]))
                        if use_alias:  # Use the input alias as input
                            emt_par.use_alias = original_name[pp]
                    else:
                        # No empty space, add a new parameter to the timing model.
                        host_component = timing_model._locate_param_host(pp)
                        timing_model.add_param_from_top(temp_par, host_component[0][0])

        if setup:
            timing_model.setup()
        if validate:
            timing_model.validate()
        return timing_model

    def _report_conflict(self, conflict_graph):
        """Report conflict components"""
        for k, v in conflict_graph.items():
            # Put all the conflict components together from the graph
            cf_cps = list(v)
            cf_cps.append(k)
            raise ComponentConflict(
                "Can not decide the one component from:" " {}".format(cf_cps)
            )


def get_model(parfile, allow_name_mixing=False):
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

    Returns
    -------
    Model instance get from parfile.
    """
    model_builder = ModelBuilder()
    try:
        contents = parfile.read()
    except AttributeError:
        contents = None
    if contents is None:
        # # parfile is a filename and can be handled by ModelBuilder
        # if _model_builder is None:
        #     _model_builder = ModelBuilder()
        model = model_builder(parfile, allow_name_mixing)
        model.name = parfile
        return model
    else:
        tm = model_builder(StringIO(contents), allow_name_mixing)
        return tm


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
    picklefilename=None,
    allow_name_mixing=False,
):
    """Load a timing model and a related TOAs, using model commands as needed

    Parameters
    ----------
    parfile : str
        The parfile name, or a file-like object to read the parfile contents from
    timfile : str
        The timfile name, or a file-like object to read the timfile contents from
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

    Returns
    -------
    A tuple with (model instance, TOAs instance)
    """
    mm = get_model(parfile, allow_name_mixing)
    tt = get_TOAs(
        timfile,
        model=mm,
        ephem=ephem,
        include_bipm=include_bipm,
        bipm_version=bipm_version,
        include_gps=include_gps,
        planets=planets,
        usepickle=usepickle,
        tdb_method=tdb_method,
        picklefilename=picklefilename,
        limits="warn",
    )
    return mm, tt
