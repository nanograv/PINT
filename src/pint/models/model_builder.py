import logging
import os
import tempfile
import copy
from collections import Counter, defaultdict
from pint.models.parameter import maskParameter
from astropy import log
from astropy.utils.decorators import lazyproperty
from pint.models.timing_model import (
    DEFAULT_ORDER,
    Component,
    TimingModel,
    ignore_prefix,
)
from pint.toa import get_TOAs
from pint.utils import PrefixError, interesting_lines, lines_of, split_prefixed_name

log = logging.getLogger(__name__)


__all__ = ["ModelBuilder", "get_model", "get_model_and_toas"]

default_models = ["StandardTimingModel"]


class UnknownBinaryModel(ValueError):
    """Signal that the par file requested a binary model no in PINT."""


class ComponentConflict(ValueError):
    """Error for mulitple components can be select but no other indications."""


class ConflictAliasError(ValueError):
    """If the same alias is used for different parameters."""


class ModelBuilder:
    def __init__(self):
        self.components = {}
        for k, v in Component.component_types.items():
            self.components[k] = v()
        # The components that always get added.
        self._validate_components()
        self.default_components = ["SolarSystemShapiro"]

    def __call__(self, parfile):
        param_inpar, repeat_par = self.parse_parfile(parfile)
        selected, conflict, unrec_param = self.choose_model(param_inpar)
        selected.update(set(self.default_components))
        # Report conflict
        if len(conflict) != 0:
            self._report_conflict(conflict)
        # Make timing model
        cps = [self.components[c] for c in selected]
        tm = TimingModel(components=cps)
        # Add indexed parameters
        leftover_params = set(unrec_param)
        # Give repeatable parameters an index.
        for k, v in repeat_par.items():
            for ii in range(len(v)):
                leftover_params.add(k + str(ii + 1))
        tm, unknown_param = self._add_indexed_params(tm, leftover_params)
        return tm

    def _validate_components(self):
        """Validate the built-in component.
        """
        for k, v in self.components.items():
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

    @lazyproperty
    def param_component_map(self):
        """Return the parameter to component map.
        """
        p2c_map = defaultdict(list)
        for k, cp in self.components.items():
            for p in cp.params:
                p2c_map[p].append(k)
                # Add alias
                par = getattr(cp, p)
                for ap in par.aliases:
                    p2c_map[ap].append(k)
        tm = TimingModel()
        for tp in tm.params:
            p2c_map[tp].append("timing_model")
            par = getattr(tm, tp)
            for ap in par.aliases:
                p2c_map[ap].append("timing_model")
        return p2c_map

    @lazyproperty
    def param_alias_map(self):
        """Return the aliases map of all parameters
        """
        alias = {}
        for k, cp in self.components.items():
            for p in cp.params:
                par = getattr(cp, p)
                # Check if an existing record
                alias = self._add_alias_to_map(p, p, alias)
                for als in par.aliases:
                    alias = self._add_alias_to_map(als, p, alias)
        tm = TimingModel()
        for tp in tm.params:
            par = getattr(tm, tp)
            alias = self._add_alias_to_map(tp, tp, alias)
            alias[tp] = tp
            for als in par.aliases:
                alias = self._add_alias_to_map(als, tp, alias)
        return alias

    @lazyproperty
    def repeatable_param(self):
        """Return the repeatable parameter map.
        """
        repeatable = []
        for k, cp in self.components.items():
            for p in cp.params:
                par = getattr(cp, p)
                if par.repeatable:
                    repeatable.append(p)
                    repeatable.append(par._parfile_name)
                    # also add the aliases to the repeatable param
                    for als in par.aliases:
                        repeatable.append(als)
        return list(set(repeatable))

    @lazyproperty
    def category_component_map(self):
        """Return the mapping from category to component.
        """
        category = defaultdict(list)
        for k, cp in self.components.items():
            cat = cp.category
            category[cat].append(k)
        return category

    @lazyproperty
    def component_category_map(self):
        """Return the mapping from component to category.
        """
        cp_ca = {}
        for k, cp in self.components.items():
            cp_ca[k] = cp.category
        return cp_ca

    @lazyproperty
    def component_unique_params(self):
        """Return the unique parameter names in each componens.

        Note
        ----
        This function only returns the pint defined parameter name, not
        including the aliases.
        """
        component_special_params = defaultdict(list)
        for param, cps in self.param_component_map.items():
            if len(cps) == 1:
                component_special_params[cps[0]].append(param)
        return component_special_params

    def _get_component_param_overlap(self, component):
        """Check if one component's parameters are overlaped with another
        component.

        Parameters
        ----------
        components: component object
            The component to be checked.
        Returns
        -------
        overlap: dict
            The component has overlap parameters and the overlaping parameters
            in the format of {overlap compnent name: (overlap parameter,
            number of non-overlap param in test component,
            number of non-overlap param in overlap component) }
        """
        overlap_entries = {}
        for k, cp in self.components.items():
            # Check name is a safer way to avoid the component compares to itself
            if component.__class__.__name__ == k:
                continue
            # We assume parameters are unique in one component
            in_param = set(component.aliases_map)
            cpm_param = set(cp.aliases_map)
            # Add aliases compare
            overlap = in_param & cpm_param
            # translate to PINT parameter
            overlap_pint_par = set([self.alias_to_pint_param(ovlp) for ovlp in overlap])
            # The degree of overlapping for input component and compared component
            overlap_deg_in = len(component.params) - len(overlap_pint_par)
            overlap_deg_cpm = len(cp.params) - len(overlap_pint_par)
            overlap_entries[k] = (overlap_pint_par, overlap_deg_in, overlap_deg_cpm)
        return overlap_entries

    def _is_subset_component(self, component):
        """Is the component's parameters a subset of another component's parameters.

        Parameters
        ----------
        component: component object
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

    def _add_alias_to_map(self, alias, param_name, alias_map):
        """Add one alias to the alias-parameter map.
        """
        if alias in alias_map.keys():
            if param_name == alias_map[alias]:
                return alias_map
            else:
                raise ConflictAliasError(
                    f"Alias {alias} has been used by" f" parameter {param_name}."
                )
        else:
            alias_map[alias] = param_name
        return alias_map

    def parse_parfile(self, parfile):
        """Preprocess the par file.
        Parameter
        ---------
        parfile: str or file-like object
            Input .par file name or string contents
        Return
        ------
        dict
            The unique parameters in .par file with the key is the parfile line.
        dict
            The repeating parameters.

        """
        repeat_par = defaultdict(list)
        param_inpar = {}
        # Parse all the useful lines
        multi_line = Counter()
        for l in interesting_lines(lines_of(parfile), comments=("#", "C ")):
            k = l.split()
            param_inpar[k[0]] = k[1:]
            # Handle the Mulit-tag lines
            multi_line[k[0]] += 1
            if k[0] in self.repeatable_param:
                repeat_par[k[0]].append(k[1:])
            else:
                if multi_line[k[0]] > 1:
                    log.info(
                        "Lines with duplicate keys in par file:"
                        " {} and {}".format(k[0], k[1:])
                    )
        return param_inpar, repeat_par

    def search_pulsar_system_components(self, system_name):
        """Search the component for the pulsar binary.
        """
        all_systems = self.category_component_map["pulsar_system"]
        # Search the system name first
        if system_name in all_systems:
            return self.components[system_name]
        else:  # search for the pulsar system aliases
            for cp_name in all_systems:
                if system_name == self.components[cp_name].binary_model_name:
                    return self.components[cp_name]
            raise UnknownBinaryModel(
                f"Pulsar system/Binary model component"
                f" {system_name} is not provided."
            )

    def alias_to_pint_param(self, alias):
        """Translate the alias to a PINT parameter name.
        """
        pint_par = self.param_alias_map.get(alias, None)
        # If it is not in the map, double check if it is a repeatable par.
        if pint_par is None:
            try:
                prefix, idx_str, idx = split_prefixed_name(alias)
                # assume the index 1 parameter is in the alias map
                # count length of idx_str and dectect leading zeros
                num_lzero = len(idx_str) - len(str(idx))
                if num_lzero > 0:  # Has leading zero
                    fmt = len(idx_str)
                else:
                    fmt = 0
                # Handle the case of start index from 0 and 1
                for start_idx in [0, 1]:
                    example_name = prefix + "{1:0{0}}".format(fmt, start_idx)
                    pint_par = self.param_alias_map.get(example_name, None)
                    if pint_par:
                        break
                if pint_par:  # Find the start parameter index
                    pint_par = split_prefixed_name(pint_par)[0] + idx_str
            except PrefixError:
                pint_par = None
        return pint_par

    def choose_model(self, param_inpar):
        """Choose the model components based on the parfile.

        Parameter
        ---------
        param_inpar: str
            Dictionary of the unique parameters in .par file with the key is the
        parfile line. Function `.parse_parfile` returns this dictionary.

        Return
        ------
        list
            List of selected components and a dictionary of conflict components.

        Note
        ----
        The selection algorithm:
        1. Look at the BINARY in the par file and catche the indicated binary model
        2. Translate para file parameters to the pint parameter name
        3. Go over the parameter-component map and pick up the components based
           on the parameters in parfile.
           3.1 Select the components that have its unique parameters in the parfile.
               In other words, select the components that have one parameter to
               on component mapping, not one parameter to mulitple components.
           3.2 Log the conflict components, one parameter to mulitple components mapping.
        4. Double check the conflict componens log remove the conflict entries if
           the one component is in the selected list.
        """
        selected_components = set()
        # 1. iteration read parfile with a no component timing_model to get
        # the overall control parameters. This will get us the binary model name
        # build the base fo the timing model
        binary = param_inpar.get("BINARY", None)
        if binary is not None:
            binary = binary[0]
            binary_cp = self.search_pulsar_system_components(binary)
            selected_components.add(binary_cp.__class__.__name__)
        # 2. Get the component list from the parameters in the parfile.
        # 2.1 Check the aliases of input parameters.
        # This does not include the repeating parameters, but it should not
        # matter in the component selection.
        unrec_param = []  # For Unrecognized parameters.
        param_components_inpar = {}
        for pp in param_inpar.keys():
            p_name = self.alias_to_pint_param(pp)
            if p_name is not None:
                p_cp = self.param_component_map.get(p_name, None)
                if p_cp:
                    param_components_inpar[p_name] = p_cp
                else:
                    unrec_param.append(pp)
            else:
                unrec_param.append(pp)
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
            if len(cps) == 1:  # No conflict, parameter only shows in one component.
                # Check if it is a binary component, if yes, skip. It is
                # controlled by the BINARY tag
                if self.components[cps[0]].category == "pulsar_system":
                    continue
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
        return selected_components, conflict_components, unrec_param

    def _add_indexed_params(self, timing_model, indexed_params):
        """Add the parameters with unknown number/indexed in parfile (maskParameter/
        prefixParameter) to timing model.

        Parameter
        ---------
        timing_model: `pint.models.TimeModel` object
            Timing model to add the parameters to.
        params: list
            A list of number unknown parameter names.

        Note
        ----
        This function do not fill the parameter values. Only adds the bare
        parameter to the timing model.
        """
        # go over the input parameter list
        unknown_param = []
        for pp in indexed_params:
            pint_p = self.alias_to_pint_param(pp)
            # A true unrecognized name
            if pint_p is None:
                unknown_param.append(pp)
                continue
            # Check if parameter in the timing model already
            if pint_p in timing_model.params:
                continue
            # Check if parameter name has and index and prefix
            try:
                prefix, idx_str, idx = split_prefixed_name(pint_p)
            except PrefixError:
                prefix = None
                idx = None
            if prefix:  #
                search_name = prefix
            else:
                search_name = pint_p
            # TODO, when the prefix parameter structure changed, this will have
            # to change.
            prefix_map = timing_model.get_prefix_mapping(search_name)
            if prefix_map == {}:  # Can not find any prefix mapping
                unknown_param.append(pp)
                continue
            # Get the parameter in the prefix map.
            prefix_param0 = list(prefix_map.items())[0]
            example = getattr(timing_model, prefix_param0[1])
            if (
                not idx
            ):  # Input name has index, init from an example param and add it to timing model.
                idx = max(list(prefix_map.keys())) + 1
            host_component = timing_model._locate_param_host(prefix_param0[1])[0][0]
            timing_model.add_param_from_top(example.new_param(idx), host_component)
        return timing_model, unknown_param

    def _report_conflict(self, conflict_graph):
        """Report conflict components
        """
        for k, v in conflict_graph.items():
            # Put all the conflict components together from the graph
            cf_cps = v.append(k)
            raise ComponentConflict(
                "Can not decide the one component from:" " {}".format(cf_cps)
            )


def get_model(parfile):
    """A one step function to build model from a parfile
    Parameters
    ----------
    parfile : str
        The parfile name, or a file-like object to read the parfile contents from
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
        model = model_builder(parfile)
        model.name = parfile
        model.read_parfile(parfile)
        return model
    else:
        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "temp.par")
            with open(fn, "wt") as f:
                f.write(contents)
            tm = model_builder(fn)
            tm.read_parfile(fn)
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
    Returns
    -------
    A tuple with (model instance, TOAs instance)
    """
    pass
    mm = get_model(parfile)
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
    )
    return mm, tt


def choose_model(
    parfile, category_order=None, name=None, check_for_missing_parameters=False
):
    """Determine which model components are appropriate for parfile."""
    if name is None:
        if isinstance(parfile, str):
            name = os.path.basename(parfile)
        else:
            name = ""
    if category_order is None:
        category_order = DEFAULT_ORDER

    models_by_category = defaultdict(list)
    for k, c_type in Component.component_types.items():
        models_by_category[c_type.category].append(c_type)

    par_dict = {}
    par_lines = []
    multi_tags = set(
        [
            "JUMP",
            "ECORR",
            "T2EFAC",
            "T2EQUAD",
            "EQUAD",
            "EFAC",
            "DMJUMP",
            "DMEFAC",
            "DMEQUAD",
        ]
    )
    multi_line = Counter()
    for l in interesting_lines(lines_of(parfile), comments=("#", "C ")):
        ll = l.split()
        k = ll[0]
        if k in multi_tags:
            multi_line[k] += 1
            k = k + str(multi_line[k])
        if k in par_dict:
            # FIXME: what happens with JUMPs?
            log.info(
                "Lines with duplicate keys in par file: {} and {}".format(
                    [k] + par_dict[k], ll
                )
            )
        par_dict[k] = ll[1:]
        par_lines.append(l)

    models_to_use = {}
    for category, models in models_by_category.items():
        acceptable = []
        for m_type in models:
            m = m_type()
            if m.is_in_parfile(par_dict):
                acceptable.append(m)
        if len(acceptable) > 1:
            raise ValueError(
                "Multiple models are compatible with this par file: {}".format(
                    acceptable
                )
            )
        if acceptable:
            models_to_use[category] = acceptable[0]

    if "BINARY" in par_dict:
        vals = par_dict["BINARY"]
        if len(vals) != 1:
            raise ValueError(
                "Mal-formed binary model selection: {}".format(
                    repr(" ".join(["BINARY"] + vals))
                )
            )
        (bm,) = vals
        if "pulsar_system" not in models_to_use:
            # Either we're missing parameters or the model is bogus
            # FIXME: distinguish
            raise UnknownBinaryModel(
                "Unknown binary model requested in par file: {}".format(bm)
            )
        # FIXME: consistency check - the componens actually chosen should know the name bm

    models_in_order = []
    for category in category_order:
        try:
            models_in_order.append(models_to_use.pop(category))
        except KeyError:
            pass
    models_in_order.extend(v for k, v in sorted(models_to_use.items()))
    tm = TimingModel(name, models_in_order)

    # FIXME: this should go in TimingModel for when you try to
    # add conflicting components
    alias_map = {}
    for prefix_type in ["prefixParameter", "maskParameter"]:
        for pn in tm.get_params_of_type_top(prefix_type):
            par = getattr(tm, pn)
            for a in [par.prefix] + par.prefix_aliases:
                if a in alias_map:
                    raise ValueError(
                        "Two prefix/mask parameters have the same "
                        "alias {}: {} and {}".format(a, alias_map[a], par)
                    )
                alias_map[a] = par

    leftover_params = par_dict.copy()
    for k in tm.get_params_mapping():
        leftover_params.pop(k, None)
        for a in getattr(tm, k).aliases:
            leftover_params.pop(a, None)

    for p in leftover_params:
        try:
            pre, idxstr, idxV = split_prefixed_name(p)
            try:
                par = alias_map[pre]
            except KeyError:
                if pre in ignore_prefix:
                    # log.warning("Ignoring unhandled prefix {}".format(pre))
                    continue
                else:
                    raise ValueError(
                        "Mystery parameter {}, prefix {} with number {}".format(
                            p, pre, idxV
                        )
                    )
            component = tm.get_params_mapping()[par.name]
            new_parameter = par.new_param(idxV)
            if hasattr(tm, new_parameter.name):
                raise ValueError(
                    "Received duplicate parameter {}".format(new_parameter.name)
                )
            tm.add_param_from_top(new_parameter, component)
            # print("added", new_parameter)
        except PrefixError:
            pass

    return tm


def get_model_new(parfile):
    """Build model from a parfile.
    Parameters
    ----------
    name : str
        Name for the model.
    parfile : str
        The parfile name.
    Returns
    -------
    pint.models.timing_model.TimingModel
        The constructed model with parameter values loaded.
    """
    tm = choose_model(parfile)
    tm.read_parfile(parfile)
    return tm
