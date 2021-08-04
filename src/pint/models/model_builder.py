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
    AllComponents,
    TimingModel,
    ignore_prefix,
    ConflictAliasError,
    UnknownBinaryModel,
    UnknownParameter,
)
from pint.toa import get_TOAs
from pint.utils import PrefixError, interesting_lines, lines_of, split_prefixed_name

log = logging.getLogger(__name__)


__all__ = ["ModelBuilder", "get_model", "get_model_and_toas"]

default_models = ["StandardTimingModel"]


class ComponentConflict(ValueError):
    """Error for mulitple components can be select but no other indications."""


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

    def __call__(self, parfile):
        """Callable object for making a timing model from .par file.

        Parameter
        ---------
        parfile: str or file-like object
            Input .par file name or string contents
        Return
        ------
        pint.models.timing_model.TimingModel
            The result timing model based on the input .parfile or file object.
        """
        param_inpar, repeat_par = self.parse_parfile(parfile)
        selected, conflict, param_not_in_pint = self.choose_model(param_inpar)
        selected.update(set(self.default_components))
        # Report conflict
        if len(conflict) != 0:
            self._report_conflict(conflict)
        # Make timing model
        cps = [self.all_components.components[c] for c in selected]
        tm = TimingModel(components=cps)
        # Add indexed parameters
        leftover_params = set(param_not_in_pint)
        # Give repeatable parameters an index.
        for k, v in repeat_par.items():
            for ii in range(len(v)):
                leftover_params.add(k + str(ii + 1))
        tm, unknown_param = self._add_indexed_params(tm, leftover_params)
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

    def parse_parfile(self, parfile):
        """Parse the par file for model buinding.

        Parameter
        ---------
        parfile: str or file-like object
            Input .par file name or string contents
        Return
        ------
        dict
            The unique parameters in .par file with the key is the parfile line.
        dict
            The parameters that have the same names in the .parfile or file-like
            object.
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
            if k[0] in self.all_components.repeatable_param:
                repeat_par[k[0]].append(k[1:])
            else:
                if multi_line[k[0]] > 1:
                    log.info(
                        "Lines with duplicate keys in par file:"
                        " {} and {}".format(k[0], k[1:])
                    )
        return param_inpar, repeat_par

    def choose_model(self, param_inpar):
        """Choose the model components based on the parfile.

        Parameter
        ---------
        param_inpar: str
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
        # 1. iteration read parfile with a no component timing_model to get
        # the overall control parameters. This will get us the binary model name
        # build the base fo the timing model
        binary = param_inpar.get("BINARY", None)
        if binary is not None:
            binary = binary[0]
            binary_cp = self.all_components.search_pulsar_system_components(binary)
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
            if len(cps) == 1:  # No conflict, parameter only shows in one component.
                # Check if it is a binary component, if yes, skip. It is
                # controlled by the BINARY tag
                if self.all_components.components[cps[0]].category == "pulsar_system":
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
        return selected_components, conflict_components, param_not_in_pint

    def _add_indexed_params(self, timing_model, indexed_params):
        """Add the parameters with unknown number/indexed in parfile (maskParameter/prefixParameter) to timing model.

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
            try:
                pint_p, first_init_par = self.all_components.alias_to_pint_param(pp)
            except UnknownParameter:
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
    """A one step function to build model from a parfile.

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
