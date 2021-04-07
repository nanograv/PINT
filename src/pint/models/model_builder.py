import logging
import os
import tempfile
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


__all__ = ["get_model"]

default_models = ["StandardTimingModel"]


class UnknownBinaryModel(ValueError):
    """Signal that the par file requested a binary model no in PINT."""


class ModelBuilder:
    def __init__(self):
        self.components = {}
        for k, v in Component.component_types.items():
            self.components[k] = v()
        self.timing_model = TimingModel()

    def __call__(self, parfile):
        pass

    @lazyproperty
    def param_component_map(self):
        """ Return the parameter to component map.
        """
        p2c_map = {}
        for k, cp in self.components.items():
            for p in cp.params:
                if p not in p2c_map.keys():
                    p2c_map[p] = [k]
                else:
                    p2c_map[p].append(k)
        for tp in self.timing_model.params:
            if tp not in p2c_map.keys():
                p2c_map[tp] = ['timing_model']
            else:
                p2c_map[tp].append('timing_model')
        return p2c_map

    @lazyproperty
    def param_alias_map(self):
        """ Return the aliases map of all parameters
        """
        alias = {}
        for k, cp in self.components.items():
            for p in cp.params:
                par = getattr(cp, p)
                alias[p] = p
                for als in par.aliases:
                    alias[als] = p
        for tp in self.timing_model.params:
            par =  getattr(self.timing_model, tp)
            alias[tp] = tp
            for als in par.aliases:
                alias[als] = tp
        return alias

    @lazyproperty
    def repeatable_param(self):
        """ Return the repeatable parameter map.
        """
        repeatable = []
        for k, cp in self.components.items():
            for p in cp.params():
                par = getattr(cp, p)
                if par.repeatable:
                    repeatable.append(p)
        return repeatable

    @lazyproperty
    def category_component_map(self):
        category = {}
        for k, cp in self.components.items():
            cat = cp.category
            if cat in category.keys():
                category[cat].append(k)
            else:
                category[cat] = [k]
        return category

    @lazyproperty
    def component_category_map(self):
        cp_ca = {}
        for k, cp in self.components.items():
            cp_ca[k] = cp.category
        return cp_ca

    def parse_parfile(self, parfile):
        """Preprocess the par file.
        Return
        ------
        A dictionary with all the parfile parameters with values in string
        """
        param = {}
        repeat_par = {}
        param_inpar = []
        for l in interesting_lines(lines_of(parfile), comments=("#", "C ")):
            # Skip blank lines
            if not l:
                continue
            # Skip commented lines
            if l.startswith("#") or l[:2] == "C ":
                continue
            k = l.split()
            if k[0] in param.keys():  # repeat parameter TODO: add JUMP1 even there is only one
                if k[0] in repeat_par.keys():
                    repeat_par[k[0]] += 1
                else:
                    repeat_par[k[0]] = 2
                param[k[0] + str(repeat_par[k[0]])] = k[1:]
            else:
                param[k[0]] = k[1:]
        param_inpar = param
        for key in repeat_par.keys():
            param_inpar[key + str(1)] = param_inpar.pop(key)
        return param_inpar

    def search_pulsar_system_components(self, system_name):
        """ Search the component for the pulsar system, mostly binaries.
        """
        all_systems = self.categories['pulsar_system']
        result = None
        # Search the system name first
        if system_name in all_systems:
            result = self.components[system_name]
        else: # search for the pulsar system aliases
            for cp_name in all_systems:
                if system_name == self.components[cp_name].binary_model_name:
                    result = self.components[cp_name]
                else:
                    continue
        if result is None:
            raise ValueError("Pulsar system/Binary model component {}"
                             " is not provided.".format(system_name))
        else:
            return result

    def alias_2_pint_param(self, alias):
        """ Translate the alias to a PINT parameter name.
        """
        pint_par = self.param_alias_map.get(alias, None)
        # If it is not in the map, double check if it is a repeatable par.
        if pint_par is None:
            try:
                prefix, _, _ = split_prefixed_name(alias)
                pint_par = self.param_alias_map.get(prefix, None)
            except PrefixError:
                pint_par = None
        return pint_par

    def _get_param_overlap_rank(self, components, params, primary_key):
        """ Get the rank of how components match a given set of paramters

        Parameters
        ----------
        components: list
            A list of the component names.
        params: list
            A list of matching parameter names.
        primary_key: str
            The primary compare key(e.g., `params` or `component_special_params`)

        Return
        ------
        A sorted tuple list based on how many overlap parameters.
        """
        param_match = {}
        for cp in components:
            check_param = getattr(self.components[cp], primary_key)
            overlap = list(set(params) & set(check_param))
            param_match[cp] = len(overlap)
        # Selected the best overlap
        # Get the max overlap
        rank = sorted(list(param_match.items()), key=lambda tup: tup[1])
        return rank

    def choose_model(self, parfile):
        """ Choose the model components based on the parfile.

        Parameter
        ---------
        parfile: str
            parfile name and StringIO of parfile.

        Return
        ------
        List of selected components and a dictionary of conflict components.

        Note
        ----
        The selection algorithm:
        1. Look at the BINARY in the par file and catche the indicated binary model
        2. Translate para file parameters to the pint parameter name
        3. Get all the components which holds the parameters
        4. Check if there is any conflict in the component(i.e., mulitple
           components in one category)
          4.1 Find the overlap between component special special parameter and
              in parfile component.
        """
        result_componets = []
        param_inpar = self.parse_parfile(parfile)
        # 1. iteration read parfile with a no component timing_model to get
        # the overall control parameters. This will get us the binary model name
        # build the base fo the timing model
        self.timing_model.read_parfile(parfile)
        # Get the binary model name
        binary = self.timing_model.BINARY.value
        if binary is not None:
            result_componets.append(self.search_pulsar_system_components(binary))
        # 2. Get the component list from the parameters in the parfile.
        # 2.1 Check the aliases of input parameters.
        # This does not include the repeating parameters, but it should not
        # matter in the component selection.
        unrec_param = [] # For Unrecognized parameters.
        pint_par = []
        for pp in param_inpar:
            p_name = self.alias_2_pint_param(pp)
            if p_name is not None:
                pint_par.append(p_name)
            else:
                unrec_param.append(pp)

        param_components_inpar = {p: self.param_component_map[p] for p in pint_par}
        # Back map the possible_components and the parameters in the parfile
        # This will remove the duplicate components.
        possible_components = {}
        for k, cps in param_components_inpar.items():
            for cp in cps:
                if cp == 'timing_model': # skip the timing model parameters.
                    continue
                if cp in possible_components.keys():
                    possible_components[cp].append(k)
                else:
                    possible_components[cp] = [k]
        # Get conflicting components based on categories
        # conflict_components = {category: [component1, component2, ...]}
        conflict_components = {}
        for k in possible_components.keys():
            cate = self.component_category_map[k]
            if cate in conflict_components.keys():
                conflict_components[cate].append(k)
            else:
                conflict_components[cate] = [k]
        # 3. resolve the conflicting components, mulitple components in the same
        # category. They generally share some same parameters
        resolved = []
        for ca, cps in conflict_components.items():
            # skip the pulsar system, since it is selected by the BINARY parameter
            if ca == 'pulsar_system':
                if binary is not None:
                    raise ValueError("Pulsar system is set by parameter `BINARY`.")
                resolved.append(ca)
                continue
            # Only one component selected, no confilicte
            if len(cps) == 1:
                result_componets.append(self.components[cps[0]])
                resolved.append(ca)
                continue
            # There is a conflict in the categories
            rank = self._get_param_overlap_rank(cps, pint_par, 'component_special_params')
            # First check the special parameters, the most special-parameter
            #-overlap component will be selected
            # Check if there are more than one components have the same rank
            # If the highest rank has the same rank with the second highest,
            # the conflict is not resolved by speical parameters.
            if rank[1][1] != rank[0][1]: # has an unique highest case.
                result_componets.append(self.components[rank[0][0]])
                resolved.append(ca)
            else:
                # First no overlap case, which should mean no special parameters
                # in the component, will do a search depends on the overlap of
                # all parameters.
                if rank[0][1] == 0:
                    # Recompute the ranks
                    rank = self._get_param_overlap_rank(cps, pint_par, 'params')
                    if rank[1][1] != rank[0][1]:
                        result_componets.append(self.components[rank[0][0]])
                        resolved.append(ca)
        # remove resolved from the conflict dict
        for r in resolved:
            del conflict_components[r]
        return result_componets, conflict_components, unrec_param

    def add_repeat_params(self):
        """ Add the repeat parameters
        """
        pass

    def report_conflict(self):
        """ Report conflict components
        """
        pass

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
    try:
        contents = parfile.read()
    except AttributeError:
        contents = None
    if contents is None:
        # parfile is a filename and can be handled by ModelBuilder
        mbmodel = ModelBuilder(parfile)
        model = mbmodel.timing_model
        model.name = parfile
        return model
    else:
        with tempfile.TemporaryDirectory() as td:
            fn = os.path.join(td, "temp.par")
            with open(fn, "wt") as f:
                f.write(contents)
            return ModelBuilder(fn).timing_model


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
    # mm = get_model(parfile)
    # tt = get_TOAs(
    #     timfile,
    #     model=mm,
    #     ephem=ephem,
    #     include_bipm=include_bipm,
    #     bipm_version=bipm_version,
    #     include_gps=include_gps,
    #     planets=planets,
    #     usepickle=usepickle,
    #     tdb_method=tdb_method,
    #     picklefilename=picklefilename,
    # )
    # return mm, tt


def choose_model(
    parfile, category_order=None, name=None, check_for_missing_parameters=False
):
    """Determine which model components are appropriate for parfile."""
    pass
    # if name is None:
    #     if isinstance(parfile, str):
    #         name = os.path.basename(parfile)
    #     else:
    #         name = ""
    # if category_order is None:
    #     category_order = DEFAULT_ORDER
    #
    # models_by_category = defaultdict(list)
    # for k, c_type in Component.component_types.items():
    #     models_by_category[c_type.category].append(c_type)
    #
    # par_dict = {}
    # par_lines = []
    # multi_tags = set(
    #     [
    #         "JUMP",
    #         "ECORR",
    #         "T2EFAC",
    #         "T2EQUAD",
    #         "EQUAD",
    #         "EFAC",
    #         "DMJUMP",
    #         "DMEFAC",
    #         "DMEQUAD",
    #     ]
    # )
    # multi_line = Counter()
    # for l in interesting_lines(lines_of(parfile), comments=("#", "C ")):
    #     ll = l.split()
    #     k = ll[0]
    #     if k in multi_tags:
    #         multi_line[k] += 1
    #         k = k + str(multi_line[k])
    #     if k in par_dict:
    #         # FIXME: what happens with JUMPs?
    #         log.info(
    #             "Lines with duplicate keys in par file: {} and {}".format(
    #                 [k] + par_dict[k], ll
    #             )
    #         )
    #     par_dict[k] = ll[1:]
    #     par_lines.append(l)
    #
    # models_to_use = {}
    # for category, models in models_by_category.items():
    #     acceptable = []
    #     for m_type in models:
    #         m = m_type()
    #         if m.is_in_parfile(par_dict):
    #             acceptable.append(m)
    #     if len(acceptable) > 1:
    #         raise ValueError(
    #             "Multiple models are compatible with this par file: {}".format(
    #                 acceptable
    #             )
    #         )
    #     if acceptable:
    #         models_to_use[category] = acceptable[0]
    #
    # if "BINARY" in par_dict:
    #     vals = par_dict["BINARY"]
    #     if len(vals) != 1:
    #         raise ValueError(
    #             "Mal-formed binary model selection: {}".format(
    #                 repr(" ".join(["BINARY"] + vals))
    #             )
    #         )
    #     (bm,) = vals
    #     if "pulsar_system" not in models_to_use:
    #         # Either we're missing parameters or the model is bogus
    #         # FIXME: distinguish
    #         raise UnknownBinaryModel(
    #             "Unknown binary model requested in par file: {}".format(bm)
    #         )
    #     # FIXME: consistency check - the componens actually chosen should know the name bm
    #
    # models_in_order = []
    # for category in category_order:
    #     try:
    #         models_in_order.append(models_to_use.pop(category))
    #     except KeyError:
    #         pass
    # models_in_order.extend(v for k, v in sorted(models_to_use.items()))
    # tm = TimingModel(name, models_in_order)
    #
    # # FIXME: this should go in TimingModel for when you try to
    # # add conflicting components
    # alias_map = {}
    # for prefix_type in ["prefixParameter", "maskParameter"]:
    #     for pn in tm.get_params_of_type_top(prefix_type):
    #         par = getattr(tm, pn)
    #         for a in [par.prefix] + par.prefix_aliases:
    #             if a in alias_map:
    #                 raise ValueError(
    #                     "Two prefix/mask parameters have the same "
    #                     "alias {}: {} and {}".format(a, alias_map[a], par)
    #                 )
    #             alias_map[a] = par
    #
    # leftover_params = par_dict.copy()
    # for k in tm.get_params_mapping():
    #     leftover_params.pop(k, None)
    #     for a in getattr(tm, k).aliases:
    #         leftover_params.pop(a, None)
    #
    # for p in leftover_params:
    #     try:
    #         pre, idxstr, idxV = split_prefixed_name(p)
    #         try:
    #             par = alias_map[pre]
    #         except KeyError:
    #             if pre in ignore_prefix:
    #                 # log.warning("Ignoring unhandled prefix {}".format(pre))
    #                 continue
    #             else:
    #                 raise ValueError(
    #                     "Mystery parameter {}, prefix {} with number {}".format(
    #                         p, pre, idxV
    #                     )
    #                 )
    #         component = tm.get_params_mapping()[par.name]
    #         new_parameter = par.new_param(idxV)
    #         if hasattr(tm, new_parameter.name):
    #             raise ValueError(
    #                 "Received duplicate parameter {}".format(new_parameter.name)
    #             )
    #         tm.add_param_from_top(new_parameter, component)
    #         # print("added", new_parameter)
    #     except PrefixError:
    #         pass
    #
    # return tm


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
    pass
    # tm = choose_model(parfile)
    # tm.read_parfile(parfile)
    # return tm
