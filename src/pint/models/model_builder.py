import logging
import os
import tempfile
from collections import Counter, defaultdict

from pint.models.parameter import maskParameter
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
        for k, v in Components.component_types.items():
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
                    p2c_map[p] = [cp]
                else:
                    p2c_map[p].append(cp)
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
            for p in cp.params():
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
        pfile = open(parfile, "r")
        for l in [pl.strip() for pl in pfile.readlines()]:
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
        pfile.close()
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
            for cp_name is all_systems:
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
            except PrefixError:
                pint_par = None
            pint_par = self.param_alias_map.get(perfix, None)
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
        self.timing_model.read_par(parfile)
        # Get the binary model name
        binary = self.BINARY.value
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
                if cp in possible_components.keys():
                    possible_components[cp].append(k)
                else:
                    possible_components[cp] += [k]
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
        return result_componets, conflict_components

    def sort_components(self, components, category_order=DEFAULT_ORDER):
        """Sort the components into order.
        Parameters
        ----------
        components: list
            List of component
        category_order: list, optional
           The order for the order sensitive component categories.

        Note
        ----
        If a category is not listed in the category_order, it will be treated
        as order non-sensitive category and put in the end of sorted order list.
        """
        sorted_components = []
        for cp in components:


        self.component_category_map
        for cat in self.get_all_categories():
            # FIXME, I am not sure adding orders here is a good idea.
            if cat not in category_order:
                category_order.append(cat)
        for co in category_order:
            if co not in self.select_comp:
                continue
            cp = self.select_comp[co]
            sorted_components.append(cp)
        return sorted_components

    def add_repeat_params(self):
        """ Add the repeat parameters
        """
        pass

    def report_conflict(self):
        """ Report conflict components
        """
        pass 
