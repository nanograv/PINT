# model_builder.py
# Defines the automatic timing model generator interface
from __future__ import absolute_import, division, print_function

import glob
import os
import sys
from collections import Counter, defaultdict

from astropy import log

from pint.models.timing_model import (
    Component,
    MissingParameter,
    TimingModel,
    ignore_prefix,
    DEFAULT_ORDER,
)
from pint.utils import PrefixError, interesting_lines, lines_of, split_prefixed_name

__all__ = ["get_model", "get_model_new"]

default_models = ["StandardTimingModel"]


class UnknownBinaryModel(ValueError):
    """Signal that the par file requested a binary model no in PINT."""


class ModelBuilder(object):
    """A class for model construction interface.

    Parameters
    ---------
    name : str
        Name for the model.
    parfile : str optional
        The .par file input for select components. If the parfile is
        provided the self.model_instance will be put model instance
        with .par file read in. If it is not provided, self.model_instance
        will return as None.

    Returns
    -------
    A class contains the result model instance if parfile is provided and
    method to build the model.
    """

    def __init__(self, parfile=None, name=""):
        self.timing_model = None
        self.name = name
        self.param_inparF = None
        self.param_unrecognized = {}
        self.param_inModel = []
        self.param_prefix = {}

        self.select_comp = {}
        self.control_params = ["EPHEM", "CLK"]
        if parfile is not None:
            self.parfile = parfile
            self.build_model(self.parfile, self.name)

    def __str__(self):
        result = "Model name : " + self.name + "\n"
        result += "Components in the model : \n"
        for c in self.select_comp:
            result += "    " + str(c) + "\n"
        if self.model_instance is not None:
            result += "Read parameters from : " + self.parfile + "\n"
            result += "The model instance is :\n" + str(self.model_instance)

        return result

    def preprocess_parfile(self, parfile):
        """Preprocess the par file.

        Return
        ------
        A dictionary with all the parfile parameters with values in string

        """
        param = {}
        repeat_par = {}
        pfile = open(parfile, "r")
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue
                # Skip commented lines
            if l.startswith("#") or l[:2] == "C ":
                continue
            k = l.split()
            if (
                k[0] in param.keys()
            ):  # repeat parameter TODO: add JUMP1 even there is only one
                if k[0] in repeat_par.keys():
                    repeat_par[k[0]] += 1
                else:
                    repeat_par[k[0]] = 2
                param[k[0] + str(repeat_par[k[0]])] = k[1:]
            else:
                param[k[0]] = k[1:]
        self.param_inparF = param
        for key in repeat_par.keys():
            self.param_inparF[key + str(1)] = self.param_inparF.pop(key)
        pfile.close()
        return self.param_inparF

    def get_all_categories(self,):
        """Obtain a dictionary from category to a list of instances."""
        comp_category = defaultdict(list)
        for k, cp in Component.component_types.items():
            comp_category[cp.category].append(cp())
        return dict(comp_category)

    def get_comp_from_parfile(self, parfile):
        """Right now we only have one component on each category."""
        params_inpar = self.preprocess_parfile(parfile)
        for cat, cmps in self.get_all_categories().items():
            selected_c = None
            for cpi in cmps:
                if cpi.component_special_params:
                    if any(par in params_inpar for par in cpi.component_special_params):
                        selected_c = cpi
                        # Once have match, stop searching
                        break
                    else:
                        continue
                else:
                    if cpi.is_in_parfile(params_inpar):
                        selected_c = cpi
            if selected_c is not None:
                self.select_comp[cat] = selected_c

    def sort_components(self, category_order=DEFAULT_ORDER):
        """Sort the components into order.

        Parameters
        ----------
        category_order: list, optional
           The order for the order sensitive component categories.

        Note
        ----
        If a category is not listed in the category_order, it will be treated
        as order non-sensitive category and put in the end of sorted order list.

        """
        sorted_components = []
        for cat in self.get_all_categories():
            if cat not in category_order:
                category_order.append(cat)
        for co in category_order:
            if co not in self.select_comp:
                continue
            cp = self.select_comp[co]
            sorted_components.append(cp)
        return sorted_components

    def search_prefix_param(self, paramList, model, prefix_type):
        """ Check if the Unrecognized parameter has prefix parameter
        """
        prefixs = {}
        prefix_inModel = model.get_params_of_type_top(prefix_type)
        for pn in prefix_inModel:
            par = getattr(model, pn)
            prefixs[par.prefix] = []
            for p in paramList:
                try:
                    pre, idxstr, idxV = split_prefixed_name(p)
                    if pre in [par.prefix] + par.prefix_aliases:
                        prefixs[par.prefix].append(p)
                except:  # FIXME: is this meant to catch KeyErrors?
                    continue

        return prefixs

    def build_model(self, parfile=None, name=""):
        """Read parfile using the model_instance attribute.
        Parameters
        ---------
        name: str, optional
            The name for the timing model
        parfile : str optional
            The parfile name
        """
        if parfile is not None:
            self.get_comp_from_parfile(parfile)
        sorted_comps = self.sort_components()
        self.timing_model = TimingModel(name, sorted_comps)
        param_inModel = self.timing_model.get_params_mapping()
        # Find unrecognised parameters in par file.

        if self.param_inparF is not None:
            parName = []
            # add aliases
            for p in list(param_inModel.keys()):
                parName += getattr(self.timing_model, p).aliases

            parName += param_inModel.keys()

            for pp in self.param_inparF.keys():
                if pp not in parName:
                    self.param_unrecognized[pp] = self.param_inparF[pp]

            for ptype in ["prefixParameter", "maskParameter"]:
                prefix_param = self.search_prefix_param(
                    self.param_unrecognized, self.timing_model, ptype
                )
                prefix_in_model = self.timing_model.get_params_of_type_top(ptype)
                for key in prefix_param:
                    ppnames = [x for x in prefix_in_model if x.startswith(key)]
                    for ppn in ppnames:
                        pfx, idxs, idxv = split_prefixed_name(ppn)
                        if pfx == key:
                            exm_par = getattr(self.timing_model, ppn)
                        else:
                            continue
                    exm_par_comp = param_inModel[exm_par.name]
                    for parname in prefix_param[key]:
                        pre, idstr, idx = split_prefixed_name(parname)
                        if idx == exm_par.index:
                            continue
                        if hasattr(exm_par, "new_param"):
                            new_par = exm_par.new_param(idx)
                            self.timing_model.add_param_from_top(new_par, exm_par_comp)
            if "BINARY" in self.param_inparF:
                vals = self.param_inparF["BINARY"]
                if len(vals) != 1:
                    raise ValueError(
                        "Mal-formed binary model selection: {}".format(
                            repr(" ".join(["BINARY"] + vals))
                        )
                    )
                (bm,) = vals
                cats = self.timing_model.get_components_by_category()
                if "pulsar_system" not in cats:
                    raise UnknownBinaryModel(
                        "Unknown binary model requested in par file: {}".format(bm)
                    )
                # FIXME: consistency check - the componens actually chosen should know the name bm

        if parfile is not None:
            self.timing_model.read_parfile(parfile)

    def get_control_info(self):
        info = {}
        if not self.param_unrecognized == {}:
            for ctrlp in self.control_params:
                if ctrlp in self.param_unrecognized:
                    info[ctrlp] = self.param_unrecognized[ctrlp]
                else:
                    # Check if the prefix match
                    for p in self.control_params.keys():
                        if p.startswith(ctrlp):
                            info[ctrlp] = self.param_unrecognized[ctrlp]
        return info


def get_model(parfile):
    """A one step function to build model from a parfile

    Parameters
    ----------
    name : str
        Name for the model.
    parfile : str
        The parfile name

    Returns
    -------
    Model instance get from parfile.

    """
    name = os.path.basename(os.path.splitext(parfile)[0])

    mb = ModelBuilder(parfile)
    return mb.timing_model


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
    multi_tags = set(["JUMP", "ECORR", "T2EFAC", "T2EQUAD", "EQUAD", "EFAC"])
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
                    log.warning("Ignoring unhandled prefix {}".format(pre))
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
            print("added", new_parameter)
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
