# model_builder.py
# Defines the automatic timing model generator interface
import os

# The timing models that we will be using
from .timing_model import TimingModel, Component
from pint.utils import split_prefixed_name
from .parameter import prefixParameter
import inspect, fnmatch
import glob
import sys


default_models = ["StandardTimingModel",]
DEFAULT_ORDER = ['astrometry', 'jump_delay', 'solar_system_shapiro',
                 'dispersion', 'pulsar_system','frequency_dependent', 'spindown']
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

        Return
        ---------
        A class contains the result model instance if parfile is provided and
        method to build the model.

        Examples
        --------
    """
    def __init__(self, parfile = None, name=''):
        self.timing_model = None
        self.name = name
        self.param_inparF = None
        self.param_unrecognized = {}
        self.param_inModel = []
        self.param_prefix = {}

        self.select_comp = {}
        self.control_params = ['EPHEM', 'CLK']
        if parfile is not None:
            self.parfile = parfile
            self.build_model(self.parfile, self.name)

    def __str__(self):
        result = 'Model name : ' + self.name + '\n'
        result += 'Components in the model : \n'
        for c in self.select_comp:
            result += '    '+str(c)+'\n'
        if self.model_instance is not None:
            result += 'Read parameters from : '+ self.parfile +'\n'
            result += 'The model instance is :\n'+str(self.model_instance)

        return result

    def preprocess_parfile(self, parfile):
        """Preprocess the par file.
        Return
        ---------
        A dictionary with all the parfile parameters with values in string
        """
        param = {}
        repeat_par = {}
        pfile = open(parfile, 'r')
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue
                # Skip commented lines
            if l.startswith('#') or l[:2]=="C ":
                continue
            k = l.split()
            if k[0] in param.keys(): # repeat parameter TODO: add JUMP1 even there is only one
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
        comp_category = {}
        for k, cp in list(Component._component_list.items()):
            ci = cp()
            category = ci.category
            if category not in list(comp_category.keys()):
                comp_category[category] = [ci,]
            else:
                comp_category[category].append(ci)
        return comp_category

    def get_comp_from_parfile(self, parfile):
        """Right now we only have one component on each category.
        """
        params_inpar = self.preprocess_parfile(parfile)
        comp_categories = self.get_all_categories()
        for cat, cmps in list(comp_categories.items()):
            selected_c = None
            for cpi in cmps:
                if cpi.component_special_params != []:
                    if any(par in params_inpar.keys() for par in \
                           cpi.component_special_params):
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
        """
        This is a function to sort the component order.
        Parameter
        ---------
        category_order: list, optional
           The order for the order sensitive component categories.
        Note
        ----
        If a category is not listed in the category_order, it will be treated
        as order non-sensitive category and put in the end of sorted order list.
        """
        cur_category = list(self.select_comp.keys())
        all_categories = list(self.get_all_categories().keys())
        sorted_components = []
        for cat in all_categories:
            if cat not in category_order:
                category_order.append(cat)
        for co in category_order:
            if co not in cur_category:
                continue
            cp = self.select_comp[co]
            sorted_components.append(cp)
        return sorted_components

    def search_prefix_param(self, paramList, prefix_inModel):
        """ Check if the Unrecognized parameter has prefix parameter
        """
        prefixs = {}
        for pn in prefix_inModel:
            try:
                pre,idxstr,idxV = split_prefixed_name(pn)
                prefixs[pre] = []
            except:
                continue

        for p in paramList:
            try:
                pre,idxstr,idxV = split_prefixed_name(p)
                if pre in prefixs.keys():
                    prefixs[pre].append(p)
            except:
                continue

        return prefixs

    def build_model(self, parfile=None, name=''):
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
                parName+= getattr(self.timing_model, p).aliases

            parName += param_inModel.keys()

            for pp in self.param_inparF.keys():
                if pp not in parName:
                    self.param_unrecognized[pp] = self.param_inparF[pp]

            for ptype in ['prefixParameter', 'maskParameter']:
                prefix_in_model = self.timing_model.get_params_of_type(ptype)
                prefix_param = \
                    self.search_prefix_param(
                        list(self.param_unrecognized.keys()),
                        prefix_in_model)
                for key in prefix_param.keys():
                    ppnames = [x for x in prefix_in_model if x.startswith(key)]
                    for ppn in ppnames:
                        pfx, idxs, idxv = split_prefixed_name(ppn)
                        if pfx == key:
                            exm_par = getattr(self.timing_model, ppn)
                        else:
                            continue
                    exm_par_comp = param_inModel[exm_par.name]
                    for parname in prefix_param[key]:
                        pre,idstr,idx = split_prefixed_name(parname)
                        if idx == exm_par.index:
                            continue
                        if hasattr(exm_par, 'new_param'):
                            new_par = exm_par.new_param(idx)
                            self.timing_model.add_param_from_top(new_par,
                                                                 exm_par_comp)

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
        ---------
        name : str
            Name for the model.
        parfile : str
            The parfile name

        Return
        ---------
        Model instance get from parfile.
    """
    name = os.path.basename(os.path.splitext(parfile)[0])

    mb = ModelBuilder(parfile)
    return mb.timing_model
