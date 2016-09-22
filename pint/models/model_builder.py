# model_builder.py
# Defines the automatic timing model generator interface
import os

# The timing models that we will be using
from .timing_model import generate_timing_model, TimingModel
from pint.utils import split_prefixed_name
from .parameter import prefixParameter
import os, inspect, fnmatch
import glob
import sys

def get_componets():
    timing_comps = {}
    path = os.path.dirname(os.path.abspath(__file__))
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.py'):
            if filename == '__init__.py':
                continue
            mod_root_start = root.find('pint/models')
            if mod_root_start + len('pint/models') > len(root):
                mod_root = ''
            else:
                mod_root = root[mod_root_start + len('pint/models/'):]
            mod = os.path.join(mod_root, filename).replace("/", ".")[:-3]
            exec('import %s as tmp' % mod)
            s = set()
            for k, v in tmp.__dict__.items():
                if inspect.isclass(v) and issubclass(v, TimingModel):
                    if k == 'TimingModel':
                        continue
                    s.add(v)
                if s != set():
                    timing_comps[tmp.__name__] = s
    return timing_comps

ComponentsList = get_componets()

default_models = ["StandardTimingModel",]
class model_builder(object):
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

        Examples:
        ---------
        Read model from parfile :
        [1] mb = model_builder("PulsarJ1955", parfile ="J1955.par" )
        [2] psrJ1955 = mb.model_instance

        Build model from sketch and read parfile:
        [1] from .bt import BT
        [2] mb = model_builder("BT_model")
        [3] mb.add_components(BT)
        [4] psrJ1955 = mb.get_model_instance('J1955.par')

        Build model instance without reading parfile:
        [1] mb = model_builder("BT_model")
        [2] mb.add_components(BT)
        [3] myModel = mb.get_model_instance()

    """
    def __init__(self, name, parfile = None):
        self.name = name
        self.model_instance = None
        self.param_inparF = None
        self.param_unrecognized = {}
        self.param_inModel = []
        self.comps = ComponentsList
        self.prefix_names = None
        self.param_prefix = {}
        self.select_comp = []
        if parfile is not None:
            self.parfile = parfile
            self.get_comp_from_parfile(self.parfile)
            self.get_model_instance(self.parfile)

    def __str__(self):
        result = 'Model name : ' + self.name + '\n'
        result += 'Components in the model : \n'
        for c in self.select_comp:
            result += '    '+str(c)+'\n'
        if self.model_instance is not None:
            result += 'Read parameters from : '+ self.parfile +'\n'
            result += 'The model instance is :\n'+str(self.model_instance)

        return result

    def preprocess_parfile(self,parfile):
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


    def get_comp_from_parfile(self,parfile):
        """Check all the components if it is needed from parfile
           Put the needed on in the selected components list
        """
        params_inpar = self.preprocess_parfile(parfile)
        for module in self.comps.keys():
            selected_c = None
            num_comp = len(self.comps[module])
            slt_tmp = None
            for ii, c in enumerate(self.comps[module]):
                cclass = c()
                #Check is this components a subclass of other components
                if TimingModel not in c.__bases__:
                    if hasattr(cclass,'model_special_params'):
                        # TODO : Need fix aliases part
                        if any(par in params_inpar.keys() for par in cclass.model_special_params):
                            selected_c = c
                            break
                        else:  # If no special parameters, ignore.
                            continue

                if cclass.is_in_parfile(params_inpar):
                        selected_c = c
            # One module will have one selected component
            if selected_c is not None and selected_c not in self.select_comp:
                self.select_comp.append(selected_c)

    def build_model(self):
        """ Return a model with all components listed in the self.components
        list.
        """
        if self.select_comp ==[]:
            raise(RuntimeError("No timing model components selected."))

        return generate_timing_model(self.name,tuple(self.select_comp))

    def add_components(self,components):
        """ Add new components to constructing model.
        """
        if not isinstance(components,list):
    	   components = [components,]
        for c in components:
    	    if c not in self.select_comp:
    	       self.select_comp.append(c)

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

    def get_model_instance(self,parfile=None):
        """Read parfile using the model_instance attribute.
            Parameters
            ---------
            parfile : str optional
                The parfile name
        """
        if self.model_instance is None:
            model = self.build_model()

        self.model_instance = model()
        self.param_inModel = self.model_instance.params
        self.prefix_names = self.model_instance.prefix_params
        # Find unrecognised parameters in par file.

        if self.param_inparF is not None:
            parName = []
            for p in self.param_inModel:
                parName+= getattr(self.model_instance,p).aliases

            parName += self.param_inModel

            for pp in self.param_inparF.keys():
                if pp not in parName:
                    self.param_unrecognized[pp] = self.param_inparF[pp]

            for ptype in ['prefixParameter', 'maskParameter']:
                prefix_in_model = self.model_instance.get_params_of_type(ptype)
                prefix_param = self.search_prefix_param(self.param_unrecognized.keys(),
                                                        prefix_in_model)
                for key in prefix_param.keys():
                    ppnames = [x for x in prefix_in_model if x.startswith(key)]
                    exm_par = getattr(self.model_instance,ppnames[0])
                    for parname in prefix_param[key]:
                        pre,idstr,idx = split_prefixed_name(parname)
                        if idx == exm_par.index:
                             continue
                        if hasattr(exm_par, 'new_param'):
                            new_par = exm_par.new_param(idx)
                            self.model_instance.add_param(new_par)

        if parfile is not None:
            self.model_instance.read_parfile(parfile)

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

    mb = model_builder(name,parfile)
    return mb.model_instance
