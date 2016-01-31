# model_builder.py
# Defines the automatic timing model generator interface
import os

# The timing models that we will be using
from .timing_model import generate_timing_model
from .astrometry import Astrometry
from .dispersion import Dispersion
from .spindown import Spindown
from .glitch import Glitch
from .bt import BT
from .solar_system_shapiro import SolarSystemShapiro

from pint.utils import split_prefixed_name
from .parameter import prefixParameter
from .pint_dd_model import DDwrapper

# List with all timing model components we will consider when pre-processing a
# parfile

ComponentsList = [Astrometry, Spindown, Dispersion, SolarSystemShapiro,
                  BT, DDwrapper, Glitch]


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
        pfile = open(parfile, 'r')
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue
                # Skip commented lines
            if l.startswith('#') or l[:2]=="C ":
                continue
            k = l.split()
            param[k[0]] = k[1:]
        self.param_inparF = param
        pfile.close()
        return self.param_inparF


    def get_comp_from_parfile(self,parfile):
        """Check all the components if it is needed from parfile
           Put the needed on in the selected components list
        """
        params_inpar = self.preprocess_parfile(parfile)
        comp_classes = [c() for c in self.comps]
        for c in self.comps:
            cclass = c()
            if cclass.is_in_parfile(params_inpar):
                if c not in self.select_comp:
                    self.select_comp.append(c)

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

    def search_prefix_param(self,paramList,prefixList):
        """ Check if the Unrecognized parameter has prefix parameter
        """
        for pn in prefixList:
            self.param_prefix[pn] = []
            pnlen = len(pn)
            for p in paramList:
                try:
                    pre,idxstr,idxV = split_prefixed_name(p)
                except:
                    continue
                if pre == pn:
                    self.param_prefix[pn].append(p)

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

            self.search_prefix_param(self.param_unrecognized.keys(),self.prefix_names)

            if self.param_prefix != {}:
                for p in self.param_prefix.keys():
                    ppnames = [x for x in self.model_instance.params if x.startswith(p)]
                    for ppn in ppnames:
                        pfxp = getattr(self.model_instance,ppn)
                        if pfxp.is_prefix is True:
                            for pp in self.param_prefix[p]:
                                pre,idstr,idx = split_prefixed_name(pp)
                                if idx == pfxp.index:
                                    continue
                                newPfxp = pfxp.new_index_prefix_param(idx)
                                self.model_instance.add_param(newPfxp)

        if parfile is not None:
            self.model_instance.read_parfile(parfile)

        return self.model_instance


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

def add_prefixed_param(modelIns,names):
    """Add a prefixed parameter into the timing model.
       Parameter
       ----------
       modelIns : PINT Timing_model class
           The timing model of parameters adding to
       names : string or list of strings
           The name list of prefixed parameters.s
    """
    if type(names) is str:
        names = [names,]
    for n in names:
        prefix,indexformat,indexV = split_prefixed_name(n)
        if prefix in modelIns.prefix_params:
            units = modelIns.prefix_params_units[prefix]
            des = modelIns.prefix_params_description[prefix]

        modelIns.add_param(prefixParameter(name = n,units = units,value = 0.0,
                           description = des))
