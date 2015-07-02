# __init__.py for PINT models/ directory
"""This module contains implementations of pulsar timing models.
"""
import importlib 
# Import the main timing model classes
from .timing_model import TimingModel, generate_timing_model

# Import all standard model components here
from .astrometry import Astrometry
from .dispersion import Dispersion
from .spindown import Spindown
from .dd import DD
from .bt import BT
from .solar_system_shapiro import SolarSystemShapiro
from .polycos import Polycos

# Define a standard basic model
StandardTimingModel = generate_timing_model("StandardTimingModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro))
BTTimingModel = generate_timing_model("BTTimingModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro, BT))
DDTimingModel = generate_timing_model("DDTimingModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro, DD))
PolycosModel = generate_timing_model("PolycosModel",
        (Astrometry, Spindown, Dispersion, SolarSystemShapiro, Polycos))


class model_constructor(object):
    """A class for model construction interface.
        Parameters
        ---------
        name : str
            Name for the model.
        Return
        ---------
        A class contains the result model if parfile is provided and method to 
        build the model.

        Examples:
        ---------
        Read model from parfile :
        mc = model_constructor("PulsarJ1955", parfile ="J1955.par" )
        psrJ1955 = mc.model()
        psrJ1955.read_parfile("J1955.par") # It should be the same par file

        Build model from sketch:
        from .polycos import Polycos 
        mc = model_constructor("polyco_model")
        mc.add_components(Polycos)
        model = mc.build_model()
    """
    def __init__(self, name, parfile = None):
        self.name = name
        self.binary_models = {'DD':DD,'BT':BT}
        self.components = [Astrometry, Spindown, Dispersion,
						   SolarSystemShapiro]
        self.model = None
        if parfile is not None:
            self.parFile =  parfile
            self.model = self.get_model_from_parfile(self.parFile)

    def __str__(self):
        result = ""
        for c in self.components:
            result += 'Components in the model: '+str(c)+'\n'
        result += str(self.model)
        return result

    def get_model_from_parfile(self,parfile):
        """Preprocess the par file to find the right model. 
        """
        T2_model_params = ['BINARY','NO_SS_SHAPIRO']
        model_params_infile = {}
        pfile = open(parfile, 'r')
        for l in [pl.strip() for pl in pfile.readlines()]:
            # Skip blank lines
            if not l:
                continue        	
                # Skip commented lines
            if l.startswith('#') or l[:2]=="C ":
                continue
            k = l.split()
            if k[0] in T2_model_params:
        	   model_params_infile[k[0]]=k[1]
        #TODO: need more parameters identifyer. 
        # Identify binary model
        if 'BINARY' in model_params_infile.keys():
    	    try:
                self.components.append(self.binary_models[model_params_infile['BINARY']])
            except:
                raise(RuntimeError('Binary Model '+k[1]+
        						  ' has not been imported.'))
        # Solar sys shaprio delay on or not. 
        if 'NO_SS_SHAPIRO' in model_params_infile.keys():
            components.remove(SolarSystemShapiro)

        pfile.close()
        self.model = self.build_model()

        return self.model
    
    def build_model(self):
        """ Return a model with all components listed in the self.components 
        list.
        """
        self.model = generate_timing_model(self.name,tuple(self.components))
        return self.model

    def add_components(self,components):
        """ Add new components to constructing model. 
        """
        if not isinstance(components,list):
    	   components = [components,]
        for c in components:
    	    if c in self.components:
    	        continue

            self.components.append(c)

    def add_binary_model(self, binary_model_name, binary_model):
        """ Add a new binary model
        """
        if binary_model_name in self.binary_models.keys():
            raise(ValueError('Binary model name '+binary_model_name+
                             ' have been used.'))
        if binary_model in self.binary_models.values():
            print ('Binary model'+str(binary_model)+
                   ' have been included in binary model list.')
        self.binary_models['binary_model_name'] = binary_model
