"""The BT (Blandford & Teukolsky) model.

See Blandford & Teukolsky 1976, ApJ, 205, 580.
"""
import astropy.units as u
from astropy.table import Table
from pint.models.parameter import (
    MJDParameter,
    floatParameter,
    prefixParameter,
    maskParameter,
)
import numpy as np
from pint.toa_select import TOASelect
from pint import GMsun, Tsun, ls
from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.BT_model import BTmodel
from pint.models.timing_model import MissingParameter, TimingModel
from pint.models.stand_alone_psr_binaries.BT_piecewise import BTpiecewise

class BinaryBTPiecewise(PulsarBinary):
    """Model implemenring the BT model.

    This is a PINT pulsar binary BT model class a subclass of PulsarBinary.
    It is a wrapper for stand alone BTmodel class defined in
    ./stand_alone_psr_binary/BT_model.py
    All the detailed calculations are in the stand alone BTmodel.
    The aim for this class is to connect the stand alone binary model with PINT platform
    BTmodel special parameters:
    GAMMA Binary Einsten delay coeeficient
    """

    register = True

    def __init__(self):
        
        super(BinaryBTPiecewise, self).__init__()
        self.binary_model_name = "BT_piecewise"
        self.binary_model_class = BTpiecewise
        self.add_param(
            floatParameter(
                name="GAMMA",
                value=0.0,
                units="second",
                description="Time dilation & gravitational redshift",
            )
        )
        self.A1_value_funcs=[]
        self.T0_value_funcs=[]
        self.remove_param("M2")
        self.remove_param("SINI")
        self.T0.value=1
        self.A1.value=1
        self.add_group_range(None, None, frozen=False, j=0)
        self.add_piecewise_param("T0","d",None,0)
        self.add_piecewise_param("A1","ls",None,0)


    def add_group_range(self,group_start_mjd,group_end_mjd,frozen=True,j=None):
        if group_end_mjd is not None and group_start_mjd is not None:
            if group_end_mjd < group_start_mjd:
                raise ValueError("Starting MJD is greater than ending MJD.")
            elif j<0: 
                raise ValueError(f"Invalid index for group: {j} should be greater than or equal to 0")
            elif j>9999:
                raise ValueError(f"Invalid index for group. Cannot index beyond 9999 (yet?)")
            
        i = f"{int(j):04d}"  
        self.add_param(
        prefixParameter(
            name="PieceLowerBound_{0}".format(i),
            units="MJD",
            unit_template=lambda x: "MJD",
            description="Beginning of paramX interval",
            description_template=lambda x: "Beginning of paramX interval",
            parameter_type="MJD",
            time_scale="utc",
            value=group_start_mjd,
        )
        )
        self.add_param(
        prefixParameter(
            name= "PieceUpperBound_{0}".format(i),
            units="MJD",
            unit_template=lambda x: "MJD",
            description="End of paramX interval",
            description_template=lambda x: "End of paramX interval",
            parameter_type="MJD",
            time_scale="utc",
            value=group_end_mjd,
        )
        )
        #print("going to binary_piecewise setup")
        self.setup()
        self.validate()
        
    def update_binary_object(self, toas=None, acc_delay=None):
        super().update_binary_object(toas=toas,acc_delay=acc_delay)
    
    
    def d_binary_delay_d_xxxx(self, toas, param, acc_delay):
        """Return the binary model delay derivtives."""
        #print(f"hello from binary piecewise parameter derivative, using toas: {toas}")
        delay = super().d_binary_delay_d_xxxx(toas=toas, param=param, acc_delay=acc_delay)
        return delay
        
    
    
        
    def remove_range(self, index):
        """Removes all DMX parameters associated with a given index/list of indices.

        Parameters
        ----------

        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to DMX indices to be removed from model.
        """

        if (
            isinstance(index, int)
            or isinstance(index, float)
            or isinstance(index, np.int64)
        ):
            indices = [index]
        elif not isinstance(index, list) or not isinstance(index, np.ndarray):
            raise TypeError(
                f"index must be a float, int, list, or array - not {type(index)}"
            )
        for index in indices:
            index_rf = f"{int(index):04d}"
            for prefix in ["T0X_","A1X_", "PieceLowerBound_", "PieceUpperBound_"]:
                self.remove_param(prefix + index_rf)
        self.setup()
        self.validate()
        
    def add_piecewise_param(self,param,param_unit,paramx,j):
        if int(j) in self.get_prefix_mapping_component("X_"):
            raise ValueError(
                "Index '%s' is already in use in this model. Please choose another."
                % j
            )
        if j is None:
            dct = self.get_prefix_mapping_component(param+"X_")
            j = np.max(list(dct.keys())) + 1
        i = f"{int(j):04d}"
        if param is "A1":
            self.add_param(
            prefixParameter(
                name=param+"X_{0}".format(i),
                units=param_unit,
                value=paramx,
                unit_template=lambda x: param_unit,
                description="Parameter" + param + "variation",
                description_template=lambda x: param,
                parameter_type="float",
                frozen=False,
            )
            )
        elif param is "T0":
            self.add_param(
            prefixParameter(
                name=param+"X_{0}".format(i),
                units=param_unit,
                value=paramx,
                unit_template=lambda x: param_unit,
                description="Parameter" + param + "variation",
                description_template=lambda x: param,
                parameter_type="float",
                frozen=False,
            )
            )
        self.setup()
    
    def setup(self):
        super(BinaryBTPiecewise, self).setup()
        for bpar in self.params:
            self.register_deriv_funcs(self.d_binary_delay_d_xxxx, bpar)
        # Setup the model isinstance
        self.binary_instance = self.binary_model_class()
        # Setup the FBX orbits if FB is set.
        # TODO this should use a smarter way to set up orbit.
        T0X_mapping = self.get_prefix_mapping_component("T0X_")
        T0Xs = {}
        A1X_mapping = self.get_prefix_mapping_component("A1X_")
        A1Xs = {}
        XR1_mapping = self.get_prefix_mapping_component("PieceLowerBound_")
        XR1s = {}
        XR2_mapping = self.get_prefix_mapping_component("PieceUpperBound_")
        XR2s = {}
        
        for t0n in T0X_mapping.values():
            #print("hello from T0 mapping")
            T0Xs[t0n] = getattr(self, t0n).quantity
        #if None in T0Xs.values():
            #print("Group with non-defined T0X, applying default T0 to group")
            #TODO set default T0 value
        if None not in T0Xs.values():
            for t0_name, t0_value in T0Xs.items():
                self.binary_instance.add_binary_params(t0_name, t0_value)
        
        
        for a1n in A1X_mapping.values():
            #print("hello from A1 mapping")
            A1Xs[a1n] = getattr(self, a1n).quantity
        
        #if None in A1Xs.values():
            #print("Group with non-defined A1X, applying default A1 to group")
            #TODO set default A1 value
            
        if None not in A1Xs.values():
            #print(len(A1Xs.items()))
            for a1_name, a1_value in A1Xs.items():
                self.binary_instance.add_binary_params(a1_name, a1_value)
        #
        for XR1n in XR1_mapping.values():
            #lower bound mapping
            #print("hello from A1 mapping")
            XR1s[XR1n] = getattr(self, XR1n).quantity
        
        #if None in XR1s.values():
            #print("Group with non-defined XR1, applying default A1 to group")
            #TODO set default A1 value
            
        if None not in XR1s.values():
            #print(len(A1Xs.items()))
            for xr1_name, xr1_value in XR1s.items():
                self.binary_instance.add_binary_params(xr1_name, xr1_value)
        
        for XR2n in XR2_mapping.values():
            #print("hello from A1 mapping")
            XR2s[XR2n] = getattr(self, XR2n).quantity
        
        #if None in XR2s.values():
            #print("Group with non-defined XR2, applying default A1 to group")
            #TODO set default A1 value
            
        if None not in XR2s.values():
            #print(len(A1Xs.items()))
            for xr2_name, xr2_value in XR2s.items():
                self.binary_instance.add_binary_params(xr2_name, xr2_value)
        self.update_binary_object()      


    def validate(self):
        """ Validate BT model parameters UNCHANGED(?)
        """
        super(BinaryBTPiecewise, self).validate()
        for p in ("T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p, "%s is required for BT" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("BT", "T0", "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True             
    
    
    def get_group_boundaries(self):
        return self.binary_instance.get_group_boundaries()

    def which_group_is_toa_in(self, toa):
        barycentric_toa = self._parent.get_barycentric_toas(toa)
        return self.binary_instance.toa_belongs_in_group(barycentric_toa)
    
    def get_number_of_groups(self):
        return len(self.binary_instance.piecewise_parameter_information)
    
    def get_group_indexes(self):
        group_indexes = []
        for i in range(0,len(self.binary_instance.piecewise_parameter_information)):
            group_indexes.append(self.binary_instance.piecewise_parameter_information[i][0])
        return group_indexes
    
    def get_group_indexes_in_four_digit_format(self):
        group_indexes = []
        for i in range(0,len(self.binary_instance.piecewise_parameter_information)):
            group_indexes.append(f"{int(self.binary_instance.piecewise_parameter_information[i][0]):04d}")
        return group_indexes
        
    def get_T0Xs_associated_with_toas(self,toas):
        if hasattr(self.binary_instance,"group_index_array"):
            temporary_storage = self.binary_instance.group_index_array
        self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
        barycentric_toa = self._parent.get_barycentric_toas(toas)
        T0X_per_toa = self.binary_instance.piecewise_parameter_from_information_array(toas)[0]
        if temporary_storage is not None:
            self.binary_instance.group_index_array = temporary_storage
        return T0X_per_toa
    
    def get_A1Xs_associated_with_toas(self,toas):
        if hasattr(self.binary_instance,"group_index_array"):
            temporary_storage = self.binary_instance.group_index_array
        self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
        barycentric_toa = self._parent.get_barycentric_toas(toas)
        A1X_per_toa = self.binary_instance.piecewise_parameter_from_information_array(toas)[1]
        if temporary_storage is not None:
            self.binary_instance.group_index_array = temporary_storage
        return A1X_per_toa
    
    def does_toa_reference_piecewise_parameter(self,toas,param):
        #if hasattr(self.binary_instance,"group_index_array"):
        #    temporary_storage = self.binary_instance.group_index_array
        
        self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
        from_in_piece = self.binary_instance.in_piece(param)
        #if temporary_storage is not None:
        #    self.binary_instance.group_index_array = temporary_storage
        return from_in_piece[0]