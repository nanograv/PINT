import astropy.constants as c
import astropy.units as u
import numpy as np
import pint.toa
from pint import GMsun, Tsun, ls
from pint.models.stand_alone_psr_binaries.BT_model import BTmodel
from .binary_generic import PSR_BINARY


class BTpiecewise(BTmodel):
    def __init__(self, axis_store_initial=None, t=None, input_params=None):
        self.binary_name = "BT_piecewise"
        super(BTpiecewise, self).__init__()
        self.axis_store_initial=[]
        self.extended_group_range=[]
        if t is not None:
            self._t = t
        if input_params is not None:
            if self.T0X is None:
                self.update_input(input_params)
            elif self.T0X is not None:
                self.update_input()
        self.binary_params = list(self.param_default_value.keys())
        self.param_aliases.update({"T0X": ["T0X"], "A1X": ["A1X"]})

        
    def set_param_values(self, valDict=None):
        super().set_param_values(valDict=valDict)
        self.piecewise_parameter_loader(valDict=valDict)
    
    def piecewise_parameter_loader(self, valDict=None):
        self.T0X_arr = []
        self.A1X_arr = []
        self.lower_group_edge = []
        self.upper_group_edge = []
        if valDict is None:
            self.T0X_arr = self.T0
            self.A1X_arr = self.A1
            self.lower_group_edge=[0]
            self.upper_group_edge=[1e9]
        else:
            for key, value in valDict.items():
                #print(key)
                if key[0:4] == "T0X_":
                    self.T0X_arr.append(value)
                elif key[0:4] == "A1X_":
                    self.A1X_arr.append(value)
                elif key[0:4] == "XR1_":
                    self.lower_group_edge.append(value)
                elif key[0:4] == "XR2_":
                    self.upper_group_edge.append(value)
        print(f"from piecewise_param_loader: T0X_arr = {self.T0X_arr}")
            
        
        

    
    def a1(self):
        #if self.A1X_arr is not None:
        #    print(self.A1X_arr)
        self.A1_val = self.A1X_arr*ls
        self.T0_val = self.T0X_arr*u.d
        self.tt0_piecewise = self._t-self.T0_piecewise_getter()
        ret = self.A1_piecewise_getter() + self.tt0_piecewise * self.A1DOT
        #print(ret)
        return ret
    
    def A1_piecewise_getter(self):
        """Toa: array of toas_mjds. Index: finds the group toa belongs to."""
        A1_val_arr=[]
        toa=self._t.value
        index_upper_edge=np.searchsorted(self.upper_group_edge,toa)
        #print(np.unique(index_upper_edge))
        for i in range(len(toa)):
            j = index_upper_edge[i]
            if j>=len(self.A1_val):
                A1_val_arr.append(self.A1.value)
            else:
                A1_val_arr.append(self.A1_val[j].value)
        #print(np.unique(A1_val_arr,return_counts=True))
        return A1_val_arr*ls
    
    def T0_piecewise_getter(self):
        """Toa: array of toas_mjds. Index: finds the group toa belongs to."""
        T0_val_arr=[]
        print(f"from T0_piecewise_setter: T0X arr = {self.T0X_arr}")
        toa=self._t.value
        if len(self.upper_group_edge)<=1:
            self.upper_group_edge=[0,1e9]
            #print(np.unique(self.upper_group_edge))
        index_upper_edge=np.searchsorted(self.upper_group_edge,toa)
        print(np.unique(index_upper_edge))
        
        for i in range(len(toa)):
            j = index_upper_edge[i]
            if j==0 | j==len(toa):
                T0_val_arr.append(self._T0)
                print("toa out of boundaries")
            else:
                #print("toa in boundary")
                if self.T0X_arr is list:
                    print("list of T0_vals given")
                    T0_val_arr.append(self.T0_val[j].value)
                else:
                    #print("singular T0_val")
                    T0_val_arr.append(self.T0_val[j].value)
        #print(np.unique(T0_val_arr,return_counts=True))
        #print(T0_val_arr)
        return T0_val_arr*u.d
    
    
    def create_group_boundary(self, axis_store_lower, axis_store_upper):
        self.extended_group_range = []
        for i in range(0,len(self.axis_store_lower)):
            if i==0:
                self.extended_group_range.append(self.axis_store_lower[i]-100)
            elif i<len(self.axis_store_lower):
                self.extended_group_range.append(self.axis_store_lower[i+1]-(self.axis_store_lower[i+1]-self.axis_store_upper[i])/2)
            elif i==len(self.axis_store_lower):
                self.extended_group_range.append(self.axis_store_upper[i]+100)
        self.axis_store=self.extended_group_range 
#____________________________________________________________________________________________________________________________________
        
   
    