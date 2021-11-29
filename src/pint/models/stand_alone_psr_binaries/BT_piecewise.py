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
        self.d_binarydelay_d_par_funcs = [self.d_BTdelay_d_par]
        if t is not None:
            self._t = t
        if input_params is not None:
            if self.T0X is None:
                self.update_input(input_params)
            elif self.T0X is not None:
                self.update_input()
        self.binary_params = list(self.param_default_value.keys())
        #self.param_aliases.update({"T0X": ["T0X"], "A1X": ["A1X"]})
        #print("goes via BT_piecewise")
        
    def set_param_values(self, valDict=None):
        super().set_param_values(valDict=valDict)
        self.piecewise_parameter_loader(valDict=valDict)
    
    def piecewise_parameter_loader(self, valDict=None):
        #print(f"Contents of valDict: {valDict}")
        self.T0X_arr = []   #initialise arrays to store T0X/A1X values per toa 
        self.A1X_arr = []
        self.lower_group_edge = []    #initialise arrays to store piecewise group boundaries
        self.upper_group_edge = []
        piecewise_parameter_information = []   #initialise array that will be 5 x n in shape. Where n is the number of pieces required by the model
        #print(f"valDict:{valDict}")
        if valDict is None:   #If there are no updates passed by binary_instance, sets default value (usually overwritten when reading from parfile)
            self.T0X_arr = self.T0   
            self.A1X_arr = self.A1
            self.lower_group_edge=[0]
            self.upper_group_edge=[1e9]
            self.piecewise_parameter_information = [0,self.T0,self.A1,0*u.d,1e9*u.d]
        else:
            piece_index = []   #iniialise array used to count the number of pieces. Operates by seaching for "A1X_i/T0X_i" and appending i to the array. Can operate if pieces are given out of order. 
            
            for key, value in valDict.items():   #Searches through updates for keys prefixes matching T0X/A1X, can be allowed to be more flexible with param+"X_" provided param is defined earlier. Arbitrary piecewise parameter model
                if key[0:4]=="T0X_" or key[0:4] == "A1X_":
                    piece_index.append((key[4:8]))   #appends index to array
            piece_index= np.unique(piece_index)   #makes sure only one instance of each index is present returns order indeces
            for index in piece_index:   #looping through each index in order they are given (0 -> n)
                param_pieces = []    #array to store specific piece i's information in the order [index,T0X,A1X,Group's lower edge, Group's upper edge,]
                piece_number = int(index)
                param_pieces.append(piece_number)
                string = ["T0X_"+index,"A1X_"+index,"PieceLowerBound_"+index,"PieceUpperBound_"+index]            
                for i in range(0,len(string)):
                    if string[i] in valDict:
                        param_pieces.append(valDict[string[i]])
                    elif string[i] not in valDict:
                        attr = string[i][0:2]
                        param_pieces.append(getattr(self, attr))
                        #Raises error if range not defined as there is no Piece upper/lower bound in the model.
                #print(param_pieces)
                piecewise_parameter_information.append(param_pieces)
            self.valDict=valDict
            self.piecewise_parameter_information = sorted(piecewise_parameter_information,key=lambda x: x[3])  #sorts the array chronologically by lower edge of each group,correctly works for unordered pieces (i.e. index 0 can correspond to an arbitrary group of data at any time)
            #print(f"Filled information array: {self.piecewise_parameter_information}")
            if len(self.piecewise_parameter_information)>0:
                #check = hasattr(self,"t")
                #print(f"Piecewise parameter loader can see t: {check}")
                if hasattr(self,"_t") is True:
                    if (self._t) is not None:
                        #self.print_toas_in_group()   #Defines object's index for each toa as an array of length = len(self._t)
                        self.group_index_array = self.toa_belongs_in_group(self._t)
                        self.T0X_per_toa, self.A1X_per_toa = self.piecewise_parameter_from_information_array(self._t)   #Uses the index for each toa array to create arrays where elements are the A1X/T0X to use with that toa
                        
                    #print(self.T0X_per_toa)

            
            
    def piecewise_parameter_from_information_array(self,t):
        A1X_per_toa = []  
        T0X_per_toa = []
        if hasattr(self,"group_index_array") is False:
            self.group_index_array = self.toa_belongs_in_group(t) 
        if len(self.group_index_array) != len(t):
            self.group_index_array = self.toa_belongs_in_group(t)
        for i in self.group_index_array:
            if i != -1:
                for j in range(len(self.piecewise_parameter_information)):
                    if self.piecewise_parameter_information[j][0] == i: #searches the 5 x n array to find the index matching the toa_index
                        T0X_per_toa.append(self.piecewise_parameter_information[j][1].value)
                        A1X_per_toa.append(self.piecewise_parameter_information[j][2].value)
            else: #if a toa lies between 2 groups, use default T0/A1 values (i.e. toa lies after previous upper bound but before next lower bound)
                T0X_per_toa.append(self.T0.value)
                A1X_per_toa.append(self.A1.value)
        T0X_per_toa = T0X_per_toa * u.d
        A1X_per_toa = A1X_per_toa * ls
        return [T0X_per_toa,A1X_per_toa]
    
    def group_index(self):
        index = []
        for i in range(len(self._t)):
            index1 = self.lower_index[i]
            index2 = self.upper_index[i]
            if (index1==index2):
                index.append(index1)
            else:
                index.append(-1)
        self.group_index_array = np.array(index)
        return self.group_index_array
            
    #def print_toas_in_group(self):   #takes array sorted by lower group edge (requires groups to be chronologically ordered). Called from piece_parameter_loader, ordering operation occurs there
    #    lower_bound = []   #seperates lower/upper bounds from 5 x n array of piece information
    #    upper_bound = []
    #    lower_index_temp = []
    #    upper_index_temp = []
    #    for i in range(0,len(self.piecewise_parameter_information)):   #loops through the array (len(...) = n)
    #        if i == 0:  #for the first group, makes the lower bound slightly earlier than defined such that ambiguity of where first toa is, is accomodated
    #            if len(self.piecewise_parameter_information)==1:
    #                lower_bound.append(self.piecewise_parameter_information[i][3].value-1) #modified bounds for singular group
    #                upper_bound.append(self.piecewise_parameter_information[i][4].value+1)  
    #            else:
    #                lower_bound.append(self.piecewise_parameter_information[i][3].value-1) #modified sorted lower bound to encompass the first toa
    #                upper_bound.append(self.piecewise_parameter_information[i][4].value) 
    #        elif i==len(self.piecewise_parameter_information)-1:  #for the last group, makes the upper bound slightly later than defined such that ambiguity of where last toa is, is accomodated
    #            lower_bound.append(self.piecewise_parameter_information[i][3].value)
    #            upper_bound.append(self.piecewise_parameter_information[i][4].value+1) #modified sorted upper bound to encompass the last toa
    #        else:
    #            lower_bound.append(self.piecewise_parameter_information[i][3].value) #append all other lower/upper bounds 
    #            upper_bound.append(self.piecewise_parameter_information[i][4].value)
    #    if hasattr(self._t, "value") is True:
    #        lower_index = np.searchsorted(lower_bound,self._t.value)-1  #Assigns group index to each toa. toa will always be on the right/left of the lower/upper bound, hence the "-1" factor
    #        upper_index = np.searchsorted(upper_bound,self._t.value) #For toas between groups i.e lower bound:(55000,55100), upper bound: (55050,55150) lower/upperindex of 55075 should be (0,1)
    #    else:
    #        lower_index = np.searchsorted(lower_bound,self._t)-1  #Assigns group index to each toa. toa will always be on the right/left of the lower/upper bound, hence the "-1" factor
    #        upper_index = np.searchsorted(upper_bound,self._t) #For toas between groups i.e lower bound:(55000,55100), upper bound: (55050,55150) lower/upperindex of 55075 should be: 0
    #    for i in lower_index: #this loop is to accomodate out of order groups
    #        lower_index_temp.append(self.piecewise_parameter_information[i][0])
    #    for i in upper_index:
    #        if i > len(upper_bound)-1:
    #            upper_index_temp.append(999)
    #        else:
    #            upper_index_temp.append(self.piecewise_parameter_information[i][0])
    #    self.lower_index = np.array(lower_index_temp)
    #    self.upper_index = np.array(upper_index_temp)
    
    
    def toa_belongs_in_group(self,t):
        group_no = []
        lower_edge, upper_edge = self.get_group_boundaries()
        for i in t:
            lower_bound = np.searchsorted(lower_edge,i)-1
            upper_bound = np.searchsorted(upper_edge,i)
            if lower_bound == upper_bound:
                index_no = (lower_bound)
            else:
                index_no = (-1)
                
            if index_no !=-1:
                group_no.append(self.piecewise_parameter_information[index_no][0])
            else:
                group_no.append(index_no)
        return group_no
        
    def get_group_boundaries(self):
        lower_group_edge = []
        upper_group_edge = []
        #print("from get_group_boundaries")
        if hasattr(self,"piecewise_parameter_information"):
            for i in range(0, len(self.piecewise_parameter_information)):
                lower_group_edge.append(self.piecewise_parameter_information[i][3])
                upper_group_edge.append(self.piecewise_parameter_information[i][4])
            return [lower_group_edge, upper_group_edge]
    
    def a1(self):
        self.A1_val = self.A1X_arr*ls
        if hasattr(self, "A1X_per_toa") is True:
            ret = self.A1X_per_toa + self.tt0 * self.A1DOT
        else:
            ret = self.A1 + self.tt0 * self.A1DOT
        return ret
    
    
    def get_tt0(self, barycentricTOA):
        """ tt0 = barycentricTOA - T0 """
        if barycentricTOA is None or self.T0 is None:
            tt0 = None
            return tt0
        if len(barycentricTOA)>1:
            if len(self.piecewise_parameter_information)>0:
                self.toa_belongs_in_group(barycentricTOA)   #Defines object's index for each toa as an array of length = len(self._t)
                self.T0X_per_toa,self.A1X_per_toa = self.piecewise_parameter_from_information_array(self._t)   #Uses the index for each toa array to create arrays where elements are the A1X/T0X to use with that toa
        if len(barycentricTOA)>1:
            check = hasattr(self, "T0X_per_toa")
            if hasattr(self, "T0X_per_toa") is True:
                if len(self.T0X_per_toa)==1:
                    T0 = self.T0X_per_toa
                    #print("hello from 1")
                else:
                    T0 = self.T0X_per_toa
                    #print("hello from 2")
            else:
                T0 = self.T0
                #print("hello from 3")
        else:
            T0 = self.T0
            #print("hello from 4")
        if not hasattr(barycentricTOA, "unit") or barycentricTOA.unit == None:
            barycentricTOA = barycentricTOA * u.day
        #print(f"Unique T0s being used in tt0 calculation: {np.unique(T0)}\n")
        tt0 = (barycentricTOA - T0).to("second")
        return tt0
   
    
    def d_delayL1_d_par(self, par):
        if par not in self.binary_params:
            errorMesg = par + " is not in binary parameter list."
            raise ValueError(errorMesg)
        par_obj = getattr(self, par)
        index,par_temp = self.in_piece(par)
        #print(index)
        if par_temp is None: # to circumvent the need for list of d_pars = [T0X_0,...,T0X_i] use par_temp
            if hasattr(self, "d_delayL1_d_" + par):
                func = getattr(self, "d_delayL1_d_" + par)
                return func() * index
            else:
                if par in self.orbits_cls.orbit_params:
                    return self.d_delayL1_d_E() * self.d_E_d_par(par)
                else:
                    return np.zeros(len(self.t)) * u.second / par_obj.unit            
        else:
            if hasattr(self, "d_delayL1_d_" + par_temp):
                func = getattr(self, "d_delayL1_d_" + par_temp)
                return func() * index
            else:
                if par in self.orbits_cls.orbit_params:
                    return self.d_delayL1_d_E() * self.d_E_d_par()
                else:
                    return np.zeros(len(self.t)) * u.second / par_obj.unit            
            
    

    def d_delayL2_d_par(self, par):
        if par not in self.binary_params:
            errorMesg = par + " is not in binary parameter list."
            raise ValueError(errorMesg)
        #print(par)
        par_obj = getattr(self, par)
        index,par_temp = self.in_piece(par)
        #print(index)
        if par_temp is None: # to circumvent the need for list of d_pars = [T0X_0,...,T0X_i] use par_temp
            if hasattr(self, "d_delayL2_d_" + par):
                func = getattr(self, "d_delayL2_d_" + par)
                return func() * index
            else:
                if par in self.orbits_cls.orbit_params:
                    return self.d_delayL2_d_E() * self.d_E_d_par(par)
                else:
                    return np.zeros(len(self.t)) * u.second / par_obj.unit            
        else:
            if hasattr(self, "d_delayL2_d_" + par_temp):
                func = getattr(self, "d_delayL2_d_" + par_temp)
                return func() * index
            else:
                if par in self.orbits_cls.orbit_params:
                    return self.d_delayL2_d_E() * self.d_E_d_par()
                else:
                    return np.zeros(len(self.t)) * u.second / par_obj.unit            
    
    def in_piece(self,par):
        if "_" in par:
            text = par.split("_")
            param = text[0]
            toa_index = int(text[1])
        else:
            param = par
        if hasattr(self, "group_index_array"):
            group_indexes = np.array(self.group_index_array)
            if param == "T0X":
                ret = (group_indexes == toa_index)
                return [ret,"T0X"]
            elif param == "A1X":
                ret = (group_indexes == toa_index)
                return [ret,"A1X"]
            else:
                ret = (group_indexes == -1)
                return [ret,None]
        else:
            return [np.zeros(len(self._t))+1,None]
    
    
    def d_BTdelay_d_par(self, par):
        return self.delayR() * (self.d_delayL2_d_par(par) + self.d_delayL1_d_par(par))
    
    def d_delayL1_d_A1X(self):
        return np.sin(self.omega()) * (np.cos(self.E()) - self.ecc()) / c.c
    
    def d_delayL2_d_A1X(self):
        return (np.cos(self.omega()) * np.sqrt(1 - self.ecc() ** 2) * np.sin(self.E()) / c.c)
    
    def d_delayL1_d_T0X(self):
        return self.d_delayL1_d_E() * self.d_E_d_T0X()

    
    def d_delayL2_d_T0X(self):
        return self.d_delayL2_d_E() * self.d_E_d_T0X()
    
    def d_E_d_T0X(self):
        """Analytic derivative

        d(E-e*sinE)/dT0 = dM/dT0
        dE/dT0(1-cosE*e)-de/dT0*sinE = dM/dT0
        dE/dT0(1-cosE*e)+eDot*sinE = dM/dT0
        """
        RHS = self.prtl_der("M", "T0")
        E = self.E()
        EDOT = self.EDOT
        ecc = self.ecc()
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            return (RHS - EDOT * np.sin(E)) / (1.0 - np.cos(E) * ecc)
        
    
    def prtl_der(self, y, x):
        """Find the partial derivatives in binary model pdy/pdx

        Parameters
        ----------
        y : str
           Name of variable to be differentiated
        x : str
           Name of variable the derivative respect to

        Returns
        -------
        np.array
           The derivatives pdy/pdx
        """
        if y not in self.binary_params + self.inter_vars:
            errorMesg = y + " is not in binary parameter and variables list."
            raise ValueError(errorMesg)

        if x not in self.inter_vars + self.binary_params:
            errorMesg = x + " is not in binary parameters and variables list."
            raise ValueError(errorMesg)
        #derivative to itself
        if x == y:
            return np.longdouble(np.ones(len(self.tt0))) * u.Unit("")
        # Get the unit right
        yAttr = getattr(self, y)
        xAttr = getattr(self, x)
        U = [None, None]
        for i, attr in enumerate([yAttr, xAttr]):
            if hasattr(attr, "units"):  # If attr is a PINT Parameter class type
                U[i] = attr.units
            elif hasattr(attr, "unit"):  # If attr is a Quantity type
                U[i] = attr.unit
            elif hasattr(attr, "__call__"):  # If attr is a method
                U[i] = attr().unit
            else:
                raise TypeError(type(attr) + "can not get unit")
        yU = U[0]
        xU = U[1]
        # Call derivtive functions
        derU = yU / xU
        if hasattr(self, "d_" + y + "_d_" + x):
            dername = "d_" + y + "_d_" + x
            result = getattr(self, dername)()

        elif hasattr(self, "d_" + y + "_d_par"):
            dername = "d_" + y + "_d_par"
            result = getattr(self, dername)(x)
            
        else:
            result = np.longdouble(np.zeros(len(self.tt0)))

        if hasattr(result, "unit"):
            return result.to(derU, equivalencies=u.dimensionless_angles())
        else:
            return result * derU

        
        
    def d_M_d_par(self, par):
        """derivative for M respect to bianry parameter.

        Parameters
        ----------
        par : string
             parameter name

        Returns
        -------
        Derivitve of M respect to par
        """
        
        if par not in self.binary_params:
            errorMesg = par + " is not in binary parameter list."
            raise ValueError(errorMesg)
        par_obj = getattr(self, par)
        result = self.orbits_cls.d_orbits_d_par(par)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            result = result.to(u.Unit("") / par_obj.unit)
        return result   