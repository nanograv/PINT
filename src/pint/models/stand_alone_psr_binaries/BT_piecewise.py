import astropy.constants as c
import astropy.units as u
import numpy as np
import pint.toa
from pint import GMsun, Tsun, ls
from pint.models.stand_alone_psr_binaries.BT_model import BTmodel
from .binary_generic import PSR_BINARY


class BTpiecewise(BTmodel):
    """
     This is a class independent from the PINT platform for pulsar BT piecewise binary model. It is a subclass of BTmodel which is a subclass of  PSR_BINARY class defined in file binary_generic.py in the same directory. This class is designed for use with the PINT platform but can be used as an independent module for piecewise binary delay calculation. To interact with the PINT platform, a pulsar_binary wrapper is needed. See the source file pint/models/binary_piecewise.py.
    Reference
    ---------
    The 'BT' binary model for the pulse period. Model as in:
    W.M. Smart, (1962), "Spherical Astronomy", p35
    Blandford & Teukolsky (1976), ApJ, 205, 580-591
    Return
    ---------
    A piecewise bt binary model class with parameters, delay calculations and derivatives.

    Example Session
    ---------
    >>import astropy.units as u
    >>import numpy as np

    >>binary_model=BTpiecewise()
    >>param_dict = {'T0': 50000, 'ECC': 0.2}
    >>binary_model.update_input(**param_dict)

    >>t=np.linspace(50001.,60000.,10)*u.d

    Adding binary parameters and piece ranges
    >>binary_model.add_binary_params('T0X_0000', 60000*u.d)
    >>binary_model.add_binary_params('XR1_0000', 50000*u.d)
    >>binary_model.add_binary_params('XR2_0000', 55000*u.d)

    Can add more pieces here...

    Overide default values values if desired
    >>updates = {'T0X_0000':60000.*u.d,'XR1_0000':50000.*u.d,'XR2_0000': 55000*u.d}

    update the model with the piecewise parameter value(s) and piece ranges
    >>binary_model.update_input(**updates)

    Using pint's get_model and loading this as a timing model and following the method described in ../binary_piecewise.py
    sets  _t multiple times during pint's residual calculation
    for simplicity we're just going to set _t directly though this is not recommended.
    >>setattr(binary_model,'_t' ,t)

    #here we call get_tt0 to get the "loaded toas" to interact with the pieces passed to the model earlier
    #sets the attribute "T0X_per_toa" and/or "A1X_per_toa", contains the piecewise parameter value that will be referenced
    #for each toa future calculations
    >>binary_model.get_tt0(t)
    #For a piecewise T0, tt0 becomes a piecewise quantity, otherwise it is how it functions in BT_model.py.

    #get_tt0 sets the attribute "T0X_per_toa" and/or "A1X_per_toa".
    #contains the piecewise parameter value that will be referenced for each toa future calculations
    >>binary_model.T0X_per_toa

    Information about any group can be found with the following:
    >>binary_model.piecewise_parameter_information
    Order: [[Group index, Piecewise T0, Piecewise A1, Piece lower bound, Piece upper bound]]

    Making sure a binary_model.tt0 exists
    >>binary_model._tt0 = binary_model.get_tt0(binary_model._t)

    Obtain piecewise BTdelay()
    >>binary_model.BTdelay()
    """

    def __init__(self, axis_store_initial=None, t=None, input_params=None):
        self.binary_name = "BT_piecewise"
        super(BTpiecewise, self).__init__()
        if t is None:
            self._t = None
        self.axis_store_initial = []
        self.extended_group_range = []
        self.param_pieces = []
        self.d_binarydelay_d_par_funcs = [self.d_BTdelay_d_par]
        if t is not None:
            self._t = t
        if input_params is not None:
            if self.T0X is None:
                self.update_input(input_params)
        self.binary_params = list(self.param_default_value.keys())

    def set_param_values(self, valDict=None):
        super().set_param_values(valDict=valDict)
        self.setup_internal_structures(valDict=valDict)

    def setup_internal_structures(self, valDict=None):
        # initialise arrays to store T0X/A1X values per toa
        self.T0X_arr = []
        self.A1X_arr = []
        # initialise arrays to store piecewise group boundaries
        self.lower_group_edge = []
        self.upper_group_edge = []
        # initialise array that will be 5 x n in shape. Where n is the number of pieces required by the model
        piecewise_parameter_information = []
        # If there are no updates passed by binary_instance, sets default value (usually overwritten when reading from parfile)

        if valDict is None:
            self.T0X_arr = [self.T0]
            self.A1X_arr = [self.A1]
            self.lower_group_edge = [0]
            self.upper_group_edge = [1e9]
            self.piecewise_parameter_information = [
                0,
                self.T0,
                self.A1,
                0 * u.d,
                1e9 * u.d,
            ]
        else:
            # initialise array used to count the number of pieces. Operates by seaching for "A1X_i/T0X_i" and appending i to the array.
            piece_index = []
            # Searches through updates for keys prefixes matching T0X/A1X, can be allowed to be more flexible with param+"X_" provided param is defined earlier.
            for key, value in valDict.items():
                if (
                    key[0:4] == "T0X_"
                    or key[0:4] == "A1X_"
                    and not (key[4:8] in piece_index)
                ):
                    # appends index to array
                    piece_index.append((key[4:8]))
                    # makes sure only one instance of each index is present returns order indeces
            piece_index = np.unique(piece_index)
            # looping through each index in order they are given (0 -> n)
            for index in piece_index:
                # array to store specific piece i's information in the order [index,T0X,A1X,Group's lower edge, Group's upper edge,]
                param_pieces = []
                piece_number = f"{int(index):04d}"
                param_pieces.append(piece_number)
                string = [
                    "T0X_" + index,
                    "A1X_" + index,
                    "XR1_" + index,
                    "XR2_" + index,
                ]

                # if string[0] not in param_pieces:
                for i in range(0, len(string)):
                    if string[i] in valDict:
                        param_pieces.append(valDict[string[i]])
                    elif string[i] not in valDict:
                        attr = string[i][0:2]

                        if hasattr(self, attr):
                            param_pieces.append(getattr(self, attr))
                        else:
                            raise AttributeError(
                                "Malformed valDict being used, attempting to set an attribute that doesn't exist. Likely a corner case slipping through validate() in binary_piecewise."
                            )
                        # Raises error if range not defined as there is no Piece upper/lower bound in the model.

                piecewise_parameter_information.append(param_pieces)

            self.valDict = valDict
            # sorts the array chronologically by lower edge of each group,correctly works for unordered pieces

            self.piecewise_parameter_information = sorted(
                piecewise_parameter_information, key=lambda x: x[3]
            )

            # Uses the index for each toa array to create arrays where elements are the A1X/T0X to use with that toa
            if len(self.piecewise_parameter_information) > 0:
                if self._t is not None:
                    self.group_index_array = self.toa_belongs_in_group(self._t)

                    (
                        self.T0X_per_toa,
                        self.A1X_per_toa,
                    ) = self.piecewise_parameter_from_information_array(self._t)

    def piecewise_parameter_from_information_array(self, t):
        """Creates a list of piecewise orbital parameters to use in calculations. It is the same dimensions as the TOAs loaded in. Each entry is the piecewise parameter value from the group it belongs to.
        ----------
        t : Quantity. TOA, not necesserily barycentered
        Returns
        -------
        list
                Quantity (length: t). T0X parameter to use for each TOA in calculations.
                Quantity (length: t). A1X parameter to use for each TOA in calculations.
        """
        A1X_per_toa = []
        T0X_per_toa = []
        if not hasattr(self, "group_index_array"):
            self.group_index_array = self.toa_belongs_in_group(t)
        if len(self.group_index_array) != len(t):
            self.group_index_array = self.toa_belongs_in_group(t)
            # searches the 5 x n array to find the index matching the toa_index
        possible_groups = [item[0] for item in self.piecewise_parameter_information]
        if len(self.group_index_array) > 1 and len(t) > 1:
            for i in self.group_index_array:
                if i != -1:
                    for k, j in enumerate(possible_groups):
                        if str(i) == j:
                            group_index = k
                            T0X_per_toa.append(
                                self.piecewise_parameter_information[group_index][
                                    1
                                ].value
                            )

                            A1X_per_toa.append(
                                self.piecewise_parameter_information[group_index][
                                    2
                                ].value
                            )

                # if a toa lies between 2 groups, use default T0/A1 values (i.e. toa lies after previous upper bound but before next lower bound)
                else:
                    T0X_per_toa.append(self.T0.value)
                    A1X_per_toa.append(self.A1.value)

        else:
            T0X_per_toa = self.T0.value
            A1X_per_toa = self.A1.value

        T0X_per_toa = T0X_per_toa * u.d
        A1X_per_toa = A1X_per_toa * ls

        return [T0X_per_toa, A1X_per_toa]

    def toa_belongs_in_group(self, toas):
        """Get the piece a TOA belongs to by finding which checking upper/lower edges of each piece.
        ----------
        toas : Astropy.quantity.Quantity.
        Returns
        -------
        list
                int (length: t). Group numbers
        """
        group_no = []
        gb = self.get_group_boundaries()

        lower_edge = []
        upper_edge = []
        for i in range(len(gb[0])):
            lower_edge.append(gb[0][i].value)
            upper_edge.append(gb[1][i].value)

        # lower_edge, upper_edge = [self.get_group_boundaries()[:].value],[self.get_group_boundaries()[1].value]
        for i in toas.value:
            lower_bound = np.searchsorted(np.array(lower_edge), i) - 1
            upper_bound = np.searchsorted(np.array(upper_edge), i)
            if lower_bound == upper_bound:
                index_no = lower_bound
            else:
                index_no = -1
            if index_no != -1:
                group_no.append(self.piecewise_parameter_information[index_no][0])
            else:
                group_no.append(index_no)
        return group_no

    def get_group_boundaries(self):
        """Get the piecewise group boundaries from the dictionary of piecewise parameter information.
        Returns
        -------
        list
                list (length: number of pieces). Contains all pieces' lower edge.
                list (length: number of pieces). Contains all pieces' upper edge.
        """
        lower_group_edge = []
        upper_group_edge = []
        if hasattr(self, "piecewise_parameter_information"):
            for i in range(0, len(self.piecewise_parameter_information)):
                lower_group_edge.append(self.piecewise_parameter_information[i][3])
                upper_group_edge.append(self.piecewise_parameter_information[i][4])
            return [lower_group_edge, upper_group_edge]

    def a1(self):
        if len(self.piecewise_parameter_information) > 0:
            # defines index for each toa as an array of length = len(self._t)
            # Uses the index for each toa array to create arrays where elements are the A1X/T0X to use with that toa
            self.A1X_per_toa = self.piecewise_parameter_from_information_array(self.t)[
                1
            ]

        if hasattr(self, "A1X_per_toa"):
            ret = self.A1X_per_toa + self.tt0 * self.A1DOT
        else:
            ret = self.A1 + self.tt0 * self.A1DOT
        return ret

    def get_tt0(self, barycentricTOA):
        """Finds (barycentricTOA - T0_x). Where T0_x is the piecewise T0 value, if it exists, correponding to the group the TOA belongs to. If T0_x does not exist, use the global T0 vlaue.
        ----------
        Returns
        -------
        astropy.quantity.Quantity
                time since T0
        """
        if barycentricTOA is None or self.T0 is None:
            return None
        if len(barycentricTOA) > 1:
            # defines index for each toa as an array of length = len(self._t)
            # Uses the index for each toa array to create arrays where elements are the A1X/T0X to use with that toa
            self.T0X_per_toa = self.piecewise_parameter_from_information_array(
                barycentricTOA
            )[0]
            T0 = self.T0X_per_toa
        else:
            T0 = self.T0
        if not hasattr(barycentricTOA, "unit") or barycentricTOA.unit == None:
            barycentricTOA = barycentricTOA * u.day
        tt0 = (barycentricTOA - T0).to("second")
        return tt0

    def d_delayL1_d_par(self, par):
        if par not in self.binary_params:
            raise ValueError(f"{par} is not in binary parameter list.")
        par_obj = getattr(self, par)
        index, par_temp = self.in_piece(par)
        if par_temp is None:
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
            raise ValueError(f"{par} is not in binary parameter list.")
        par_obj = getattr(self, par)
        index, par_temp = self.in_piece(par)
        if par_temp is None:
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

    def in_piece(self, par):
        """Finds which TOAs reference which piecewise binary parameter group using the group_index_array property.
        ----------
        par : str
                Name of piecewise parameter e.g. 'T0X_0001' or 'A1X_0001'
        Returns
        -------
        list
                boolean list (length: self._t). True where TOA references a given group, False otherwise.
                binary piecewise parameter label str. e.g. 'T0X' or 'A1X'.
        """
        if "_" in par:
            text = par.split("_")
            param = text[0]
            toa_index = f"{int(text[1]):04d}"
        else:
            param = par
        if hasattr(self, "group_index_array"):
            # group_index_array should exist before fitting, constructing the model/residuals should add this(?)
            group_indexes = np.array(self.group_index_array)
            if param == "T0X":
                ret = group_indexes == toa_index
                return [ret, "T0X"]
            elif param == "A1X":
                ret = group_indexes == toa_index
                return [ret, "A1X"]
            # The toa_index = -1 corresponds to TOAs that don't reference any groups
            else:
                ret = group_indexes == -1
                return [ret, None]
        #'None' corresponds to a parameter without a piecewise counterpart, so will effect all TOAs
        else:
            return [np.zeros(len(self._t)) + 1, None]

    def d_BTdelay_d_par(self, par):
        return self.delayR() * (self.d_delayL2_d_par(par) + self.d_delayL1_d_par(par))

    def d_delayL1_d_A1X(self):
        return np.sin(self.omega()) * (np.cos(self.E()) - self.ecc()) / c.c

    def d_delayL2_d_A1X(self):
        return (
            np.cos(self.omega()) * np.sqrt(1 - self.ecc() ** 2) * np.sin(self.E()) / c.c
        )

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
        # derivative to itself
        if x == y:
            return np.longdouble(np.ones(len(self.tt0))) * u.Unit("")
        # Get the unit right
        yAttr = getattr(self, y)
        xAttr = getattr(self, x)
        U = [None, None]
        for i, attr in enumerate([yAttr, xAttr]):
            # If attr is a PINT Parameter class type
            if hasattr(attr, "units"):
                U[i] = attr.units
            # If attr is a Quantity type
            elif hasattr(attr, "unit"):
                U[i] = attr.unit
            # If attr is a method
            elif hasattr(attr, "__call__"):
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
