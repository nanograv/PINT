"""The BT (Blandford & Teukolsky) model."""
import numpy as np
from pint.models.parameter import floatParameter
from pint.models.pulsar_binary import PulsarBinary
from pint.models.stand_alone_psr_binaries.BT_model import BTmodel
from pint.models.stand_alone_psr_binaries.BT_piecewise import BTpiecewise
from pint.models.timing_model import MissingParameter, TimingModel
import astropy.units as u
from pint import GMsun, Tsun, ls
from astropy.table import Table
from astropy.time import Time
from pint.models.parameter import (
    MJDParameter,
    floatParameter,
    prefixParameter,
    maskParameter,
)

from pint.toa_select import TOASelect





class BinaryBT(PulsarBinary):
    """Blandford and Teukolsky binary model.

    This binary model is described in Blandford and Teukolshy 1976. It is
    a relatively simple parametrized post-Keplerian model that does not
    support Shapiro delay calculations.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.BT_model.BTmodel`.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_bt.BinaryBT

    Notes
    -----
    Because PINT's binary models all support specification of multiple orbital
    frequency derivatives FBn, this is capable of behaving like the model called
    BTX in tempo2. The model called BTX in tempo instead supports multiple
    (non-interacting) companions, and that is not supported here. Neither can
    PINT accept "BTX" as an alias for this model.

    See Blandford & Teukolsky 1976, ApJ, 205, 580.
    """

    register = True

    def __init__(self):
        super().__init__()
        self.binary_model_name = "BT"
        self.binary_model_class = BTmodel

        self.add_param(
            floatParameter(
                name="GAMMA",
                value=0.0,
                units="second",
                description="Time dilation & gravitational redshift",
            )
        )
        self.remove_param("M2")
        self.remove_param("SINI")

    def validate(self):
        """Validate BT model parameters"""
        super().validate()
        for p in ("T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p, f"{p} is required for BT")

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).value = "0"
                getattr(self, p).frozen = True

        if self.GAMMA.value is None:
            self.GAMMA.value = "0"
            self.GAMMA.frozen = True


"""The BT (Blandford & Teukolsky) model with piecewise orbital parameters.
See Blandford & Teukolsky 1976, ApJ, 205, 580.
"""

class BinaryBTPiecewise(PulsarBinary):
    """Model implementing the BT model with piecewise orbital parameters A1X and T0X. This model lets the user specify time ranges and fit for a different piecewise orbital parameter in each time range,
    This is a PINT pulsar binary BTPiecewise model class, a subclass of PulsarBinary.
    It is a wrapper for stand alone BTPiecewise class defined in 
    ./stand_alone_psr_binary/BT_piecewise.py
    The aim for this class is to connect the stand alone binary model with the PINT platform.
    BTpiecewise special parameters, where xxxx denotes the 4-digit index of the piece:
    T0X_xxxx Piecewise T0 values for piece
    A1X_xxxx Piecewise A1 values for piece
    XR1_xxxx Lower time boundary of piece
    XR2_xxxx Upper time boundary of piece 
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
        self.A1_value_funcs = []
        self.T0_value_funcs = []
        self.remove_param("M2")
        self.remove_param("SINI")


    def add_group_range(self, group_start_mjd, group_end_mjd, piece_index = None, ):
        """Add an orbital piecewise parameter group range. If piece_index is not provided a new piece will be added with index equal to the number of pieces plus one. Pieces cannot have the duplicate pieces and cannot have the same index. A pair of consisting of a piecewise A1 and T0 may share an index and will act over the same piece range. 
        Parameters
        ----------
        group_start_mjd : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for the piece lower boundary
        group_end_mjd : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for the piece upper boundary
        piece_index : int
                Number to label the piece being added.
        """
        if group_start_mjd is not None and group_end_mjd is not None:
            if isinstance(group_start_mjd,Time):
                group_start_mjd= group_start_mjd.mjd
            elif isinstance(group_start_mjd,u.quantity.Quantity):
                group_start_mjd= group_start_mjd.value
            if isinstance(group_end_mjd,Time):
                group_end_mjd = group_end_mjd.mjd
            elif isinstance(group_end_mjd,u.quantity.Quantity):
                group_end_mjd = group_end_mjd.value
        if piece_index is None:
            dct = self.get_prefix_mapping_component("XR1_")
            if len(list(dct.keys()))>0:
            	piece_index = np.max(list(dct.keys())) + 1
            else:
                piece_index = 0


        # check the validity of the desired group to add


        if group_end_mjd is not None and group_start_mjd is not None:
            if group_end_mjd <= group_start_mjd:
                raise ValueError("Starting MJD is greater than ending MJD.")
            elif piece_index < 0:
                raise ValueError(
                    f"Invalid index for group: {piece_index} should be greater than or equal to 0"
                )
            elif piece_index > 9999:
                raise ValueError(
                    f"Invalid index for group. Cannot index beyond 9999 (yet?)"
                )
                
                
        i = f"{int(piece_index):04d}"
        self.add_param(
            prefixParameter(
                name="XR1_{0}".format(i),
                units="MJD",
                description="Beginning of paramX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=group_start_mjd,
            )
        )
        self.add_param(
            prefixParameter(
                name="XR2_{0}".format(i),
                units="MJD",
                description="End of paramX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=group_end_mjd,
            )
        )
        self.setup()

    def remove_range(self, index):
        """Removes all orbital piecewise parameters associated with a given index/list of indices.
        Parameters
        ----------
        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to T0X/A1X indices to be removed from model.
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
            for prefix in ["T0X_", "A1X_", "XR1_", "XR2_"]:
                self.remove_param(prefix + index_rf)
        self.validate()
        self.setup()

    def add_piecewise_param(self,piece_index,**kwargs ):
        """Add an orbital piecewise parameter.
        Parameters
        ----------
        piece_index : int
                Number to label the piece being added. Expected to match a set of piece boundaries.
        param : str
                Piecewise parameter label e.g. "T0" or "A1".
        paramx : np.float128 or astropy.quantity.Quantity
                Piecewise parameter value.
        """
        for key in ('T0','A1'):
            if key in kwargs:
                param = key
                paramx = kwargs[key]
                if isinstance(paramx,u.quantity.Quantity):
                    paramx = paramx.value
                elif isinstance(paramx,np.float128):
                    paramx = paramx
                else:
                    raise ValueError(
                "Unspported data type '%s'. Ensure the piecewise parameter value is a np.float128 or astropy.quantity.Quantity" % type(paramx)
                )
                if key == 'T0':
                    param_unit = u.d
                elif key =='A1':
                    param_unit = ls
                key_found = True
        
        if not key_found:
            raise AttributeError(
            "No piecewise parameters passed. Use T0 = / A1 = to declare a piecewise variable."
        )
                
        if piece_index is None:
            dct = self.get_prefix_mapping_component(param + "X_")
            if len(list(dct.keys()))>0:
            	piece_index = np.max(list(dct.keys())) + 1
            else:
                piece_index = 0
        elif int(piece_index) in self.get_prefix_mapping_component(param +"X_"):
            raise ValueError(
                "Index '%s' is already in use in this model. Please choose another." % piece_index
            )
        i = f"{int(piece_index):04d}"

        # handling if None are passed as arguments
        if any(i is None for i in [param, param_unit, paramx]):
            if param is not None:
                # if parameter value or unit unset, set with default according to param
                if param_unit is None:
                    param_unit = (getattr(self, param)).units
                if paramx is None:
                    paramx = (getattr(self, param)).value
        # check if name exits and is currently available

        self.add_param(
	    prefixParameter(
	        name=param + f"X_{i}",
		units=param_unit,
		value=paramx,
		description="Parameter" + param + "variation",
		parameter_type="float",
		frozen=False,
    	    )
    	)
        self.setup()

    
    def lock_groups(self):
        self.validate()
        self.update_binary_object(None)
        self.setup()

    def setup(self):
        """Raises
        ------
        ValueError
                if there are values that have been added without name/ranges associated (should only be raised if add_piecewise_param has been side-stepped with an alternate method)
        """
        super().setup()
        for bpar in self.params:
            self.register_deriv_funcs(self.d_binary_delay_d_xxxx, bpar)
        # Setup the model isinstance
        self.binary_instance = self.binary_model_class()
        # piecewise T0's
        T0X_mapping = self.get_prefix_mapping_component("T0X_")
        T0Xs = {}
        # piecewise A1's (doing piecewise A1's requires more thought and work)
        A1X_mapping = self.get_prefix_mapping_component("A1X_")
        A1Xs = {}
        # piecewise parameter ranges XR1-piece lower bound
        XR1_mapping = self.get_prefix_mapping_component("XR1_")
        XR1s = {}
        # piecewise parameter ranges XR2-piece upper bound
        XR2_mapping = self.get_prefix_mapping_component("XR2_")
        XR2s = {}

        for index in XR1_mapping.values():
            index = index.split("_")[1]
            piece_index = f"{int(index):04d}" 
            if hasattr(self,f"T0X_{piece_index}"):
               if getattr(self,f"T0X_{piece_index}") is not None:
                   self.binary_instance.add_binary_params(f"T0X_{piece_index}",getattr(self,f"T0X_{piece_index}"))
               else:
                   self.binary_instance.add_binary_params(f"T0X_{piece_index}",self.T0.value)    
                    
                    
            if hasattr(self,f"A1X_{piece_index}"):
               if hasattr(self,f"A1X_{piece_index}"):
                  if getattr(self,f"A1X_{piece_index}") is not None:
                     self.binary_instance.add_binary_params(f"A1X_{piece_index}",getattr(self,f"A1X_{piece_index}"))
                  else:
                     self.binary_instance.add_binary_params(f"A1X_{piece_index}",self.A1.value)

                                                    
            if hasattr(self,f"XR1_{piece_index}"):
                if getattr(self,f"XR1_{piece_index}") is not None:
                   self.binary_instance.add_binary_params(f"XR1_{piece_index}", getattr(self,f"XR1_{piece_index}"))
                else:
                    raise ValueError(
                    f"No date provided to create a group with"
                )
            else:
                raise ValueError(
                    f"No name provided to create a group with"
                )
                
                
                
            if hasattr(self,f"XR2_{piece_index}"):
                if getattr(self,f"XR2_{piece_index}") is not None:
                   self.binary_instance.add_binary_params(f"XR2_{piece_index}", getattr(self,f"XR2_{piece_index}"))
                else:
                    raise ValueError(
                    f"No date provided to create a group with"
                )
            else:
                raise ValueError(
                    f"No name provided to create a group with"
                )

        self.update_binary_object(None)

    
    def validate(self):
        """Include catches for overlapping groups. etc
        Raises
        ------
        ValueError
                if there are pieces with no associated boundaries (T0X_0000 does not have a corresponding XR1_0000/XR2_0000)
        ValueError
                if any boundaries overlap (as it makes TOA assignment to a single group ambiguous). i.e. XR1_0000<XR2_0000 and XR2_0000>XR1_0001
        ValueError
                if the number of lower and upper bounds don't match (should only be raised if XR1 is defined without XR2 and validate is run or vice versa)
        """
        super().validate()
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

        dct_plb = self.get_prefix_mapping_component("XR1_")
        dct_pub = self.get_prefix_mapping_component("XR2_")
        dct_T0X = self.get_prefix_mapping_component("T0X_")
        dct_A1X = self.get_prefix_mapping_component("A1X_")
        if len(dct_plb) > 0 and len(dct_pub) > 0:
            ls_plb = list(dct_plb.items())
            ls_pub = list(dct_pub.items())
            ls_T0X = list(dct_T0X.items())
            ls_A1X = list(dct_A1X.items())

            j_plb = [((tup[1]).split("_"))[1] for tup in ls_plb]
            j_pub = [((tup[1]).split("_"))[1] for tup in ls_pub]
            j_T0X = [((tup[1]).split("_"))[1] for tup in ls_T0X]
            j_A1X = [((tup[1]).split("_"))[1] for tup in ls_A1X]

            if j_plb != j_pub:
                raise ValueError(
                    f"Group boundary mismatch error. Number of detected lower bounds: {j_plb}. Number of detected upper bounds: {j_pub}"
                )
            if len(np.setdiff1d(j_plb, j_pub)) > 0:
                raise ValueError(
                    f"Group index mismatch error. Check the indexes of XR1_/XR2_ parameters in the model"
                )
            if len(ls_pub) > 0 and len(ls_T0X) > 0:
                if len(np.setdiff1d(j_pub, j_T0X)) > 0:
                    raise ValueError(
                        f"Group index mismatch error. Check the indexes of T0X groups, make sure they match there are corresponding group ranges (XR1/XR2)"
                    )
            if len(ls_pub) > 0 and len(ls_A1X) > 0:
                if len(np.setdiff1d(j_pub, j_A1X)) > 0:
                    raise ValueError(
                        f"Group index mismatch error. Check the indexes of A1X groups, make sure they match there are corresponding group ranges (/XR2)"
                    )
            lb = [(getattr(self, tup[1])).value for tup in ls_plb]
            ub = [(getattr(self, tup[1])).value for tup in ls_pub]

            for i in range(len(lb)):
                for j in range(len(lb)):
                    if i != j:
                        if max(lb[i], lb[j]) < min(ub[i], ub[j]):
                            raise ValueError(
                                f"Group boundary overlap detected. Make sure groups are not overlapping"
                            )



    def paramx_per_toa(self,param_name,toas):
        condition = {}
        tbl = toas.table
        XR1_mapping = self.get_prefix_mapping_component("XR1_")
        XR2_mapping = self.get_prefix_mapping_component("XR2_")
        if not hasattr(self,"toas_selector"):
            self.toas_selector=TOASelect(is_range=True)
        if param_name[0:2] == 'T0':
            paramX_mapping = self.get_prefix_mapping_component("T0X_")
            param_unit = u.d
        elif param_name[0:2] == 'A1':
            paramX_mapping = self.get_prefix_mapping_component("A1X_")        
            param_unit = u.ls
        else:
            raise AttributeError(
                "param '%s' not found. Please choose another. Currently implemented: 'T0' or 'A1' "
                % param_name
            )

        for piece_index in paramX_mapping.keys():
            r1 = getattr(self,XR1_mapping[piece_index]).quantity
            r2 = getattr(self,XR2_mapping[piece_index]).quantity
            condition[paramX_mapping[piece_index]] = (r1.mjd,r2.mjd)
        select_idx = self.toas_selector.get_select_index(condition,tbl["mjd_float"])
        paramx = np.zeros(len(tbl))*param_unit
        for k, v in select_idx.items():
            paramx[v]+=getattr(self,k).quantity
        return paramx
        
    
    def get_number_of_groups(self):
        """Get the number of piecewise parameters"""
        return len(self.binary_instance.piecewise_parameter_information)

    #def get_group_boundaries(self):
    #    """Get a all pieces' date boundaries.
    #    Returns
    #    -------
    #    list
    #            np.array
    #                    (length: toas) List of piecewise orbital parameter lower boundaries
    #            np.array
    #                    (length: toas) List of piecewise orbital parameter upper boundaries
    #    """
    #    # asks the object for the number of piecewise groups
    #    return self.binary_instance.get_group_boundaries()

    #def which_group_is_toa_in(self, toa):
    #    """Find the group a toa belongs to based on the boundaries of groups passed to BT_piecewise
    #    Parameters
    #    ----------
    #    toa : toa
    #       TOA/TOAs to check which group they're in
    #    Returns
    #    -------
    #    np.array
    #       str elements, look like ['0000','0001'] for two TOAs where one refences T0X_0000 or T0X_0001.
    #    """
    #    return self.binary_instance.toa_belongs_in_group(toa)

    #def get_group_indexes(self):
    #    """Get all the piecewise parameter labels
    #    Returns
    #    -------
    #    np.array
    #            (length: number of piecewise groups) List of piecewise parameter labels e.g with pieces T0X_0000, T0X_0001, T0X_0003, returns [0,1,3]
    #    """
    #    group_indexes = []
    #    for i in range(0, len(self.binary_instance.piecewise_parameter_information)):
    #        group_indexes.append(
    #            self.binary_instance.piecewise_parameter_information[i][0]
    #        )
    #    return group_indexes

    #def get_T0Xs_associated_with_toas(self, toas):
    #    """Get a of all the piecewise T0s associated with TOAs
    #    Parameters
    #    ----------
    #    toas :
    #            Barycentric TOAs
    #    Returns
    #    -------
    #    np.array
    #            (length: toas) List of piecewise T0X values being used for each TOA
    #    """
    #    if hasattr(self.binary_instance, "group_index_array"):
    #        temporary_storage = self.binary_instance.group_index_array
    #    self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
    #    barycentric_toa = self._parent.get_barycentric_toas(toas)
    #    T0X_per_toa = self.binary_instance.piecewise_parameter_from_information_array(
    #        toas
    #    )[0]
    #    if temporary_storage is not None:
    #        self.binary_instance.group_index_array = temporary_storage
    #    return T0X_per_toa

    #def get_A1Xs_associated_with_toas(self, toas):
    #    """Get a of all the piecewise A1s associated with TOAs
    #    Parameters
    #    ----------
    #    toas :
    #            Barycentric TOAs
    #    Returns
    #    -------
    #    np.array
    #            (length: toas) List of piecewise A1X values being used for each TOA
    #    """
    #    if hasattr(self.binary_instance, "group_index_array"):
    #        temporary_storage = self.binary_instance.group_index_array
    #    self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
    #    barycentric_toa = self._parent.get_barycentric_toas(toas)
    #    A1X_per_toa = self.binary_instance.piecewise_parameter_from_information_array(
    #        toas
    #    )[1]
    #    if temporary_storage is not None:
    #        self.binary_instance.group_index_array = temporary_storage
    #    return A1X_per_toa

    #def does_toa_reference_piecewise_parameter(self, toas, param):
    #    """Query whether a TOA/list of TOAs belong(s) to a specific group
    #    Parameters
    #    ----------
    #    toas :
    #            Barycentric TOAs
    #    param : str
    #            Orbital piecewise parameter alias  e.g. "T0X_0001" or "A1X_0001"
    #    Returns
    #    -------
    #    np.array
    #            boolean array (length: toas). True where toa is within piece boundaries corresponding to param
    #    """
    #    self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
    #    from_in_piece = self.binary_instance.in_piece(param)
    #    return from_in_piece[0]


