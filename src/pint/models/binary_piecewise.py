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
        self.A1_value_funcs = []
        self.T0_value_funcs = []
        self.remove_param("M2")
        self.remove_param("SINI")
        self.T0.value = 1
        self.A1.value = 1
        self.add_group_range(50000, 51000, frozen=False, j=0)
        self.add_piecewise_param("T0", 0)
        self.add_piecewise_param("A1", 0)

    def add_group_range(self, group_start_mjd, group_end_mjd, frozen=True, j=None):
        """Add an orbital piecewise parameter group range.
        Parameters
        ----------
        group_start_mjd : np.float128
                Lower group edge
        group_end_mjd : np.float128
                Upper group edge
        j : int
                Number to label the piece being added.
        """
        # check the validity of the desired group to add
        if group_end_mjd is not None and group_start_mjd is not None:
            if group_end_mjd <= group_start_mjd:
                raise ValueError("Starting MJD is greater than ending MJD.")
            elif j < 0:
                raise ValueError(
                    f"Invalid index for group: {j} should be greater than or equal to 0"
                )
            elif j > 9999:
                raise ValueError(
                    f"Invalid index for group. Cannot index beyond 9999 (yet?)"
                )

        i = f"{int(j):04d}"
        self.add_param(
            prefixParameter(
                name="PLB_{0}".format(i),
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
                name="PUB_{0}".format(i),
                units="MJD",
                unit_template=lambda x: "MJD",
                description="End of paramX interval",
                description_template=lambda x: "End of paramX interval",
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
            for prefix in ["T0X_", "A1X_", "PLB_", "PUB_"]:
                self.remove_param(prefix + index_rf)
        self.validate()
        self.setup()

    def add_piecewise_param(self, param, j, param_unit=None, paramx=None):
        """Add an orbital piecewise parameter.
        Parameters
        ----------
        param : str
                Piecewise parameter label e.g. "T0" or "A1".
        param_unit : astropy.unit
                Piecewise parameter unit.
        paramx : np.float128
                Piecewise parameter value.
        j : int
                Number to label the piece being added. Expected to match a set of piece boundaries.
        """
        if j is None:
            dct = self.get_prefix_mapping_component(param + "X_")
            j = np.max(list(dct.keys())) + 1
        elif int(j) in self.get_prefix_mapping_component("X_"):
            raise ValueError(
                "Index '%s' is already in use in this model. Please choose another." % j
            )
        i = f"{int(j):04d}"

        # handling if None are passed as arguments
        if any(i is None for i in [param, param_unit, paramx]):
            if param is not None:
                # if parameter value or unit unset, set with default according to param
                if param_unit is None:
                    param_unit = (getattr(self, param)).units
                if paramx is None:
                    paramx = (getattr(self, param)).value
        # check if name exits and is currently available
        if param is None or param not in ["T0", "A1"]:
            raise AttributeError(
                "param '%s' not found. Please choose another. Currently implemented: 'T0' or 'A1' "
                % param
            )

        if param == "A1":
            self.add_param(
                prefixParameter(
                    name=param + f"X_{i}",
                    units=param_unit,
                    value=paramx,
                    unit_template=lambda x: param_unit,
                    description="Parameter" + param + "variation",
                    description_template=lambda x: param,
                    parameter_type="float",
                    frozen=False,
                )
            )
        elif param == "T0":
            self.add_param(
                prefixParameter(
                    name=param + f"X_{i}",
                    units=param_unit,
                    value=paramx,
                    unit_template=lambda x: param_unit,
                    description="Parameter" + param + "variation",
                    description_template=lambda x: param,
                    parameter_type="float",
                    frozen=False,
                )
            )

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
        # piecewise parameter ranges PLB-piece lower bound
        XR1_mapping = self.get_prefix_mapping_component("PLB_")
        XR1s = {}
        # piecewise parameter ranges PUB-piece upper bound
        XR2_mapping = self.get_prefix_mapping_component("PUB_")
        XR2s = {}

        for t0n in T0X_mapping.values():
            T0Xs[t0n] = getattr(self, t0n).quantity

        for t0_name, t0_value in T0Xs.items():
            if (t0_value is not None) and (t0_name is not None):
                self.binary_instance.add_binary_params(t0_name, t0_value)
            elif t0_name is not None:
                self.binary_instance.add_binary_params(t0_name, self.T0.value)
            else:
                raise ValueError(f"No name and value to create a group with")

        for a1n in A1X_mapping.values():
            A1Xs[a1n] = getattr(self, a1n).quantity
        for a1_name, a1_value in A1Xs.items():
            if (a1_value is not None) and (a1_name is not None):
                self.binary_instance.add_binary_params(a1_name, a1_value)
            elif t0_name is not None:
                self.binary_instance.add_binary_params(a1_name, self.A1.value)
            else:
                raise ValueError(f"No name and value to create a group with")

        for XR1n in XR1_mapping.values():
            XR1s[XR1n] = getattr(self, XR1n).quantity

        for xr1_name, xr1_value in XR1s.items():
            if (xr1_value is not None) and (xr1_name is not None):
                self.binary_instance.add_binary_params(xr1_name, xr1_value)
            else:
                raise ValueError(
                    f"No name or lower bound provided to create a group with"
                )

        for XR2n in XR2_mapping.values():
            XR2s[XR2n] = getattr(self, XR2n).quantity

        for xr2_name, xr2_value in XR2s.items():
            if (xr2_value is not None) and (xr2_name is not None):
                self.binary_instance.add_binary_params(xr2_name, xr2_value)
            else:
                raise ValueError(
                    f"No name or lower bound provided to create a group with"
                )

        self.update_binary_object(None)

    def validate(self):
        """Include catches for overlapping groups. etc
        Raises
        ------
        ValueError
                if there are pieces with no associated boundaries (T0X_0000 does not have a corresponding PLB_0000/PUB_0000)
        ValueError
                if any boundaries overlap (as it makes TOA assignment to a single group ambiguous). i.e. PLB_0000<PLB_0000 and PUB_0000>PLB_0001
        ValueError
                if the number of lower and upper bounds don't match (should only be raised if PLB is defined without PUB and validate is run or vice versa)
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

        dct_plb = self.get_prefix_mapping_component("PLB_")
        dct_pub = self.get_prefix_mapping_component("PUB_")
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
                    f"Group boundary mismatch error. Number of detected upper bounds: {j_plb}. Number of detected upper bounds:{j_pub}"
                )
            if len(np.setdiff1d(j_plb, j_pub)) > 0:
                raise ValueError(
                    f"Group index mismatch error. Check the indexes of PLB_/PUB_ parameters in the model"
                )
            if len(ls_pub) > 0 and len(ls_T0X) > 0:
                if len(np.setdiff1d(j_pub, j_T0X)) > 0:
                    raise ValueError(
                        f"Group index mismatch error. Check the indexes of T0X groups, make sure they match there are corresponding group ranges (PLB/PUB)"
                    )
            if len(ls_pub) > 0 and len(ls_A1X) > 0:
                if len(np.setdiff1d(j_pub, j_A1X)) > 0:
                    raise ValueError(
                        f"Group index mismatch error. Check the indexes of A1X groups, make sure they match there are corresponding group ranges (PLB/PUB)"
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

    def get_group_boundaries(self):
        """Get a all pieces' date boundaries.
        Returns
        -------
        list
                np.array
                        (length: toas) List of piecewise orbital parameter lower boundaries
                np.array
                        (length: toas) List of piecewise orbital parameter upper boundaries
        """
        # asks the object for the number of piecewise groups
        return self.binary_instance.get_group_boundaries()

    def which_group_is_toa_in(self, toa):
        """Find the group a toa belongs to based on the boundaries of groups passed to BT_piecewise
        Parameters
        ----------
        toa : toa
           TOA/TOAs to check which group they're in
        Returns
        -------
        np.array
           str elements, look like ['0000','0001'] for two TOAs where one refences T0X_0000 or T0X_0001.
        """
        # asks the model what group a single TOA/list of TOAs
        barycentric_toa = self._parent.get_barycentric_toas(toa)
        return self.binary_instance.toa_belongs_in_group(barycentric_toa)

    def get_number_of_groups(self):
        """Get the number of piecewise parameters"""
        return len(self.binary_instance.piecewise_parameter_information)

    def get_group_indexes(self):
        """Get all the piecewise parameter labels
        Returns
        -------
        np.array
                (length: number of piecewise groups) List of piecewise parameter labels e.g with pieces T0X_0000, T0X_0001, T0X_0003, returns [0,1,3]
        """
        group_indexes = []
        for i in range(0, len(self.binary_instance.piecewise_parameter_information)):
            group_indexes.append(
                self.binary_instance.piecewise_parameter_information[i][0]
            )
        return group_indexes

    def get_group_indexes_in_four_digit_format(self):
        """Get all the piecewise parameter labels in 4 digit format
        Returns
        -------
        np.array
                (length: number of piecewise groups) List of piecewise parameter labels in 4 digit format e.g with pieces T0X_0000, T0X_0001, T0X_0003, returns [0000,0001,0003]
        """
        group_indexes = []
        for i in range(0, len(self.binary_instance.piecewise_parameter_information)):
            group_indexes.append(
                f"{int(self.binary_instance.piecewise_parameter_information[i][0]):04d}"
            )
        return group_indexes

    def get_T0Xs_associated_with_toas(self, toas):
        """Get a of all the piecewise T0s associated with TOAs
        Parameters
        ----------
        toas :
                Barycentric TOAs
        Returns
        -------
        np.array
                (length: toas) List of piecewise T0X values being used for each TOA
        """
        if hasattr(self.binary_instance, "group_index_array"):
            temporary_storage = self.binary_instance.group_index_array
        self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
        barycentric_toa = self._parent.get_barycentric_toas(toas)
        T0X_per_toa = self.binary_instance.piecewise_parameter_from_information_array(
            toas
        )[0]
        if temporary_storage is not None:
            self.binary_instance.group_index_array = temporary_storage
        return T0X_per_toa

    def get_A1Xs_associated_with_toas(self, toas):
        """Get a of all the piecewise A1s associated with TOAs
        Parameters
        ----------
        toas :
                Barycentric TOAs
        Returns
        -------
        np.array
                (length: toas) List of piecewise A1X values being used for each TOA
        """
        if hasattr(self.binary_instance, "group_index_array"):
            temporary_storage = self.binary_instance.group_index_array
        self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
        barycentric_toa = self._parent.get_barycentric_toas(toas)
        A1X_per_toa = self.binary_instance.piecewise_parameter_from_information_array(
            toas
        )[1]
        if temporary_storage is not None:
            self.binary_instance.group_index_array = temporary_storage
        return A1X_per_toa

    def does_toa_reference_piecewise_parameter(self, toas, param):
        """Query whether a TOA/list of TOAs belong(s) to a specific group
        Parameters
        ----------
        toas :
                Barycentric TOAs
        param : str
                Orbital piecewise parameter alias  e.g. "T0X_0001" or "A1X_0001"
        Returns
        -------
        np.array
                boolean array (length: toas). True where toa is within piece boundaries corresponding to param
        """
        self.binary_instance.group_index_array = self.which_group_is_toa_in(toas)
        from_in_piece = self.binary_instance.in_piece(param)
        return from_in_piece[0]
