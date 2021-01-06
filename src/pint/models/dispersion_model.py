"""A simple model of a base dispersion delay and DMX dispersion."""
from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from pint.models.parameter import (
    MJDParameter,
    floatParameter,
    prefixParameter,
    maskParameter,
)
from pint.models.timing_model import DelayComponent, MissingParameter, MissingTOAs
from pint.toa_select import TOASelect
from pint.utils import split_prefixed_name, taylor_horner, taylor_horner_deriv

# This value is cited from Duncan Lorimer, Michael Kramer, Handbook of Pulsar
# Astronomy, Second edition, Page 86, Note 1
DMconst = 1.0 / 2.41e-4 * u.MHz * u.MHz * u.s * u.cm ** 3 / u.pc


class Dispersion(DelayComponent):
    """A base dispersion timing model."""

    def __init__(self):
        super(Dispersion, self).__init__()
        self.dm_value_funcs = []
        self.dm_deriv_funcs = {}

    def dispersion_time_delay(self, DM, freq):
        """Return the dispersion time delay for a set of frequency.

        This equation if cited from Duncan Lorimer, Michael Kramer,
        Handbook of Pulsar Astronomy, Second edition, Page 86, Equation [4.7]
        Here we assume the reference frequency is at infinity and the EM wave
        frequency is much larger than plasma frequency.
        """
        # dm delay
        dmdelay = DM * DMconst / freq.to(u.MHz) ** 2.0
        return dmdelay.to(u.s)

    def dispersion_type_delay(self, toas):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]

        dm = self.dm_value(toas)
        return self.dispersion_time_delay(dm, bfreq)

    def dm_value(self, toas):
        """Compute modeled DM value at given TOAs.

        Parameters
        ----------
        toas : `TOAs` object or TOA table(TOAs.table)
            If given a TOAs object, it will use the whole TOA table in the
             `TOAs` object.

        Return
        ------
            DM values at given TOAs in the unit of DM.
        """
        if isinstance(toas, Table):
            toas_table = toas
        else:
            toas_table = toas.table

        dm = np.zeros(len(toas_table)) * self._parent.DM.units

        for dm_f in self.dm_value_funcs:
            dm += dm_f(toas)
        return dm

    def d_delay_d_dmparam(self, toas, param_name, acc_delay=None):
        """Derivative of delay wrt to DM parameter.

        Parameters
        ----------
        toas : `pint.TOAs` object.
            Input toas.
        param_name : str
            Derivative parameter name
        acc_delay : `astropy.quantity` or `numpy.ndarray`
            Accumlated delay values. This parameter is to keep the unified API,
            but not used in this function.
        """
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"]
        param_unit = getattr(self, param_name).units
        d_dm_d_dmparam = np.zeros(toas.ntoas) * u.pc / u.cm ** 3 / param_unit
        for df in self.dm_deriv_funcs[param_name]:
            d_dm_d_dmparam += df(toas, param_name)
        return DMconst * d_dm_d_dmparam / bfreq ** 2.0

    def register_dm_deriv_funcs(self, func, param):
        """Register the derivative function in to the deriv_func dictionaries.

        Parameters
        ----------
        func : callable
            Calculates the derivative
        param : str
            Name of parameter the derivative is with respect to

        """
        pn = self.match_param_aliases(param)

        if pn not in list(self.dm_deriv_funcs.keys()):
            self.dm_deriv_funcs[pn] = [func]
        else:
            # TODO:
            # Runing setup() mulitple times can lead to adding derivative
            # function multiple times. This prevent it from happening now. But
            # in the future, we should think a better way to do so.
            if func in self.dm_deriv_funcs[pn]:
                return
            else:
                self.dm_deriv_funcs[pn] += [func]


class DispersionDM(Dispersion):
    """Simple DM dispersion model.

    This model uses Taylor expansion to model DM variation over time. It
    can also be used for a constant DM.

    """

    register = True
    category = "dispersion_constant"

    def __init__(self):
        super(DispersionDM, self).__init__()
        self.add_param(
            floatParameter(
                name="DM",
                units="pc cm^-3",
                value=0.0,
                description="Dispersion measure",
                long_double=True,
            )
        )
        self.add_param(
            prefixParameter(
                name="DM1",
                value=0.0,
                units="pc cm^-3/yr^1",
                description="First order time derivative of the dispersion measure",
                unit_template=self.DM_dervative_unit,
                description_template=self.DM_dervative_description,
                type_match="float",
                long_double=True,
            )
        )
        self.add_param(
            MJDParameter(
                name="DMEPOCH", description="Epoch of DM measurement", time_scale="tdb"
            )
        )

        self.dm_value_funcs += [self.base_dm]
        self.delay_funcs_component += [self.constant_dispersion_delay]

    def setup(self):
        super(Dispersion, self).setup()
        base_dms = list(self.get_prefix_mapping_component("DM").values())
        base_dms += ["DM"]

        for dm_name in base_dms:
            self.register_deriv_funcs(self.d_delay_d_dmparam, dm_name)
            self.register_dm_deriv_funcs(self.d_dm_d_DMs, dm_name)

    def validate(self):
        """Validate the DM parameters input."""
        super(Dispersion, self).validate()
        # If DM1 is set, we need DMEPOCH
        if self.DM1.value != 0.0:
            if self.DMEPOCH.value is None:
                raise MissingParameter(
                    "Dispersion",
                    "DMEPOCH",
                    "DMEPOCH is required if DM1 or higher are set",
                )

    def DM_dervative_unit(self, n):
        return "pc cm^-3/yr^%d" % n if n else "pc cm^-3"

    def DM_dervative_description(self, n):
        return "%d'th time derivative of the dispersion measure" % n

    def get_DM_terms(self):
        """Return a list of the DM term values in the model: [DM, DM1, ..., DMn]"""
        prefix_dm = list(self.get_prefix_mapping_component("DM").values())
        dm_terms = [self.DM.quantity]
        dm_terms += [getattr(self, x).quantity for x in prefix_dm]
        return dm_terms

    def base_dm(self, toas):
        tbl = toas.table
        dm = np.zeros(len(tbl))
        dm_terms = self.get_DM_terms()
        if self.DMEPOCH.value is None:
            DMEPOCH = tbl["tdbld"][0]
        else:
            DMEPOCH = self.DMEPOCH.value
        dt = (tbl["tdbld"] - DMEPOCH) * u.day
        dt_value = (dt.to(u.yr)).value
        dm_terms_value = [d.value for d in dm_terms]
        dm = taylor_horner(dt_value, dm_terms_value)
        return dm * self.DM.units

    def constant_dispersion_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.dispersion_type_delay(toas)

    def print_par(self,):
        # TODO we need to have a better design for print out the parameters in
        # an inhertance class.
        result = ""
        prefix_dm = list(self.get_prefix_mapping_component("DM").values())
        dms = ["DM"] + prefix_dm
        for dm in dms:
            result += getattr(self, dm).as_parfile_line()
        if hasattr(self, "components"):
            all_params = self.components["DispersionDM"].params
        else:
            all_params = self.params
        for pm in all_params:
            if pm not in dms:
                result += getattr(self, pm).as_parfile_line()
        return result

    def d_dm_d_DMs(
        self, toas, param_name, acc_delay=None
    ):  # NOTE we should have a better name for this.)
        """Derivatives of DM wrt the DM taylor expansion parameters."""
        tbl = toas.table
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = tbl["freq"]
        par = getattr(self, param_name)
        unit = par.units
        if param_name == "DM":
            order = 0
        else:
            pn, idxf, idxv = split_prefixed_name(param_name)
            order = idxv
        dms = self.get_DM_terms()
        dm_terms = np.longdouble(np.zeros(len(dms)))
        dm_terms[order] = np.longdouble(1.0)
        if self.DMEPOCH.value is None:
            DMEPOCH = tbl["tdbld"][0]
        else:
            DMEPOCH = self.DMEPOCH.value
        dt = (tbl["tdbld"] - DMEPOCH) * u.day
        dt_value = (dt.to(u.yr)).value
        d_dm_d_dm_param = taylor_horner(dt_value, dm_terms) * (
            self.DM.units / par.units
        )

        return d_dm_d_dm_param

    def change_dmepoch(self, new_epoch):
        """Change DMEPOCH to a new value and update DM accordingly.

        Parameters
        ----------
        new_epoch: float MJD (in TDB) or `astropy.Time` object
            The new DMEPOCH value.
        """
        if isinstance(new_epoch, Time):
            new_epoch = Time(new_epoch, scale="tdb", precision=9)
        else:
            new_epoch = Time(new_epoch, scale="tdb", format="mjd", precision=9)

        if self.DMEPOCH.value is None:
            raise ValueError("DMEPOCH is not currently set.")

        dmepoch_ld = self.DMEPOCH.quantity.tdb.mjd_long
        dt = (new_epoch.tdb.mjd_long - dmepoch_ld) * u.day
        dmterms = [0.0 * u.Unit("")] + self.get_DM_terms()

        for n in range(len(dmterms) - 1):
            cur_deriv = self.DM if n == 0 else getattr(self, "DM{}".format(n))
            cur_deriv.value = taylor_horner_deriv(
                dt.to(u.yr), dmterms, deriv_order=n + 1
            )
        self.DMEPOCH.value = new_epoch


class DispersionDMX(Dispersion):
    """This class provides a DMX model - multiple DM values.

    This model lets the user specify time ranges and fit for a different
    DM value in each time range.
    """

    register = True
    category = "dispersion_dmx"

    def __init__(self):
        super(DispersionDMX, self).__init__()
        # DMX is for info output right now
        self.add_param(
            floatParameter(
                name="DMX",
                units="pc cm^-3",
                value=0.0,
                description="Dispersion measure",
            )
        )

        self.add_DMX_range(None, None, dmx=0, frozen=False, index=1)

        self.dm_value_funcs += [self.dmx_dm]
        self.set_special_params(["DMX_0001", "DMXR1_0001", "DMXR2_0001"])
        self.delay_funcs_component += [self.DMX_dispersion_delay]

    def add_DMX_range(self, mjd_start, mjd_end, index=None, dmx=0, frozen=True):
        """ Add DMX range to a dispersion model with specified start/end MJDs and DMX. 

        Parameters
        ----------

        mjd_start : float
            MJD for beginning of DMX event.
        mjd_end : float
            MJD for end of DMX event.
        index : int, None
            Integer label for DMX event. If None, will increment largest used index by 1. 
        dmx : float
            Change in DM during DMX event.
        frozen : bool
            Indicates whether DMX will be fit.

        Returns 
        -------

        index : int
            Index that has been assigned to new DMX event.
            
        """

        #### Setting up the DMX title convention. If index is None, want to increment the current max DMX index by 1.
        if index is None:
            dct = self.get_prefix_mapping_component("DMX_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if mjd_end is not None and mjd_start is not None:
            if mjd_end < mjd_start:
                raise ValueError("Starting MJD is greater than ending MJD.")
        elif mjd_start != mjd_end:
            raise ValueError("Only one MJD bound is set.")

        if int(index) in self.get_prefix_mapping_component("DMX_"):
            raise ValueError(
                "Index '%s' is already in use in this model. Please choose another."
                % index
            )

        self.add_param(
            prefixParameter(
                name="DMX_" + i,
                units="pc cm^-3",
                value=dmx,
                unit_template=lambda x: "pc cm^-3",
                description="Dispersion measure variation",
                description_template=lambda x: "Dispersion measure",
                parameter_type="float",
                frozen=frozen,
            )
        )
        self.add_param(
            prefixParameter(
                name="DMXR1_" + i,
                units="MJD",
                unit_template=lambda x: "MJD",
                description="Beginning of DMX interval",
                description_template=lambda x: "Beginning of DMX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_start,
            )
        )
        self.add_param(
            prefixParameter(
                name="DMXR2_" + i,
                units="MJD",
                unit_template=lambda x: "MJD",
                description="End of DMX interval",
                description_template=lambda x: "End of DMX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_end,
            )
        )
        self.setup()
        self.validate()
        return index

    def remove_DMX_range(self, index):
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
            for prefix in ["DMX_", "DMXR1_", "DMXR2_"]:
                self.remove_param(prefix + index_rf)
        self.validate()

    def get_indices(self):
        """Returns an array of integers corresponding to DMX parameters. 

        Returns
        -------
        inds : np.ndarray
        Array of DMX indices in model.
        """
        inds = []
        for p in self.params:
            if "DMX_" in p:
                inds.append(int(p.split("_")[-1]))
        return np.array(inds)

    def setup(self):
        super(DispersionDMX, self).setup()
        # Get DMX mapping.
        # Register the DMX derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("DMX_"):
                self.register_deriv_funcs(self.d_delay_d_dmparam, prefix_par)
                self.register_dm_deriv_funcs(self.d_dm_d_DMX, prefix_par)

    def validate(self):
        """Validate the DMX parameters."""
        super(DispersionDMX, self).validate()
        DMX_mapping = self.get_prefix_mapping_component("DMX_")
        DMXR1_mapping = self.get_prefix_mapping_component("DMXR1_")
        DMXR2_mapping = self.get_prefix_mapping_component("DMXR2_")
        if DMX_mapping.keys() != DMXR1_mapping.keys():
            # FIXME: report mismatch
            raise ValueError(
                "DMX_ parameters do not "
                "match DMXR1_ parameters. "
                "Please check your prefixed parameters."
            )
        if DMX_mapping.keys() != DMXR2_mapping.keys():
            raise ValueError(
                "DMX_ parameters do not "
                "match DMXR2_ parameters. "
                "Please check your prefixed parameters."
            )

    def validate_toas(self, toas):
        DMX_mapping = self.get_prefix_mapping_component("DMX_")
        DMXR1_mapping = self.get_prefix_mapping_component("DMXR1_")
        DMXR2_mapping = self.get_prefix_mapping_component("DMXR2_")
        bad_parameters = []
        for k in DMXR1_mapping.keys():
            if self._parent[DMX_mapping[k]].frozen:
                continue
            b = self._parent[DMXR1_mapping[k]].quantity.mjd * u.d
            e = self._parent[DMXR2_mapping[k]].quantity.mjd * u.d
            mjds = toas.get_mjds()
            n = np.sum((b <= mjds) & (mjds < e))
            if n == 0:
                bad_parameters.append(DMX_mapping[k])
        if bad_parameters:
            raise MissingTOAs(bad_parameters)

    def dmx_dm(self, toas):
        condition = {}
        tbl = toas.table
        if not hasattr(self, "dmx_toas_selector"):
            self.dmx_toas_selector = TOASelect(is_range=True)
        DMX_mapping = self.get_prefix_mapping_component("DMX_")
        DMXR1_mapping = self.get_prefix_mapping_component("DMXR1_")
        DMXR2_mapping = self.get_prefix_mapping_component("DMXR2_")
        for epoch_ind in DMX_mapping.keys():
            r1 = getattr(self, DMXR1_mapping[epoch_ind]).quantity
            r2 = getattr(self, DMXR2_mapping[epoch_ind]).quantity
            condition[DMX_mapping[epoch_ind]] = (r1.mjd, r2.mjd)
        select_idx = self.dmx_toas_selector.get_select_index(
            condition, tbl["mjd_float"]
        )
        # Get DMX delays
        dm = np.zeros(len(tbl)) * self._parent.DM.units
        for k, v in select_idx.items():
            dm[v] = getattr(self, k).quantity
        return dm

    def DMX_dispersion_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.dispersion_type_delay(toas)

    def d_dm_d_DMX(self, toas, param_name, acc_delay=None):
        condition = {}
        tbl = toas.table
        if not hasattr(self, "dmx_toas_selector"):
            self.dmx_toas_selector = TOASelect(is_range=True)
        param = getattr(self, param_name)
        dmx_index = param.index
        DMXR1_mapping = self.get_prefix_mapping_component("DMXR1_")
        DMXR2_mapping = self.get_prefix_mapping_component("DMXR2_")
        r1 = getattr(self, DMXR1_mapping[dmx_index]).quantity
        r2 = getattr(self, DMXR2_mapping[dmx_index]).quantity
        condition = {param_name: (r1.mjd, r2.mjd)}
        select_idx = self.dmx_toas_selector.get_select_index(
            condition, tbl["mjd_float"]
        )

        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = tbl["freq"]
        dmx = np.zeros(len(tbl))
        for k, v in select_idx.items():
            dmx[v] = 1.0
        return dmx * (u.pc / u.cm ** 3) / (u.pc / u.cm ** 3)

    def print_par(self,):
        result = ""
        DMX_mapping = self.get_prefix_mapping_component("DMX_")
        DMXR1_mapping = self.get_prefix_mapping_component("DMXR1_")
        DMXR2_mapping = self.get_prefix_mapping_component("DMXR2_")
        result += getattr(self, "DMX").as_parfile_line()
        sorted_list = sorted(DMX_mapping.keys())
        for ii in sorted_list:
            result += getattr(self, DMX_mapping[ii]).as_parfile_line()
            result += getattr(self, DMXR1_mapping[ii]).as_parfile_line()
            result += getattr(self, DMXR2_mapping[ii]).as_parfile_line()
        return result


class DispersionJump(Dispersion):
    """This class provides the contant offsets to the DM values.

    Notes
    -----
    This DM jump is only for modeling the DM values, and will not apply to the
    dispersion time delay.
    """

    register = True
    category = "dispersion_jump"

    def __init__(self):
        super(DispersionJump, self).__init__()
        self.dm_value_funcs += [self.jump_dm]
        # Dispersion jump only model the dm values.

        self.add_param(
            maskParameter(
                name="DMJUMP",
                units="pc cm^-3",
                value=None,
                description="DM value offset.",
            )
        )

    def setup(self):
        super(DispersionJump, self).setup()
        self.dm_jumps = []
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("DMJUMP"):
                self.dm_jumps.append(mask_par)
        for j in self.dm_jumps:
            self.register_dm_deriv_funcs(self.d_dm_d_dmjump, j)
            # Note we can not use the derivative function 'd_delay_d_dmparam',
            # Since dmjump does not effect delay.
            # The function 'd_delay_d_dmparam' applies d_dm_d_dmparam first and
            # than applys the time delay part.
            self.register_deriv_funcs(self.d_delay_d_dmjump, j)

    def validate(self):
        super(DispersionJump, self).validate()

    def jump_dm(self, toas):
        """Return the DM jump for each dm section collected by dmjump parameters.

        The delay value is determined by DMJUMP parameter
        value in the unit of pc / cm ** 3.
        """
        tbl = toas.table
        jdm = np.zeros(len(tbl))
        for dm_jump in self.dm_jumps:
            dm_jump_par = getattr(self, dm_jump)
            mask = dm_jump_par.select_toa_mask(toas)
            jdm[mask] += -dm_jump_par.value
        return jdm * dm_jump_par.units

    def d_dm_d_dmjump(self, toas, jump_param):
        """Derivative of dm values wrt dm jumps."""
        tbl = toas.table
        d_dm_d_j = np.zeros(len(tbl))
        jpar = getattr(self, jump_param)
        mask = jpar.select_toa_mask(toas)
        d_dm_d_j[mask] = -1.0
        return d_dm_d_j * jpar.units / jpar.units

    def d_delay_d_dmjump(self, toas, param_name, acc_delay=None):
        """Derivative for delay wrt to dm jumps.

        Since DMJUMPS does not affect delay, this would be zero.
        """
        dmjump = getattr(self, param_name)
        return np.zeros(toas.ntoas) * (u.s / dmjump.units)
