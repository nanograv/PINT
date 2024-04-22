"""A simple model of a base dispersion delay and DMX dispersion."""

from warnings import warn

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from loguru import logger as log

from pint.models.parameter import (
    MJDParameter,
    floatParameter,
    prefixParameter,
    maskParameter,
)
from pint.models.timing_model import DelayComponent, MissingParameter, MissingTOAs
from pint.toa_select import TOASelect
from pint.utils import (
    split_prefixed_name,
    taylor_horner,
    taylor_horner_deriv,
    get_prefix_timeranges,
)
from pint import DMconst

# This value is cited from Duncan Lorimer, Michael Kramer, Handbook of Pulsar
# Astronomy, Second edition, Page 86, Note 1
# DMconst = 1.0 / 2.41e-4 * u.MHz * u.MHz * u.s * u.cm**3 / u.pc


class Dispersion(DelayComponent):
    """A base dispersion timing model.

    See https://nanograv-pint.readthedocs.io/en/latest/explanation.html#dispersion-measure
    for an explanation on the dispersion delay and dispersion measure."""

    def __init__(self):
        super().__init__()
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
        toas_table = toas if isinstance(toas, Table) else toas.table
        dm = np.zeros(len(toas_table)) * self._parent.DM.units

        for dm_f in self.dm_value_funcs:
            dm += dm_f(toas)
        return dm

    def dispersion_slope_value(self, toas):
        return

    def d_delay_d_dmparam(self, toas, param_name, acc_delay=None):
        """Derivative of delay wrt to DM parameter.

        Parameters
        ----------
        toas : `pint.TOAs` object.
            Input toas.
        param_name : str
            Derivative parameter name
        acc_delay : `astropy.quantity` or `numpy.ndarray`
            Accumulated delay values. This parameter is to keep the unified API,
            but not used in this function.
        """
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas.table["freq"].quantity
        param_unit = getattr(self, param_name).units
        d_dm_d_dmparam = np.zeros(toas.ntoas) * u.pc / u.cm**3 / param_unit
        for df in self.dm_deriv_funcs[param_name]:
            d_dm_d_dmparam += df(toas, param_name)
        return DMconst * d_dm_d_dmparam / bfreq**2.0

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
        elif func in self.dm_deriv_funcs[pn]:
            return
        else:
            self.dm_deriv_funcs[pn] += [func]


class DispersionDM(Dispersion):
    """Simple DM dispersion model.

    This model uses Taylor expansion to model DM variation over time. It
    can also be used for a constant DM.

    Parameters supported:

    .. paramtable::
        :class: pint.models.dispersion_model.DispersionDM
    """

    register = True
    category = "dispersion_constant"

    def __init__(self):
        super().__init__()
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
        super().setup()
        base_dms = list(self.get_prefix_mapping_component("DM").values())
        base_dms += ["DM"]

        for dm_name in base_dms:
            self.register_deriv_funcs(self.d_delay_d_dmparam, dm_name)
            self.register_dm_deriv_funcs(self.d_dm_d_DMs, dm_name)

    def validate(self):
        """Validate the DM parameters input."""
        super().validate()
        # If DM1 is set, we need DMEPOCH
        if self.DM1.value is not None and self.DM1.value != 0.0:
            if self.DMEPOCH.value is None:
                # Copy PEPOCH (PEPOCH must be set!)
                self.DMEPOCH.value = self._parent.PEPOCH.value
            if self.DMEPOCH.value is None:
                raise MissingParameter(
                    "Dispersion",
                    "DMEPOCH",
                    "DMEPOCH or PEPOCH is required if DM1 or higher are set",
                )

    def DM_dervative_unit(self, n):
        return "pc cm^-3/yr^%d" % n if n else "pc cm^-3"

    def DM_dervative_description(self, n):
        return "%d'th time derivative of the dispersion measure" % n

    def get_DM_terms(self):
        """Return a list of the DM term values in the model: [DM, DM1, ..., DMn]"""
        return [self.DM.quantity] + self._parent.get_prefix_list("DM", start_index=1)

    def base_dm(self, toas):
        dm = np.zeros(len(toas))
        dm_terms = self.get_DM_terms()
        if any(t.value != 0 for t in dm_terms[1:]):
            DMEPOCH = self.DMEPOCH.value
            if DMEPOCH is None:
                # Should be ruled out by validate()
                raise ValueError(
                    f"DMEPOCH not set but some derivatives are not zero: {dm_terms}"
                )
            else:
                dt = (toas["tdbld"] - DMEPOCH) * u.day
            dt_value = dt.to_value(u.yr)
        else:
            dt_value = np.zeros(len(toas), dtype=np.longdouble)
        dm_terms_value = [d.value for d in dm_terms]
        dm = taylor_horner(dt_value, dm_terms_value)
        return dm * self.DM.units

    def constant_dispersion_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.dispersion_type_delay(toas)

    def print_par(self, format="pint"):
        prefix_dm = list(self.get_prefix_mapping_component("DM").values())
        dms = ["DM"] + prefix_dm
        result = "".join(getattr(self, dm).as_parfile_line(format=format) for dm in dms)
        if hasattr(self, "components"):
            all_params = self.components["DispersionDM"].params
        else:
            all_params = self.params
        for pm in all_params:
            if pm not in dms:
                result += getattr(self, pm).as_parfile_line(format=format)
        return result

    def d_dm_d_DMs(
        self, toas, param_name, acc_delay=None
    ):  # NOTE we should have a better name for this.)
        """Derivatives of DM wrt the DM taylor expansion parameters."""
        par = getattr(self, param_name)
        if param_name == "DM":
            order = 0
        else:
            pn, idxf, idxv = split_prefixed_name(param_name)
            order = idxv
        dms = self.get_DM_terms()
        dm_terms = np.longdouble(np.zeros(len(dms)))
        dm_terms[order] = np.longdouble(1.0)
        if self.DMEPOCH.value is None:
            if any(t.value != 0 for t in dms[1:]):
                # Should be ruled out by validate()
                raise ValueError(f"DMEPOCH is not set but {param_name} is not zero")
            DMEPOCH = 0
        else:
            DMEPOCH = self.DMEPOCH.value
        dt = (toas["tdbld"] - DMEPOCH) * u.day
        dt_value = (dt.to(u.yr)).value
        return taylor_horner(dt_value, dm_terms) * (self.DM.units / par.units)

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

        dmterms = [0.0 * u.Unit("")] + self.get_DM_terms()
        if self.DMEPOCH.value is None:
            if any(d.value != 0 for d in dmterms[2:]):
                # Should be ruled out by validate()
                raise ValueError(
                    f"DMEPOCH not set but some DM derivatives are not zero: {dmterms}"
                )
            self.DMEPOCH.value = new_epoch

        dmepoch_ld = self.DMEPOCH.quantity.tdb.mjd_long
        dt = (new_epoch.tdb.mjd_long - dmepoch_ld) * u.day

        for n in range(len(dmterms) - 1):
            cur_deriv = self.DM if n == 0 else getattr(self, f"DM{n}")
            cur_deriv.value = taylor_horner_deriv(
                dt.to(u.yr), dmterms, deriv_order=n + 1
            )
        self.DMEPOCH.value = new_epoch


class DispersionDMX(Dispersion):
    """This class provides a DMX model - multiple DM values.

    This model lets the user specify time ranges and fit for a different
    DM value in each time range.

    Parameters supported:

    .. paramtable::
        :class: pint.models.dispersion_model.DispersionDMX
    """

    register = True
    category = "dispersion_dmx"

    def __init__(self):
        super().__init__()
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
        """Add DMX range to a dispersion model with specified start/end MJDs and DMX.

        Parameters
        ----------

        mjd_start : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for beginning of DMX event.
        mjd_end : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for end of DMX event.
        index : int, None
            Integer label for DMX event. If None, will increment largest used index by 1.
        dmx : float or astropy.quantity.Quantity
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
                f"Index '{index}' is already in use in this model. Please choose another."
            )

        if isinstance(dmx, u.quantity.Quantity):
            dmx = dmx.to_value(u.pc / u.cm**3)
        if isinstance(mjd_start, Time):
            mjd_start = mjd_start.mjd
        elif isinstance(mjd_start, u.quantity.Quantity):
            mjd_start = mjd_start.value
        if isinstance(mjd_end, Time):
            mjd_end = mjd_end.mjd
        elif isinstance(mjd_end, u.quantity.Quantity):
            mjd_end = mjd_end.value
        self.add_param(
            prefixParameter(
                name=f"DMX_{i}",
                units="pc cm^-3",
                value=dmx,
                description="Dispersion measure variation",
                parameter_type="float",
                frozen=frozen,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"DMXR1_{i}",
                units="MJD",
                description="Beginning of DMX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_start,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"DMXR2_{i}",
                units="MJD",
                description="End of DMX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_end,
            )
        )
        self.setup()
        self.validate()
        return index

    def add_DMX_ranges(self, mjd_starts, mjd_ends, indices=None, dmxs=0, frozens=True):
        """Add DMX ranges to a dispersion model with specified start/end MJDs and DMXs.

        Parameters
        ----------

        mjd_starts : iterable of float or astropy.quantity.Quantity or astropy.time.Time
            MJD for beginning of DMX event.
        mjd_end : iterable of float or astropy.quantity.Quantity or astropy.time.Time
            MJD for end of DMX event.
        indices : iterable of int, None
            Integer label for DMX event. If None, will increment largest used index by 1.
        dmxs : iterable of float or astropy.quantity.Quantity, or float or astropy.quantity.Quantity
            Change in DM during DMX event.
        frozens : iterable of bool or bool
            Indicates whether DMX will be fit.

        Returns
        -------

        indices : list
            Indices that has been assigned to new DMX events

        """
        if len(mjd_starts) != len(mjd_ends):
            raise ValueError(
                f"Number of mjd_start values {len(mjd_starts)} must match number of mjd_end values {len(mjd_ends)}"
            )
        if indices is None:
            indices = [None] * len(mjd_starts)
        dmxs = np.atleast_1d(dmxs)
        if len(dmxs) == 1:
            dmxs = np.repeat(dmxs, len(mjd_starts))
        if len(dmxs) != len(mjd_starts):
            raise ValueError(
                f"Number of mjd_start values {len(mjd_starts)} must match number of dmx values {len(dmxs)}"
            )
        frozens = np.atleast_1d(frozens)
        if len(frozens) == 1:
            frozens = np.repeat(frozens, len(mjd_starts))
        if len(frozens) != len(mjd_starts):
            raise ValueError(
                f"Number of mjd_start values {len(mjd_starts)} must match number of frozen values {len(frozens)}"
            )

        #### Setting up the DMX title convention. If index is None, want to increment the current max DMX index by 1.
        dct = self.get_prefix_mapping_component("DMX_")
        last_index = np.max(list(dct.keys()))
        added_indices = []
        for mjd_start, mjd_end, index, dmx, frozen in zip(
            mjd_starts, mjd_ends, indices, dmxs, frozens
        ):
            if index is None:
                index = last_index + 1
                last_index += 1
            elif index in list(dct.keys()):
                raise ValueError(
                    f"Attempting to insert DMX_{index:04d} but it already exists"
                )
            added_indices.append(index)
            i = f"{int(index):04d}"

            if mjd_end is not None and mjd_start is not None:
                if mjd_end < mjd_start:
                    raise ValueError("Starting MJD is greater than ending MJD.")
            elif mjd_start != mjd_end:
                raise ValueError("Only one MJD bound is set.")
            if int(index) in dct:
                raise ValueError(
                    f"Index '{index}' is already in use in this model. Please choose another."
                )
            if isinstance(dmx, u.quantity.Quantity):
                dmx = dmx.to_value(u.pc / u.cm**3)
            if isinstance(mjd_start, Time):
                mjd_start = mjd_start.mjd
            elif isinstance(mjd_start, u.quantity.Quantity):
                mjd_start = mjd_start.value
            if isinstance(mjd_end, Time):
                mjd_end = mjd_end.mjd
            elif isinstance(mjd_end, u.quantity.Quantity):
                mjd_end = mjd_end.value
            log.trace(f"Adding DMX_{i} from MJD {mjd_start} to MJD {mjd_end}")
            self.add_param(
                prefixParameter(
                    name=f"DMX_{i}",
                    units="pc cm^-3",
                    value=dmx,
                    description="Dispersion measure variation",
                    parameter_type="float",
                    frozen=frozen,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"DMXR1_{i}",
                    units="MJD",
                    description="Beginning of DMX interval",
                    parameter_type="MJD",
                    time_scale="utc",
                    value=mjd_start,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"DMXR2_{i}",
                    units="MJD",
                    description="End of DMX interval",
                    parameter_type="MJD",
                    time_scale="utc",
                    value=mjd_end,
                )
            )
        self.setup()
        self.validate()
        return added_indices

    def remove_DMX_range(self, index):
        """Removes all DMX parameters associated with a given index/list of indices.

        Parameters
        ----------

        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to DMX indices to be removed from model.
        """

        if isinstance(index, (int, float, np.int64)):
            indices = [index]
        elif isinstance(index, (list, set, np.ndarray)):
            indices = index
        else:
            raise TypeError(
                f"index must be a float, int, set, list, or array - not {type(index)}"
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
        inds = [int(p.split("_")[-1]) for p in self.params if "DMX_" in p]
        return np.array(inds)

    def setup(self):
        super().setup()
        # Get DMX mapping.
        # Register the DMX derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("DMX_"):
                self.register_deriv_funcs(self.d_delay_d_dmparam, prefix_par)
                self.register_dm_deriv_funcs(self.d_dm_d_DMX, prefix_par)

    def validate(self):
        """Validate the DMX parameters."""
        super().validate()
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
        r1 = np.zeros(len(DMX_mapping))
        r2 = np.zeros(len(DMX_mapping))
        indices = np.zeros(len(DMX_mapping), dtype=np.int32)
        for j, index in enumerate(DMX_mapping):
            if (
                getattr(self, f"DMXR1_{index:04d}").quantity is not None
                and getattr(self, f"DMXR2_{index:04d}").quantity is not None
            ):
                r1[j] = getattr(self, f"DMXR1_{index:04d}").quantity.mjd
                r2[j] = getattr(self, f"DMXR2_{index:04d}").quantity.mjd
                indices[j] = index
        for j, index in enumerate(DMXR1_mapping):
            if np.any((r1[j] > r1) & (r1[j] < r2)):
                k = np.where((r1[j] > r1) & (r1[j] < r2))[0]
                for kk in k.flatten():
                    log.warning(
                        f"Start of DMX_{index:04d} ({r1[j]}-{r2[j]}) overlaps with DMX_{indices[kk]:04d} ({r1[kk]}-{r2[kk]})"
                    )
            if np.any((r2[j] > r1) & (r2[j] < r2)):
                k = np.where((r2[j] > r1) & (r2[j] < r2))[0]
                for kk in k.flatten():
                    log.warning(
                        f"End of DMX_{index:04d} ({r1[j]}-{r2[j]}) overlaps with DMX_{indices[kk]:04d} ({r1[kk]}-{r2[kk]})"
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
            dm[v] += getattr(self, k).quantity
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
        return dmx * (u.pc / u.cm**3) / (u.pc / u.cm**3)

    def print_par(self, format="pint"):
        result = ""
        DMX_mapping = self.get_prefix_mapping_component("DMX_")
        DMXR1_mapping = self.get_prefix_mapping_component("DMXR1_")
        DMXR2_mapping = self.get_prefix_mapping_component("DMXR2_")
        result += getattr(self, "DMX").as_parfile_line(format=format)
        sorted_list = sorted(DMX_mapping.keys())
        for ii in sorted_list:
            result += getattr(self, DMX_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, DMXR1_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, DMXR2_mapping[ii]).as_parfile_line(format=format)
        return result


class DispersionJump(Dispersion):
    """This class provides the constant offsets to the DM values.

    Parameters supported:

    .. paramtable::
        :class: pint.models.dispersion_model.DispersionJump

    Notes
    -----
    This DM jump is only for modeling the DM values, and will not apply to the
    dispersion time delay.
    """

    register = True
    category = "dispersion_jump"

    def __init__(self):
        super().__init__()
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
        super().setup()
        self.dm_jumps = []
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("DMJUMP"):
                self.dm_jumps.append(mask_par)
        for j in self.dm_jumps:
            self.register_dm_deriv_funcs(self.d_dm_d_dmjump, j)
            # Note we can not use the derivative function 'd_delay_d_dmparam',
            # Since dmjump does not effect delay.
            # The function 'd_delay_d_dmparam' applies d_dm_d_dmparam first and
            # than applies the time delay part.
            self.register_deriv_funcs(self.d_delay_d_dmjump, j)

    def validate(self):
        super().validate()

    def jump_dm(self, toas):
        """Return the DM jump for each DM section collected by DMJUMP parameters.

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
        """Derivative of the DM values w.r.t DM jumps."""
        tbl = toas.table
        d_dm_d_j = np.zeros(len(tbl))
        jpar = getattr(self, jump_param)
        mask = jpar.select_toa_mask(toas)
        d_dm_d_j[mask] = -1.0
        return d_dm_d_j * u.dimensionless_unscaled

    def d_delay_d_dmjump(self, toas, param_name, acc_delay=None):
        """Derivative of the delay w.r.t DM jumps.

        Since DMJUMPs do not affect the delay, this should be zero.
        """
        dmjump = getattr(self, param_name)
        return np.zeros(toas.ntoas) * (u.s / dmjump.units)


class FDJumpDM(Dispersion):
    """This class provides system-dependent DM offsets for narrow-band
    datasets. Such offsets can arise if different fiducial DMs are used
    to dedisperse the template profiles used to derive the TOAs for different
    systems. They can also arise while combining TOAs obtained using frequency-
    collapsed templates with those obtained using frequency-resolved templates.

    FDJumpDM is not to be confused with DMJump, which provides a DM offset
    without providing the corresponding DM delay. DMJump is specific to
    wideband datasets whereas FDJumpDM is intended to be used with narrowband
    datasets.

    This component is called FDJumpDM because the name DMJump was already taken,
    and because this is often used in conjunction with FDJumps which account for
    the fact that the templates may not adequately model the frequency-dependent
    profile evolution.

    Parameters supported:

    .. paramtable::
        :class: pint.models.dispersion_model.FDJumpDM
    """

    register = True
    category = "fdjumpdm"

    def __init__(self):
        super().__init__()
        self.dm_value_funcs += [self.fdjump_dm]
        self.delay_funcs_component += [self.fdjump_dm_delay]

        self.add_param(
            maskParameter(
                name="FDJUMPDM",
                units="pc cm^-3",
                value=None,
                description="System-dependent DM offset.",
            )
        )

    def setup(self):
        super().setup()
        self.fdjump_dms = []
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("FDJUMPDM"):
                self.fdjump_dms.append(mask_par)
        for j in self.fdjump_dms:
            self.register_dm_deriv_funcs(self.d_dm_d_fdjumpdm, j)
            self.register_deriv_funcs(self.d_delay_d_dmparam, j)

    def validate(self):
        super().validate()

    def fdjump_dm(self, toas):
        """Return the system-dependent DM offset.

        The delay value is determined by FDJUMPDM parameter
        value in the unit of pc / cm ** 3.
        """
        tbl = toas.table
        jdm = np.zeros(len(tbl))
        for fdjumpdm in self.fdjump_dms:
            fdjumpdm_par = getattr(self, fdjumpdm)
            mask = fdjumpdm_par.select_toa_mask(toas)
            jdm[mask] += -fdjumpdm_par.value
        return jdm * fdjumpdm_par.units

    def fdjump_dm_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.dispersion_type_delay(toas)

    def d_dm_d_fdjumpdm(self, toas, jump_param):
        """Derivative of DM values w.r.t FDJUMPDM parameters."""
        tbl = toas.table
        d_dm_d_j = np.zeros(len(tbl))
        jpar = getattr(self, jump_param)
        mask = jpar.select_toa_mask(toas)
        d_dm_d_j[mask] = -1.0
        return d_dm_d_j * u.dimensionless_unscaled
