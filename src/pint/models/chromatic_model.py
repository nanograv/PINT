from warnings import warn

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.time import Time
from loguru import logger as log

from pint import DMconst
from pint.exceptions import MissingParameter
from pint.models.parameter import MJDParameter, floatParameter, prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter, MissingTOAs
from pint.toa import TOAs
from pint.toa_select import TOASelect
from pint.utils import split_prefixed_name, taylor_horner, taylor_horner_deriv

cmu = u.pc / u.cm**3 / u.MHz**2


class Chromatic(DelayComponent):
    """A base chromatic timing model with a constant chromatic index."""

    def __init__(self):
        super().__init__()

        self.cm_value_funcs = []
        self.cm_deriv_funcs = {}

        self.alpha_deriv_funcs = {}

    def chromatic_time_delay(self, cm, alpha, freq):
        """Return the chromatic time delay for a set of frequencies.

        delay_chrom = cm * DMconst * (freq / 1 MHz)**alpha
        """
        cmdelay = cm * DMconst * (freq / u.MHz) ** (-alpha)
        return cmdelay.to(u.s)

    def chromatic_type_delay(self, toas):
        try:
            bfreq = self._parent.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for chromatic delay!")
            bfreq = toas.table["freq"]

        cm = self.cm_value(toas)
        alpha = self._parent["TNCHROMIDX"].quantity
        return self.chromatic_time_delay(cm, alpha, bfreq)

    def cm_value(self, toas):
        """Compute modeled CM value at given TOAs.

        Parameters
        ----------
        toas : `TOAs` object or TOA table(TOAs.table)
            If given a TOAs object, it will use the whole TOA table in the
             `TOAs` object.

        Return
        ------
            CM values at given TOAs in the unit of CM.
        """
        toas_table = toas if isinstance(toas, Table) else toas.table
        cm = np.zeros(len(toas_table)) * self._parent.CM.units

        for cm_f in self.cm_value_funcs:
            cm += cm_f(toas)
        return cm

    def d_delay_d_cmparam(self, toas, param_name, acc_delay=None):
        """Derivative of delay wrt to CM parameter.

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
            bfreq = toas.table["freq"]

        param_unit = getattr(self, param_name).units
        d_cm_d_cmparam = np.zeros(toas.ntoas) * cmu / param_unit
        alpha = self._parent["TNCHROMIDX"].quantity

        for df in self.cm_deriv_funcs[param_name]:
            d_cm_d_cmparam += df(toas, param_name)

        return DMconst * d_cm_d_cmparam * (bfreq / u.MHz) ** (-alpha)

    def register_cm_deriv_funcs(self, func, param):
        """Register the derivative function in to the deriv_func dictionaries.

        Parameters
        ----------
        func : callable
            Calculates the derivative
        param : str
            Name of parameter the derivative is with respect to

        """
        pn = self.match_param_aliases(param)

        if pn not in list(self.cm_deriv_funcs.keys()):
            self.cm_deriv_funcs[pn] = [func]
        elif func in self.cm_deriv_funcs[pn]:
            return
        else:
            self.cm_deriv_funcs[pn] += [func]


class ChromaticCM(Chromatic):
    """Simple chromatic delay model with a constant chromatic index.

    This model uses Taylor expansion to model CM variation over time. It
    can also be used for a constant CM.

    Fitting for the chromatic index is not supported because the fit is too
    unstable when fit simultaneously with the DM.

    Parameters supported:

    .. paramtable::
        :class: pint.models.chromatic_model.ChromaticCM
    """

    register = True
    category = "chromatic_constant"

    def __init__(self):
        super().__init__()
        self.add_param(
            floatParameter(
                name="CM",
                units=cmu,
                value=0.0,
                description="Chromatic measure",
                long_double=True,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            prefixParameter(
                name="CM1",
                units=cmu / u.year,
                description="First order time derivative of the chromatic measure",
                unit_template=self.CM_derivative_unit,
                description_template=self.CM_derivative_description,
                type_match="float",
                long_double=True,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            MJDParameter(
                name="CMEPOCH",
                description="Epoch of CM measurement",
                time_scale="tdb",
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            floatParameter(
                name="TNCHROMIDX",
                units=u.dimensionless_unscaled,
                value=4.0,
                description="Chromatic measure index",
                long_double=True,
                convert_tcb2tdb=False,
            )
        )

        self.cm_value_funcs += [self.base_cm]
        self.delay_funcs_component += [self.constant_chromatic_delay]

    def setup(self):
        super().setup()
        base_cms = list(self.get_prefix_mapping_component("CM").values())
        base_cms += ["CM"]

        for cm_name in base_cms:
            self.register_deriv_funcs(self.d_delay_d_cmparam, cm_name)
            self.register_cm_deriv_funcs(self.d_cm_d_CMs, cm_name)

    def validate(self):
        """Validate the CM parameters input."""
        super().validate()
        # If CM1 is set, we need CMEPOCH
        if (
            self.CM1.value is not None
            and self.CM1.value != 0.0
            and self.CMEPOCH.value is None
        ):
            if self._parent.PEPOCH.value is not None:
                self.CMEPOCH.value = self._parent.PEPOCH.value
            else:
                raise MissingParameter(
                    "Chromatic",
                    "CMEPOCH",
                    "CMEPOCH or PEPOCH is required if CM1 or higher are set",
                )

    def CM_derivative_unit(self, n):
        return f"pc cm^-3 MHz^-2 / yr^{n:d}" if n else "pc cm^-3 MHz^-2"

    def CM_derivative_description(self, n):
        return f"{n:d}'th time derivative of the chromatic measure"

    def get_CM_terms(self):
        """Return a list of CM term values in the model: [CM, CM1, ..., CMn]"""
        return [self.CM.quantity] + self._parent.get_prefix_list("CM", start_index=1)

    def base_cm(self, toas):
        cm = np.zeros(len(toas))
        cm_terms = self.get_CM_terms()
        if any(cmi.value != 0 for cmi in cm_terms[1:]):
            CMEPOCH = self.CMEPOCH.value
            if CMEPOCH is None:
                # Should be ruled out by validate()
                raise ValueError(
                    f"CMEPOCH not set but some derivatives are not zero: {cm_terms}"
                )
            else:
                dt = (toas["tdbld"] - CMEPOCH) * u.day
            dt_value = dt.to_value(u.yr)
        else:
            dt_value = np.zeros(len(toas), dtype=np.longdouble)
        cm_terms_value = [c.value for c in cm_terms]
        cm = taylor_horner(dt_value, cm_terms_value)
        return cm * cmu

    def alpha_value(self, toas):
        return np.ones(len(toas)) * self.CMIDX.quantity

    def constant_chromatic_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.chromatic_type_delay(toas)

    def print_par(self, format="pint"):
        prefix_cm = list(self.get_prefix_mapping_component("CM").values())
        cms = ["CM"] + prefix_cm
        result = "".join(getattr(self, cm).as_parfile_line(format=format) for cm in cms)
        if hasattr(self, "components"):
            all_params = self.components["ChromaticCM"].params
        else:
            all_params = self.params
        for pm in all_params:
            if pm not in cms:
                result += getattr(self, pm).as_parfile_line(format=format)
        return result

    def d_cm_d_CMs(self, toas, param_name, acc_delay=None):
        """Derivatives of CM wrt the CM taylor expansion coefficients."""
        par = getattr(self, param_name)
        if param_name == "CM":
            order = 0
        else:
            pn, idxf, idxv = split_prefixed_name(param_name)
            order = idxv
        cms = self.get_CM_terms()
        cm_terms = np.longdouble(np.zeros(len(cms)))
        cm_terms[order] = np.longdouble(1.0)
        if self.CMEPOCH.value is None:
            if any(t.value != 0 for t in cms[1:]):
                # Should be ruled out by validate()
                raise ValueError(f"CMEPOCH is not set but {param_name} is not zero")
            CMEPOCH = 0
        else:
            CMEPOCH = self.CMEPOCH.value
        dt = (toas["tdbld"] - CMEPOCH) * u.day
        dt_value = (dt.to(u.yr)).value
        return taylor_horner(dt_value, cm_terms) * (cmu / par.units)

    def change_cmepoch(self, new_epoch):
        """Change CMEPOCH to a new value and update CM accordingly.

        Parameters
        ----------
        new_epoch: float MJD (in TDB) or `astropy.Time` object
            The new CMEPOCH value.
        """
        if isinstance(new_epoch, Time):
            new_epoch = Time(new_epoch, scale="tdb", precision=9)
        else:
            new_epoch = Time(new_epoch, scale="tdb", format="mjd", precision=9)

        cmterms = [0.0 * u.Unit("")] + self.get_CM_terms()
        if self.CMEPOCH.value is None:
            if any(d.value != 0 for d in cmterms[2:]):
                # Should be ruled out by validate()
                raise ValueError(
                    f"CMEPOCH not set but some CM derivatives are not zero: {cmterms}"
                )
            self.CMEPOCH.value = new_epoch

        cmepoch_ld = self.CMEPOCH.quantity.tdb.mjd_long
        dt = (new_epoch.tdb.mjd_long - cmepoch_ld) * u.day

        for n in range(len(cmterms) - 1):
            cur_deriv = self.CM if n == 0 else getattr(self, f"CM{n}")
            cur_deriv.value = taylor_horner_deriv(
                dt.to(u.yr), cmterms, deriv_order=n + 1
            )
        self.CMEPOCH.value = new_epoch


class ChromaticCMX(Chromatic):
    """This class provides a CMX model - piecewise-constant chromatic variations with constant
    chromatic index.

    This model lets the user specify time ranges and fit for a different CMX value in each time range.

    It should be used in combination with the `ChromaticCM` model. Specifically, TNCHROMIDX must be
    set.

    Parameters supported:

    .. paramtable::
        :class: pint.models.chromatic_model.ChromaticCMX
    """

    register = True
    category = "chromatic_cmx"

    def __init__(self):
        super().__init__()

        self.add_CMX_range(None, None, cmx=0, frozen=False, index=1)

        self.cm_value_funcs += [self.cmx_cm]
        self.set_special_params(["CMX_0001", "CMXR1_0001", "CMXR2_0001"])
        self.delay_funcs_component += [self.CMX_chromatic_delay]

    def add_CMX_range(self, mjd_start, mjd_end, index=None, cmx=0, frozen=True):
        """Add CMX range to a chromatic model with specified start/end MJDs and CMX value.

        Parameters
        ----------

        mjd_start : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for beginning of CMX event.
        mjd_end : float or astropy.quantity.Quantity or astropy.time.Time
            MJD for end of CMX event.
        index : int, None
            Integer label for CMX event. If None, will increment largest used index by 1.
        cmx : float or astropy.quantity.Quantity
            Change in CM during CMX event.
        frozen : bool
            Indicates whether CMX will be fit.

        Returns
        -------

        index : int
            Index that has been assigned to new CMX event.
        """

        #### Setting up the CMX title convention. If index is None, want to increment the current max CMX index by 1.
        if index is None:
            dct = self.get_prefix_mapping_component("CMX_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if mjd_end is not None and mjd_start is not None:
            if mjd_end < mjd_start:
                raise ValueError("Starting MJD is greater than ending MJD.")
        elif mjd_start != mjd_end:
            raise ValueError("Only one MJD bound is set.")

        if int(index) in self.get_prefix_mapping_component("CMX_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another."
            )

        if isinstance(cmx, u.quantity.Quantity):
            cmx = cmx.to_value(cmu)

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
                name=f"CMX_{i}",
                units=cmu,
                value=cmx,
                description="Dispersion measure variation",
                parameter_type="float",
                frozen=frozen,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"CMXR1_{i}",
                units="MJD",
                description="Beginning of CMX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_start,
                convert_tcb2tdb=False,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"CMXR2_{i}",
                units="MJD",
                description="End of CMX interval",
                parameter_type="MJD",
                time_scale="utc",
                value=mjd_end,
                convert_tcb2tdb=False,
            )
        )
        self.setup()
        self.validate()
        return index

    def add_CMX_ranges(self, mjd_starts, mjd_ends, indices=None, cmxs=0, frozens=True):
        """Add CMX ranges to a dispersion model with specified start/end MJDs and CMXs.

        Parameters
        ----------

        mjd_starts : iterable of float or astropy.quantity.Quantity or astropy.time.Time
            MJD for beginning of CMX event.
        mjd_end : iterable of float or astropy.quantity.Quantity or astropy.time.Time
            MJD for end of CMX event.
        indices : iterable of int, None
            Integer label for CMX event. If None, will increment largest used index by 1.
        cmxs : iterable of float or astropy.quantity.Quantity, or float or astropy.quantity.Quantity
            Change in CM during CMX event.
        frozens : iterable of bool or bool
            Indicates whether CMX will be fit.

        Returns
        -------

        indices : list
            Indices that has been assigned to new CMX events
        """
        if len(mjd_starts) != len(mjd_ends):
            raise ValueError(
                f"Number of mjd_start values {len(mjd_starts)} must match number of mjd_end values {len(mjd_ends)}"
            )
        if indices is None:
            indices = [None] * len(mjd_starts)
        cmxs = np.atleast_1d(cmxs)
        if len(cmxs) == 1:
            cmxs = np.repeat(cmxs, len(mjd_starts))
        if len(cmxs) != len(mjd_starts):
            raise ValueError(
                f"Number of mjd_start values {len(mjd_starts)} must match number of cmx values {len(cmxs)}"
            )
        frozens = np.atleast_1d(frozens)
        if len(frozens) == 1:
            frozens = np.repeat(frozens, len(mjd_starts))
        if len(frozens) != len(mjd_starts):
            raise ValueError(
                f"Number of mjd_start values {len(mjd_starts)} must match number of frozen values {len(frozens)}"
            )

        #### Setting up the CMX title convention. If index is None, want to increment the current max CMX index by 1.
        dct = self.get_prefix_mapping_component("CMX_")
        last_index = np.max(list(dct.keys()))
        added_indices = []
        for mjd_start, mjd_end, index, cmx, frozen in zip(
            mjd_starts, mjd_ends, indices, cmxs, frozens
        ):
            if index is None:
                index = last_index + 1
                last_index += 1
            elif index in list(dct.keys()):
                raise ValueError(
                    f"Attempting to insert CMX_{index:04d} but it already exists"
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
            if isinstance(cmx, u.quantity.Quantity):
                cmx = cmx.to_value(u.pc / u.cm**3)
            if isinstance(mjd_start, Time):
                mjd_start = mjd_start.mjd
            elif isinstance(mjd_start, u.quantity.Quantity):
                mjd_start = mjd_start.value
            if isinstance(mjd_end, Time):
                mjd_end = mjd_end.mjd
            elif isinstance(mjd_end, u.quantity.Quantity):
                mjd_end = mjd_end.value
            log.trace(f"Adding CMX_{i} from MJD {mjd_start} to MJD {mjd_end}")
            self.add_param(
                prefixParameter(
                    name=f"CMX_{i}",
                    units=cmu,
                    value=cmx,
                    description="Dispersion measure variation",
                    parameter_type="float",
                    frozen=frozen,
                    convert_tcb2tdb=False,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"CMXR1_{i}",
                    units="MJD",
                    description="Beginning of CMX interval",
                    parameter_type="MJD",
                    time_scale="utc",
                    value=mjd_start,
                    convert_tcb2tdb=False,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"CMXR2_{i}",
                    units="MJD",
                    description="End of CMX interval",
                    parameter_type="MJD",
                    time_scale="utc",
                    value=mjd_end,
                    convert_tcb2tdb=False,
                )
            )
        self.setup()
        self.validate()
        return added_indices

    def remove_CMX_range(self, index):
        """Removes all CMX parameters associated with a given index/list of indices.

        Parameters
        ----------

        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to CMX indices to be removed from model.
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
            for prefix in ["CMX_", "CMXR1_", "CMXR2_"]:
                self.remove_param(prefix + index_rf)
        self.validate()

    def get_indices(self):
        """Returns an array of integers corresponding to CMX parameters.

        Returns
        -------
        inds : np.ndarray
            Array of CMX indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "CMX_" in p]
        return np.array(inds)

    def setup(self):
        super().setup()
        # Get CMX mapping.
        # Register the CMX derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("CMX_"):
                self.register_deriv_funcs(self.d_delay_d_cmparam, prefix_par)
                self.register_cm_deriv_funcs(self.d_cm_d_CMX, prefix_par)

    def validate(self):
        """Validate the CMX parameters."""
        super().validate()
        CMX_mapping = self.get_prefix_mapping_component("CMX_")
        CMXR1_mapping = self.get_prefix_mapping_component("CMXR1_")
        CMXR2_mapping = self.get_prefix_mapping_component("CMXR2_")
        if CMX_mapping.keys() != CMXR1_mapping.keys():
            # FIXME: report mismatch
            raise ValueError(
                "CMX_ parameters do not "
                "match CMXR1_ parameters. "
                "Please check your prefixed parameters."
            )
        if CMX_mapping.keys() != CMXR2_mapping.keys():
            raise ValueError(
                "CMX_ parameters do not "
                "match CMXR2_ parameters. "
                "Please check your prefixed parameters."
            )
        r1 = np.zeros(len(CMX_mapping))
        r2 = np.zeros(len(CMX_mapping))
        indices = np.zeros(len(CMX_mapping), dtype=np.int32)
        for j, index in enumerate(CMX_mapping):
            if (
                getattr(self, f"CMXR1_{index:04d}").quantity is not None
                and getattr(self, f"CMXR2_{index:04d}").quantity is not None
            ):
                r1[j] = getattr(self, f"CMXR1_{index:04d}").quantity.mjd
                r2[j] = getattr(self, f"CMXR2_{index:04d}").quantity.mjd
                indices[j] = index
        for j, index in enumerate(CMXR1_mapping):
            if np.any((r1[j] > r1) & (r1[j] < r2)):
                k = np.where((r1[j] > r1) & (r1[j] < r2))[0]
                for kk in k.flatten():
                    log.warning(
                        f"Start of CMX_{index:04d} ({r1[j]}-{r2[j]}) overlaps with CMX_{indices[kk]:04d} ({r1[kk]}-{r2[kk]})"
                    )
            if np.any((r2[j] > r1) & (r2[j] < r2)):
                k = np.where((r2[j] > r1) & (r2[j] < r2))[0]
                for kk in k.flatten():
                    log.warning(
                        f"End of CMX_{index:04d} ({r1[j]}-{r2[j]}) overlaps with CMX_{indices[kk]:04d} ({r1[kk]}-{r2[kk]})"
                    )

    def validate_toas(self, toas):
        CMX_mapping = self.get_prefix_mapping_component("CMX_")
        CMXR1_mapping = self.get_prefix_mapping_component("CMXR1_")
        CMXR2_mapping = self.get_prefix_mapping_component("CMXR2_")
        bad_parameters = []
        for k in CMXR1_mapping.keys():
            if self._parent[CMX_mapping[k]].frozen:
                continue
            b = self._parent[CMXR1_mapping[k]].quantity.mjd * u.d
            e = self._parent[CMXR2_mapping[k]].quantity.mjd * u.d
            mjds = toas.get_mjds()
            n = np.sum((b <= mjds) & (mjds < e))
            if n == 0:
                bad_parameters.append(CMX_mapping[k])
        if bad_parameters:
            raise MissingTOAs(bad_parameters)

    def get_select_idxs(self, toas: TOAs):
        condition = {}
        tbl = toas.table
        if not hasattr(self, "cmx_toas_selector"):
            self.cmx_toas_selector = TOASelect(is_range=True)
        CMX_mapping = self.get_prefix_mapping_component("CMX_")
        CMXR1_mapping = self.get_prefix_mapping_component("CMXR1_")
        CMXR2_mapping = self.get_prefix_mapping_component("CMXR2_")
        for epoch_ind in CMX_mapping.keys():
            r1 = getattr(self, CMXR1_mapping[epoch_ind]).quantity
            r2 = getattr(self, CMXR2_mapping[epoch_ind]).quantity
            condition[CMX_mapping[epoch_ind]] = (r1.mjd, r2.mjd)
        return self.cmx_toas_selector.get_select_index(condition, tbl["mjd_float"])

    def cmx_cm(self, toas: TOAs):
        if (
            self._parent is not None
            and self._parent.toas_for_cache is toas
            and self._parent.piecewise_cache is not None
            and "ChromaticCMX" in self._parent.piecewise_cache
        ):
            select_idx = self._parent.piecewise_cache["ChromaticCMX"]
        else:
            select_idx = self.get_select_idxs(toas)

        # Get CMX delays
        cm = np.zeros(len(toas)) << self._parent.CM.units
        for k, v in select_idx.items():
            cm[v] += getattr(self, k).quantity
        return cm

    def CMX_chromatic_delay(self, toas, acc_delay=None):
        """This is a wrapper function for interacting with the TimingModel class"""
        return self.chromatic_type_delay(toas)

    def d_cm_d_CMX(self, toas, param_name, acc_delay=None):
        condition = {}
        tbl = toas.table
        if not hasattr(self, "cmx_toas_selector"):
            self.cmx_toas_selector = TOASelect(is_range=True)
        param = getattr(self, param_name)
        cmx_index = param.index
        CMXR1_mapping = self.get_prefix_mapping_component("CMXR1_")
        CMXR2_mapping = self.get_prefix_mapping_component("CMXR2_")
        r1 = getattr(self, CMXR1_mapping[cmx_index]).quantity
        r2 = getattr(self, CMXR2_mapping[cmx_index]).quantity
        condition = {param_name: (r1.mjd, r2.mjd)}
        select_idx = self.cmx_toas_selector.get_select_index(
            condition, tbl["mjd_float"]
        )

        cmx = np.zeros(len(tbl))
        for k, v in select_idx.items():
            cmx[v] = 1.0
        return cmx * (u.pc / u.cm**3) / (u.pc / u.cm**3)

    def print_par(self, format="pint"):
        result = ""
        CMX_mapping = self.get_prefix_mapping_component("CMX_")
        CMXR1_mapping = self.get_prefix_mapping_component("CMXR1_")
        CMXR2_mapping = self.get_prefix_mapping_component("CMXR2_")
        sorted_list = sorted(CMX_mapping.keys())
        for ii in sorted_list:
            result += getattr(self, CMX_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, CMXR1_mapping[ii]).as_parfile_line(format=format)
            result += getattr(self, CMXR2_mapping[ii]).as_parfile_line(format=format)
        return result
