from astropy.table import Table
from warnings import warn
import numpy as np
import astropy.units as u
from pint.models.timing_model import DelayComponent, MissingParameter
from pint.models.parameter import floatParameter, prefixParameter, MJDParameter
from pint.utils import split_prefixed_name, taylor_horner, taylor_horner_deriv
from pint import DMconst
from astropy.time import Time

cmu = u.pc / u.cm**3 / u.MHz**2


class Chromatic(DelayComponent):
    """A base chromatic timing model."""

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
    """Simple chromatic delay model.

    This model uses Taylor expansion to model CM variation over time. It
    can also be used for a constant CM.

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
        if any(t.value != 0 for t in cm_terms[1:]):
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
    """This class provides a CMX model - piecewise-constant chromatic variations.

    This model lets the user specify time ranges and fit for a different
    CMX value in each time range.

    Parameters supported:

    .. paramtable::
        :class: pint.models.dispersion_model.DispersionDMX
    """

    register = True
    category = "dispersion_dmx"

    def __init__(self):
        super().__init__()

        # DMX is for info output right now
        # @abhisrkckl: What exactly is the use of this parameter?
        self.add_param(
            floatParameter(
                name="DMX",
                units="pc cm^-3",
                value=0.0,
                description="Dispersion measure",
                convert_tcb2tdb=False,
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
                tcb2tdb_scale_factor=DMconst,
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
                tcb2tdb_scale_factor=u.Quantity(1),
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
                tcb2tdb_scale_factor=u.Quantity(1),
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
                    tcb2tdb_scale_factor=DMconst,
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
                    tcb2tdb_scale_factor=u.Quantity(1),
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
                    tcb2tdb_scale_factor=u.Quantity(1),
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
