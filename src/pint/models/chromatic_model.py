from astropy.table import Table
from warnings import warn
import numpy as np
import astropy.units as u
from pint.models.timing_model import DelayComponent
from pint.models.parameter import floatParameter, prefixParameter, MJDParameter
from pint.utils import split_prefixed_name, taylor_horner, taylor_horner_deriv
from pint import DMconst
from pint.exceptions import MissingParameter
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
