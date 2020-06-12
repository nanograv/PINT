""" Defining Model Sector, classes for grouping the same type of model components and provide
the uniformed API 
"""
import numpy as np
import astropy.units as u
from astropy import time
import copy
from collections import defaultdict

from pint.utils import get_component_type
from pint.phase import Phase


class ModelSector(object):
    """ A class that groups the same type of component and provide the API for
    gathering the information from each component.

    Parameter
    ---------
    components: list of `Component` sub-class object.
        Components that are belong to the same type, for example, DelayComponent.

    Note
    ----
    The order of the component in the list is the order a component get computed.
    """

    _methods = tuple()

    def __init__(self, components):
        if not hasattr(self, "sector_name"):
            raise TypeError(
                "Please use ModelSector's subclass to "
                "initialize for a specific model sector."
            )
        # If only one component is given, convert it to a list
        if not isinstance(components, (list, tuple)):
            components = [
                components,
            ]
        # Check if the components are the same type
        for cp in components:
            cp_type = get_component_type(cp)
            if cp_type != self.sector_name:
                raise ValueError(
                    "Component {} is not a {} of"
                    " component.".format(cp.__class__.__name__, self.sector_name)
                )
        self.component_list = components
        self._parent = None

    def __getattr__(self, name):
        try:
            return super(ModelSector, self).__getattribute__(name)
        except AttributeError:
            try:
                p = super(ModelSector, self).__getattribute__("_parent")
                if p is None:
                    raise AttributeError(
                        "'%s' object has no attribute '%s'."
                        % (self.__class__.__name__, name)
                    )
                else:
                    return self._parent.__getattr__(name)
            except:
                raise AttributeError(
                    "'%s' object has no attribute '%s'."
                    % (self.__class__.__name__, name)
                )

    @property
    def component_names(self):
        return [cp.__class__.__name__ for cp in self.component_list]

    @property
    def component_classes(self):
        return [cp.__class__ for cp in self.component_list]

    def get_quantity_funcs(self, func_list_name):
        """List of all model sector functions that contribute to the final
        modeled quantity.
        """
        fs = []
        for cp in self.component_list:
            fs += getattr(cp, func_list_name)
        return fs

    def get_deriv_funcs(self, deriv_dict_name):
        """Return dictionary of derivative functions."""
        deriv_funcs = defaultdict(list)
        for cp in self.component_list:
            for k, v in getattr(cp, deriv_dict_name).items():
                deriv_funcs[k] += v
        return dict(deriv_funcs)


class DelaySector(ModelSector):
    """ Class for holding all delay components and their APIs
    """

    _methods = (
        "delay_components",
        "delay",
        "delay_funcs",
        "get_barycentric_toas",
        "d_delay_d_param",
        "delay_deriv_funcs",
        "d_delay_d_param_num",
        "delay",
    )

    def __init__(self, delay_components, sector_map={}):
        self.sector_name = "DelayComponent"
        super(DelaySector, self).__init__(delay_components)

    @property
    def delay_components(self):
        return self.component_list

    @property
    def delay_funcs(self):
        return self.get_quantity_funcs("delay_funcs_component")

    @property
    def delay_deriv_funcs(self):
        """List of derivative functions for delay components."""
        return self.get_deriv_funcs("deriv_funcs")

    def delay(self, toas, cutoff_component="", include_last=True):
        """Total delay for the TOAs.

        Parameters
        ----------
        toas: toa.table
            The toas for analysis delays.
        cutoff_component: str
            The delay component name that a user wants the calculation to stop
            at.
        include_last: bool
            If the cutoff delay component is included.

        Return the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.

        """
        delay = np.zeros(toas.ntoas) * u.second
        if cutoff_component == "":
            idx = len(self.component_list)
        else:
            delay_names = self.component_names
            if cutoff_component in delay_names:
                idx = delay_names.index(cutoff_component)
                if include_last:
                    idx += 1
            else:
                raise KeyError("No delay component named '%s'." % cutoff_component)

        # Do NOT cycle through delay_funcs - cycle through components until cutoff
        for dc in self.component_list[:idx]:
            for df in dc.delay_funcs_component:
                delay += df(toas, delay)
        return delay

    def get_barycentric_toas(self, toas, cutoff_component=""):
        """Conveniently calculate the barycentric TOAs.

       Parameters
       ----------
       toas: TOAs object
           The TOAs the barycentric corrections are applied on
       cutoff_delay: str, optional
           The cutoff delay component name. If it is not provided, it will
           search for binary delay and apply all the delay before binary.

       Return
       ------
       astropy.quantity.
           Barycentered TOAs.

        """
        tbl = toas.table
        if cutoff_component == "":
            delay_list = self.component_list
            for cp in delay_list:
                if cp.category == "pulsar_system":
                    cutoff_component = cp.__class__.__name__
        corr = self.delay(toas, cutoff_component, False)
        return tbl["tdbld"] * u.day - corr

    def d_delay_d_param(self, toas, param, acc_delay=None):
        """Return the derivative of delay with respect to the parameter."""
        par = getattr(self, param)
        result = np.longdouble(np.zeros(toas.ntoas) * u.s / par.units)
        delay_derivs = self.delay_deriv_funcs
        if param not in list(delay_derivs.keys()):
            raise AttributeError(
                "Derivative function for '%s' is not provided"
                " or not registered. " % param
            )
        for df in delay_derivs[param]:
            result += df(toas, param, acc_delay).to(
                result.unit, equivalencies=u.dimensionless_angles()
            )
        return result

    def d_delay_d_param_num(self, toas, param, step=1e-2):
        """Return the derivative of delay with respect to the parameter.

        Compute the value numerically, using a symmetric finite difference.

        """
        # TODO : We need to know the range of parameter.
        par = getattr(self, param)
        ori_value = par.value
        if ori_value is None:
            # A parameter did not get to use in the model
            log.warning("Parameter '%s' is not used by timing model." % param)
            return np.zeros(toas.ntoas) * (u.second / par.units)
        unit = par.units
        if ori_value == 0:
            h = 1.0 * step
        else:
            h = ori_value * step
        parv = [par.value - h, par.value + h]
        delay = np.zeros((toas.ntoas, 2))
        for ii, val in enumerate(parv):
            par.value = val
            try:
                delay[:, ii] = self.delay(toas)
            except:
                par.value = ori_value
                raise
        d_delay = (-delay[:, 0] + delay[:, 1]) / 2.0 / h
        par.value = ori_value
        return d_delay * (u.second / unit)


class PhaseSector(ModelSector):
    """ Class for holding all phase components and their APIs.

    Parameters
    ----------
    """

    _methods = (
        "phase_components",
        "phase_funcs",
        "phase",
        "phase_deriv_funcs",
        "d_phase_d_param",
        "d_phase_d_param_num",
        "d_phase_d_toa",
        "d_phase_d_delay_funcs",
        "get_spin_frequency",
    )

    def __init__(self, phase_components, sector_map={}):
        self.sector_name = "PhaseComponent"
        super(PhaseSector, self).__init__(phase_components)

    @property
    def phase_components(self):
        return self.component_list

    @property
    def phase_funcs(self):
        """List of all phase functions."""
        return self.get_quantity_funcs("phase_funcs_component")

    @property
    def phase_deriv_funcs(self):
        """List of derivative functions for phase components."""
        return self.get_deriv_funcs("deriv_funcs")

    @property
    def d_phase_d_delay_funcs(self):
        """List of d_phase_d_delay functions."""
        Dphase_Ddelay = []
        for cp in self.component_list:
            Dphase_Ddelay += cp.phase_derivs_wrt_delay
        return Dphase_Ddelay

    def phase(self, toas, delay=None, abs_phase=False):
        """Return the model-predicted pulse phase for the given TOAs.

        Parameters
        ----------
        toas: `~pint.toa.TOAs` object.
            TOAs for evaluating the phase.
        delay: `numpy.ndarray`, optional.
            Input time delay values. If not given, phase will calculate the
            delay. Default is None,
        abs_phase: bool
            Flag for using the absolute phase. Default is False.

        Return
        ------
        `~pint.phase.Phase` object. The spin phase that at given TOAs.
        """
        # First compute the delays to "pulsar time"
        delay = self.delay(toas)
        phase = Phase(np.zeros(toas.ntoas), np.zeros(toas.ntoas))
        # Then compute the relevant pulse phases
        for pf in self.phase_funcs:
            phase += Phase(pf(toas, delay))

        # If the absolute phase flag is on, use the TZR parameters to compute
        # the absolute phase.
        if abs_phase:
            if "AbsPhase" not in list(self.components.keys()):
                # if no absolute phase (TZRMJD), add the component to the model and calculate it
                from pint.models import absolute_phase

                self.add_component(absolute_phase.AbsPhase())
                self.make_TZR_toa(
                    toas
                )  # TODO:needs timfile to get all toas, but model doesn't have access to timfile. different place for this?
            tz_toa = self.get_TZR_toa(toas)
            tz_delay = self.delay(tz_toa)
            tz_phase = Phase(np.zeros(len(toas.table)), np.zeros(len(toas.table)))
            for pf in self.phase_funcs:
                tz_phase += Phase(pf(tz_toa, tz_delay))
            return phase - tz_phase
        else:
            return phase

    def d_phase_d_param(self, toas, delay, param):
        """Return the derivative of phase with respect to the parameter."""
        # TODO need to do correct chain rule stuff wrt delay derivs, etc
        # Is it safe to assume that any param affecting delay only affects
        # phase indirectly (and vice-versa)??
        par = getattr(self, param)
        result = np.longdouble(np.zeros(toas.ntoas)) / par.units
        param_phase_derivs = []
        phase_derivs = self.phase_deriv_funcs
        if param in list(phase_derivs.keys()):
            for df in phase_derivs[param]:
                result += df(toas, param, delay).to(
                    result.unit, equivalencies=u.dimensionless_angles()
                )
        else:
            # Apply chain rule for the parameters in the delay.
            # total_phase = Phase1(delay(param)) + Phase2(delay(param))
            # d_total_phase_d_param = d_Phase1/d_delay*d_delay/d_param +
            #                         d_Phase2/d_delay*d_delay/d_param
            #                       = (d_Phase1/d_delay + d_Phase2/d_delay) *
            #                         d_delay_d_param
            d_delay_d_p = self.d_delay_d_param(toas, param)
            dpdd_result = np.longdouble(np.zeros(toas.ntoas)) / u.second
            for dpddf in self.d_phase_d_delay_funcs:
                dpdd_result += dpddf(toas, delay)
            result = dpdd_result * d_delay_d_p
        return result.to(result.unit, equivalencies=u.dimensionless_angles())

    def d_phase_d_param_num(self, toas, param, step=1e-2):
        """Return the derivative of phase with respect to the parameter.

        Compute the value numerically, using a symmetric finite difference.

        """
        # TODO : We need to know the range of parameter.
        par = getattr(self, param)
        ori_value = par.value
        unit = par.units
        if ori_value == 0:
            h = 1.0 * step
        else:
            h = ori_value * step
        parv = [par.value - h, par.value + h]

        phase_i = (
            np.zeros((toas.ntoas, 2), dtype=np.longdouble) * u.dimensionless_unscaled
        )
        phase_f = (
            np.zeros((toas.ntoas, 2), dtype=np.longdouble) * u.dimensionless_unscaled
        )
        for ii, val in enumerate(parv):
            par.value = val
            ph = self.phase(toas)
            phase_i[:, ii] = ph.int
            phase_f[:, ii] = ph.frac
        res_i = -phase_i[:, 0] + phase_i[:, 1]
        res_f = -phase_f[:, 0] + phase_f[:, 1]
        result = (res_i + res_f) / (2.0 * h * unit)
        # shift value back to the original value
        par.quantity = ori_value
        return result

    def d_phase_d_toa(self, toas, sample_step=None):
        """Return the derivative of phase wrt TOA.

        Parameters
        ----------
        toas : PINT TOAs class
            The toas when the derivative of phase will be evaluated at.
        sample_step : float optional
            Finite difference steps. If not specified, it will take 1/10 of the
            spin period.

        """
        copy_toas = copy.deepcopy(toas)
        if sample_step is None:
            pulse_period = 1.0 / (self.F0.quantity)
            sample_step = pulse_period * 1000
        sample_dt = [-sample_step, 2 * sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = [dt.value] * copy_toas.ntoas * dt._unit
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT)
            phase = self.phase(copy_toas)
            sample_phase.append(phase)
        # Use finite difference method.
        # phase'(t) = (phase(t+h)-phase(t-h))/2+ 1/6*F2*h^2 + ..
        # The error should be near 1/6*F2*h^2
        dp = sample_phase[1] - sample_phase[0]
        d_phase_d_toa = dp.int / (2 * sample_step) + dp.frac / (2 * sample_step)
        del copy_toas
        return d_phase_d_toa.to(u.Hz)

    def get_spin_frequency(self, toas=None, apparent_frequency=False):
        pass


class NoiseSector(ModelSector):
    """ Class for holding all delay components and their APIs
    """

    _methods = (
        "covariance_matrix_funcs",
        "scaled_sigma_funcs",
        "basis_funcs",
        "scaled_sigma",
        "noise_model_designmatrix",
        "noise_model_basis_weight",
        "noise_model_dimensions",
    )

    def __init__(self, noise_components):
        self.sector_name = "NoiseComponent"
        super(NoiseSector, self).__init__(noise_components)

    @property
    def covariance_matrix_funcs(self,):
        """List of covariance matrix functions."""
        return self.get_quantity_funcs("covariance_matrix_funcs")

    @property
    def scaled_sigma_funcs(self,):
        """List of scaled uncertainty functions."""
        return self.get_quantity_funcs("scaled_sigma_funcs")

    @property
    def basis_funcs(self,):
        """List of scaled uncertainty functions."""
        return self.get_quantity_funcs("basis_funcs")

    def noise_model_designmatrix(self, toas):
        result = []
        if len(self.basis_funcs) == 0:
            return None

        for nf in self.basis_funcs:
            result.append(nf(toas)[0])
        return np.hstack([r for r in result])

    def noise_model_basis_weight(self, toas):
        result = []
        if len(self.basis_funcs) == 0:
            return None

        for nf in self.basis_funcs:
            result.append(nf(toas)[1])
        return np.hstack([r for r in result])

    def noise_model_dimensions(self, toas):
        """Returns a dictionary of correlated-noise components in the noise
        model.  Each entry contains a tuple (offset, size) where size is the
        number of basis funtions for the component, and offset is their
        starting location in the design matrix and weights vector."""
        result = {}

        # Correct results rely on this ordering being the
        # same as what is done in the self.basis_funcs
        # property.
        if len(self.basis_funcs) > 0:
            ntot = 0
            for nc in self.component_list:
                bfs = nc.basis_funcs
                if len(bfs) == 0:
                    continue
                nbf = 0
                for bf in bfs:
                    nbf += len(bf(toas)[1])
                result[nc.category] = (ntot, nbf)
                ntot += nbf

        return result


builtin_sector_map = {
    "DelayComponent": DelaySector,
    "PhaseComponent": PhaseSector,
    "NoiseComponent": NoiseSector,
}
