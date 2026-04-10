"""Transient chromatic events due to, e.g., profile change events in J1713+0747 or extreme scattering events (ESE)"""

from typing import List, Optional, Union
import numpy as np
import astropy.units as u
from astropy.time import Time
from pint.models.parameter import floatParameter, prefixParameter
from pint.models.timing_model import DelayComponent
from pint.toa import TOAs


class SimpleExponentialDip(DelayComponent):
    """Simple chromatic exponential dip model for events like the profile changes
    in J1713+0747.

    The dip is modeled as a logistic function multiplied by an exponential decay.
    This is different from the tempo2 implementation where the exponential decay
    is multiplied by a Heaviside step function. We cannot fit for the event epoch
    while using the latter because it is not differentiable at the event epoch.
    This implementation approaches the tempo2 version in the limit EXPDIPEPS -> 0.
    The parameter names are also different between this implementation and tempo2.
    The tempo2 parameter names have been added here as aliases so that tempo2 par files
    can be read.

    The explicit mathematical form of the dips is as follows.

    .. math::
            \\Delta_{\\text{dip}}(t)=-\\sum_{i}A_{i}\\left(\\frac{f}{f_{\\text{ref}}}\\right)^{\\gamma_{i}}\\left(\\frac{\\tau_{i}}{\\epsilon}\\right)^{\\epsilon/\\tau_{i}}\\left(\\frac{\\tau_{i}}{\\tau_{i}-\\epsilon}\\right)^{\\frac{\\tau_{i}-\\epsilon}{\\tau_{i}}}\\frac{\\exp\\left[-\\frac{t-T_{i}}{\\tau_{i}}\\right]}{1+\\exp\\left[-\\frac{t-T_{i}}{\\epsilon}\\right]}

    An exponential dip event is normalized such that the EXPDIPAMP_ is its extremum
    value. Note that the extremum occurs slightly after the event epoch.

    Parameters supported:

    .. paramtable::
        :class: pint.models.transient_events.SimpleExponentialDip
    """

    register = True
    category = "simple_exp_dip"

    def __init__(self):
        super().__init__()

        self.add_param(
            floatParameter(
                name=f"EXPDIPEPS",
                units="day",
                description="Chromatic exponential dip step timescale",
                value=1e-3,
                frozen=True,
                tcb2tdb_scale_factor=1,
            )
        )

        self.add_param(
            floatParameter(
                name=f"EXPDIPFREF",
                units="MHz",
                description="Chromatic exponential dip reference frequency",
                value=1400,
                frozen=True,
                tcb2tdb_scale_factor=1,
            )
        )

        self.add_exp_dip(None, 0, None, None, index=1, frozen=True)

        self.delay_funcs_component += [self.expdip_delay]

    def add_exp_dip(
        self,
        epoch: Union[float, u.Quantity, Time],
        ampl: Union[float, u.Quantity],
        gamma: Union[float, u.Quantity],
        tau: Union[float, u.Quantity],
        index: Optional[int] = None,
        frozen: bool = True,
    ) -> int:
        """Add an exponential dip to the model."""

        if index is None:
            dct = self.get_prefix_mapping_component("EXPEP_")
            index = np.max(list(dct.keys())) + 1
        elif int(index) in self.get_prefix_mapping_component("EXPEP_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another."
            )

        if isinstance(epoch, Time):
            epoch = epoch.mjd
        elif isinstance(epoch, u.Quantity):
            epoch = epoch.value

        self.add_param(
            prefixParameter(
                name=f"EXPDIPEP_{index}",
                units="MJD",
                description="Chromatic exponential dip epoch",
                parameter_type="MJD",
                time_scale="utc",
                value=epoch,
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["EXPEP_"],
            )
        )

        if isinstance(ampl, u.Quantity):
            ampl = ampl.to_value(u.s)

        self.add_param(
            prefixParameter(
                name=f"EXPDIPAMP_{index}",
                units="s",
                value=ampl,
                description="Chromatic exponential dip amplitude",
                parameter_type="float",
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["EXPPH_"],
            )
        )

        if isinstance(gamma, u.Quantity):
            gamma = gamma.to_value(u.s)

        self.add_param(
            prefixParameter(
                name=f"EXPDIPIDX_{index}",
                units="",
                value=gamma,
                description="Chromatic exponential dip index",
                parameter_type="float",
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["EXPINDEX_"],
            )
        )

        if isinstance(tau, u.Quantity):
            tau = tau.to_value(u.s)

        self.add_param(
            prefixParameter(
                name=f"EXPDIPTAU_{index}",
                units="day",
                value=tau,
                description="Chromatic exponential dip decay timescale",
                parameter_type="float",
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["EXPTAU_"],
            )
        )

        self.setup()
        self.validate()

        return index

    def remove_exp_dip(self, index: Union[float, int, List[int], np.ndarray]) -> None:
        """Removes all exp dip parameters associated with a given index/list of indices.

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
            index_rf = f"{int(index):d}"
            for prefix in ["EXPDIPEP_", "EXPDIPAMP_", "EXPDIPTAU_", "EXPDIPIDX_"]:
                self.remove_param(f"{prefix}{index_rf}")
        self.validate()

    def get_indices(self) -> np.ndarray:
        """Returns an array of integers corresponding to exp dip parameters.

        Returns
        -------
        inds : np.ndarray
            Array of exp dip indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "EXPDIPEP_" in p]
        return np.array(inds)

    def setup(self) -> None:
        super().setup()
        # Get DMX mapping.
        # Register the DMX derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("EXPDIPEP_"):
                self.register_deriv_funcs(self.d_delay_d_T, prefix_par)
            elif prefix_par.startswith("EXPDIPAMP_"):
                self.register_deriv_funcs(self.d_delay_d_A, prefix_par)
            elif prefix_par.startswith("EXPDIPTAU_"):
                self.register_deriv_funcs(self.d_delay_d_tau, prefix_par)
            elif prefix_par.startswith("EXPDIPIDX_"):
                self.register_deriv_funcs(self.d_delay_d_gamma, prefix_par)

    def get_ffac(self, toas: TOAs) -> np.ndarray:
        """Compute f/fref where f is the observing frequency."""
        f = self._parent.barycentric_radio_freq(toas)
        fref = self.EXPDIPFREF.quantity
        return (f / fref).to_value(u.dimensionless_unscaled)

    def expdip_delay_term(
        self, t_mjd: np.ndarray, ffac: np.ndarray, ii: int
    ) -> u.Quantity:
        """Compute the delay for a single exponential dip event."""
        T = getattr(self, f"EXPDIPEP_{ii}").value
        dt = (t_mjd - T) * u.day

        A = getattr(self, f"EXPDIPAMP_{ii}").quantity
        gamma = getattr(self, f"EXPDIPIDX_{ii}").quantity
        tau = getattr(self, f"EXPDIPTAU_{ii}").quantity
        eps = self.EXPDIPEPS.quantity

        # Done this way to avoid overflow in exp
        expfac = np.zeros(len(dt))
        expfac[dt >= 0] = np.exp(-dt[dt >= 0] / tau) / (1 + np.exp(-dt[dt >= 0] / eps))
        expfac[dt < 0] = np.exp(dt[dt < 0] * (tau - eps) / (tau * eps)) / (
            1 + np.exp(dt[dt < 0] / eps)
        )

        return (
            -A
            * ffac**gamma
            * (tau / eps) ** (eps / tau)
            * (tau / (tau - eps)) ** ((tau - eps) / tau)
            * expfac
        )

    def expdip_delay(self, toas: TOAs, acc_delay=None) -> u.Quantity:
        """Total exponential dip delay."""
        indices = self.get_indices()

        ffac = self.get_ffac(toas)
        t_mjd = toas["tdbld"]

        delay = np.zeros(len(toas)) * u.s
        for ii in indices:
            delay += self.expdip_delay_term(t_mjd, ffac, ii)

        return delay

    def d_delay_d_A(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. exponential dip amplitude."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)
        A = getattr(self, f"EXPDIPAMP_{ii}").quantity
        return self.expdip_delay_term(toas["tdbld"], ffac, ii) / A

    def d_delay_d_gamma(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. exponential dip chromatic index."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)
        return self.expdip_delay_term(toas["tdbld"], ffac, ii) * np.log(ffac)

    def d_delay_d_tau(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. exponential dip decay timescale."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)

        t0_mjd = getattr(self, f"EXPDIPEP_{ii}").value
        dt = (toas["tdbld"] - t0_mjd) * u.day

        tau = getattr(self, f"EXPDIPTAU_{ii}").quantity
        eps = self.EXPDIPEPS.quantity

        return (
            self.expdip_delay_term(toas["tdbld"], ffac, ii)
            * (dt + eps * np.log(eps / (tau - eps)))
            / tau**2
        )

    def d_delay_d_T(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. exponential dip epoch."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)

        T = getattr(self, f"EXPDIPEP_{ii}").value
        dt = (toas["tdbld"] - T) * u.day

        tau = getattr(self, f"EXPDIPTAU_{ii}").quantity
        eps = self.EXPDIPEPS.quantity

        # Done this way to avoid overflow in exp
        expfac1 = np.zeros(len(dt))
        expfac1[dt >= 0] = np.exp(-dt[dt >= 0] / eps) / (1 + np.exp(-dt[dt >= 0] / eps))
        expfac1[dt < 0] = 1 / (1 + np.exp(dt[dt < 0] / eps))

        return self.expdip_delay_term(toas["tdbld"], ffac, ii) * (
            (1 / tau) - (1 / eps) * expfac1
        )


class ChromaticGaussianEvent(DelayComponent):
    r"""Simple chromatic Gaussian model for extreme scattering events or other chromatic transient features

    This phenomenological model is defined as a Gaussian in time multiplied by a power law in radio frequency with variable chromaticity.
    The model parameters include the event epoch, sign, amplitude, chromatic index, and standard deviation of the Gaussian.
    See Coles et al. (2015), The Astrophysical Journal, 808, 113 (arXiv:1506.07948) for more details.
    The explicit mathematical form of the Gaussian event is as follows.

    .. math::

        \Delta_{\text{Gaussian}}(t)= \sum_{i}A_{i}\left(\frac{f}{f_{\text{ref}}}\right)^{\text{chromidx}_{i}}\exp\left[-\frac{(t-T_{i})^2}{2\sigma_{i}^2}\right]


    Parameters supported:

    .. paramtable::
        :class: pint.models.transient_events.ChromaticGaussianEvent
    """

    register = True
    category = "chromatic_gaussian_event"

    def __init__(self):
        super().__init__()

        self.add_param(
            floatParameter(
                name=f"CHROMGAUSS_FREF",
                units="MHz",
                description="Chromatic Gaussian event reference frequency",
                value=1400,
                frozen=True,
                tcb2tdb_scale_factor=1,
            )
        )

        # Register delay function once (it checks for events internally)
        self.delay_funcs_component += [self.chrom_gauss_delay]

    def add_chromatic_gaussian_event(
        self,
        epoch: Union[float, u.Quantity, Time],
        log10amp: Union[float, u.Quantity],
        chromidx: Union[float, u.Quantity],
        log10sigma: Union[float, u.Quantity],
        sign: Union[int, float, u.Quantity],
        index: Optional[int] = None,
        frozen: bool = True,
        force: bool = False,
    ) -> int:
        """Add a chromatic Gaussian event to the model."""

        if index is None:
            dct = self.get_prefix_mapping_component("CHROMGAUSS_EPOCH_")
            if dct:
                index = np.max(list(dct.keys())) + 1
            else:
                index = 1  # Start at 1 if no events exist yet
        elif int(index) in self.get_prefix_mapping_component("CHROMGAUSS_EPOCH_"):
            if not force:
                raise ValueError(
                    f"Index '{index}' is already in use in this model. Please choose another."
                )
            else:
                # Remove existing event so we can re-add with new values
                self.remove_chrom_gauss_event(int(index))

        if isinstance(epoch, Time):
            epoch = epoch.mjd
        elif isinstance(epoch, u.Quantity):
            epoch = epoch.value

        self.add_param(
            prefixParameter(
                name=f"CHROMGAUSS_EPOCH_{index}",
                units="MJD",
                description="Chromatic Gaussian event epoch",
                parameter_type="MJD",
                time_scale="utc",
                value=epoch,
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["CHROMGAUSS_EPOCH_"],
            )
        )

        self.add_param(
            prefixParameter(
                name=f"CHROMGAUSS_LOGAMP_{index}",
                units="",
                value=log10amp,
                description="Log10 Chromatic Gaussian event amplitude (log10 in seconds)",
                parameter_type="float",
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["CHROMGAUSS_LOGAMP_"],
            )
        )

        self.add_param(
            prefixParameter(
                name=f"CHROMGAUSS_CHROMIDX_{index}",
                units="",
                value=chromidx,
                description="Chromatic Gaussian event chromatic index",
                parameter_type="float",
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["CHROMGAUSS_CHROMIDX_"],
            )
        )

        if isinstance(sign, u.Quantity):
            sign = sign.value
        sign = np.sign(sign)  # get the actual sign

        self.add_param(
            prefixParameter(
                name=f"CHROMGAUSS_SIGN_{index}",
                units="",
                value=sign,
                description="Chromatic Gaussian event sign",
                parameter_type="float",
                frozen=False,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["CHROMGAUSS_SIGN_"],
            )
        )

        self.add_param(
            prefixParameter(
                name=f"CHROMGAUSS_LOGSIG_{index}",
                units="",
                value=log10sigma,
                description="Log10 Chromatic Gaussian event standard deviation (log10 in days)",
                parameter_type="float",
                frozen=frozen,
                tcb2tdb_scale_factor=1,
                prefix_aliases=["CHROMGAUSS_LOGSIG_"],
            )
        )

        self.setup()
        self.validate()

        return index

    def remove_chrom_gauss_event(
        self, index: Union[float, int, List[int], np.ndarray]
    ) -> None:
        """Removes all chromatic Gaussian event parameters associated with a given index/list of indices.

        Parameters
        ----------

        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to chromatic Gaussian event indices to be removed from model.
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
            index_rf = f"{int(index):d}"
            for prefix in [
                "CHROMGAUSS_EPOCH_",
                "CHROMGAUSS_LOGAMP_",
                "CHROMGAUSS_LOGSIG_",
                "CHROMGAUSS_CHROMIDX_",
                "CHROMGAUSS_SIGN_",
            ]:
                param_name = f"{prefix}{index_rf}"
                if param_name in self.params:
                    self.remove_param(param_name)

        self.setup()
        self.validate()
        # Also re-setup parent model if attached, so model-level derivative
        # registry is cleaned up properly.
        if hasattr(self, "_parent") and self._parent is not None:
            self._parent.setup()

    def get_indices(self) -> np.ndarray:
        """Returns an array of integers corresponding to chromatic Gaussian event parameters.

        Returns
        -------
        inds : np.ndarray
            Array of chromatic Gaussian event indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "CHROMGAUSS_EPOCH_" in p]
        return np.array(inds)

    def setup(self) -> None:
        super().setup()
        # Get mapping.
        # Register the ChromGauss derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("CHROMGAUSS_EPOCH_"):
                self.register_deriv_funcs(self.d_delay_d_epoch, prefix_par)
            elif prefix_par.startswith("CHROMGAUSS_LOGAMP_"):
                self.register_deriv_funcs(self.d_delay_d_log10amp, prefix_par)
            elif prefix_par.startswith("CHROMGAUSS_LOGSIG_"):
                self.register_deriv_funcs(self.d_delay_d_log10sigma, prefix_par)
            elif prefix_par.startswith("CHROMGAUSS_CHROMIDX_"):
                self.register_deriv_funcs(self.d_delay_d_chromidx, prefix_par)
            elif prefix_par.startswith("CHROMGAUSS_SIGN_"):
                self.register_deriv_funcs(self.d_delay_d_sign, prefix_par)

    def get_ffac(self, toas: TOAs) -> np.ndarray:
        """Compute f/fref where f is the observing frequency."""
        f = self._parent.barycentric_radio_freq(toas)
        fref = self.CHROMGAUSS_FREF.quantity
        return (f / fref).to_value(u.dimensionless_unscaled)

    def chrom_gauss_delay_term(
        self, t_mjd: np.ndarray, ffac: np.ndarray, ii: int
    ) -> u.Quantity:
        """Compute the delay for a single chromatic Gaussian event."""
        T = getattr(self, f"CHROMGAUSS_EPOCH_{ii}").value
        dt = (t_mjd - T) * u.day

        log10Amp = getattr(self, f"CHROMGAUSS_LOGAMP_{ii}").value
        chromidx = getattr(self, f"CHROMGAUSS_CHROMIDX_{ii}").value
        log10sigma = getattr(self, f"CHROMGAUSS_LOGSIG_{ii}").value
        sign = getattr(self, f"CHROMGAUSS_SIGN_{ii}").value

        sigma = 10 ** (log10sigma) * u.day
        return (
            sign
            * 10**log10Amp
            * u.s
            * np.exp(-0.5 * (dt.value) ** 2 / (sigma.value) ** 2)
            * ffac ** (-chromidx)
        )

    def chrom_gauss_delay(self, toas: TOAs, acc_delay=None) -> u.Quantity:
        """Total delay across all chromatic Gaussian events."""
        indices = self.get_indices()

        ffac = self.get_ffac(toas)
        t_mjd = toas["tdbld"]

        delay = np.zeros(len(toas)) * u.s
        for ii in indices:
            delay += self.chrom_gauss_delay_term(t_mjd, ffac, ii)

        return delay

    def d_delay_d_log10amp(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. chromatic Gaussian amplitude."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)
        return self.chrom_gauss_delay_term(toas["tdbld"], ffac, ii) * np.log(10)

    def d_delay_d_chromidx(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. chromatic Gaussian index."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)
        return self.chrom_gauss_delay_term(toas["tdbld"], ffac, ii) * np.log(ffac**-1)

    def d_delay_d_log10sigma(
        self, toas: TOAs, param: str, acc_delay=None
    ) -> u.Quantity:
        """Derivative of delay w.r.t. chromatic Gaussian st. deviation."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)

        t0_mjd = getattr(self, f"CHROMGAUSS_EPOCH_{ii}").value
        dt = (toas["tdbld"] - t0_mjd) * u.day

        log10sigma = getattr(self, f"CHROMGAUSS_LOGSIG_{ii}").value
        sigma = 10 ** (log10sigma) * u.day

        return (
            self.chrom_gauss_delay_term(toas["tdbld"], ffac, ii)
            * (dt.value) ** 2
            / (10 ** (-2*log10sigma))
            * np.log(10)
        )

    def d_delay_d_epoch(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. chromatic Gaussian epoch."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)

        T = getattr(self, f"CHROMGAUSS_EPOCH_{ii}").value
        dt = (toas["tdbld"] - T) * u.day

        log10sigma = getattr(self, f"CHROMGAUSS_LOGSIG_{ii}").value

        return self.chrom_gauss_delay_term(toas["tdbld"], ffac, ii) * (
            dt.value / (10 ** (2 * log10sigma))
        )

    def d_delay_d_sign(self, toas: TOAs, param: str, acc_delay=None) -> u.Quantity:
        """Derivative of delay w.r.t. chromatic Gaussian sign."""
        ii = getattr(self, param).index
        ffac = self.get_ffac(toas)
        # Derivative w.r.t. sign is just the delay term divided by sign
        sign = getattr(self, f"CHROMGAUSS_SIGN_{ii}").value
        return self.chrom_gauss_delay_term(toas["tdbld"], ffac, ii) / sign
