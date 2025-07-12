"""Exponential dips due to, e.g., profile change events in J1713+0747"""

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

    Parameters supported:

    .. paramtable::
        :class: pint.models.expdip.SimpleExponentialDip
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
            )
        )

        self.setup()
        self.validate()

        return index

    def remove_exp_dip(self, index: Union[float, int, List[int], np.ndarray]):
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
        inds = [int(p.split("_")[-1]) for p in self.params if "EXPEP_" in p]
        return np.array(inds)

    def setup(self):
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

    def expdip_delay(self, toas: TOAs, acc_delay=None):
        indices = self.get_indices()

        delay = np.zeros(len(toas)) * u.s

        f = self._parent.barycentric_radio_freq(toas)
        fref = 1400 * u.MHz
        ffac = f / fref

        eps = self.EXPDIPEPS.quantity

        for ii in indices:
            T = getattr(self, f"EXPDIPEP_{ii}").value
            dt = (toas["tdbld"] - T) * u.day

            A = getattr(self, f"EXPDIPAMP_{ii}").quantity
            gamma = getattr(self, f"EXPDIPIDX_{ii}").quantity
            tau = getattr(self, f"EXPDIPTAU_{ii}").quantity

            delay += (
                -A
                * ffac**gamma
                * (tau / eps) ** (eps / tau)
                * (tau / (tau - eps)) ** ((tau - eps) / tau)
                * np.exp(-dt / tau)
                / (1 + np.exp(-dt / eps))
            )

        return delay

    def d_delay_d_A(self, toas: TOAs, param: str, acc_delay=None):
        ii = getattr(self, param).index

        f = self._parent.barycentric_radio_freq(toas)
        fref = 1400 * u.MHz
        ffac = f / fref

        eps = self.EXPDIPEPS.quantity

        T = getattr(self, f"EXPDIPEP_{ii}").value
        dt = (toas["tdbld"] - T) * u.day

        gamma = getattr(self, f"EXPDIPIDX_{ii}").quantity
        tau = getattr(self, f"EXPDIPTAU_{ii}").quantity

        return (
            -(ffac**gamma)
            * (tau / eps) ** (eps / tau)
            * (tau / (tau - eps)) ** ((tau - eps) / tau)
            * np.exp(-dt / tau)
            / (1 + np.exp(-dt / eps))
        )

    def d_delay_d_expindex(self, toas: TOAs, param: str, acc_delay=None):
        ii = getattr(self, param).index

        f = self._parent.barycentric_radio_freq(toas)
        fref = 1400 * u.MHz
        ffac = f / fref

        eps = self.EXPEPS.quantity

        t0_mjd = getattr(self, f"EXPEP_{ii}").value
        dt = (toas["tdbld"] - t0_mjd) * u.day

        mask_factor = 1 / (1 + np.exp(-dt / eps))

        A = getattr(self, f"EXPPH_{ii}").quantity
        gamma = getattr(self, f"EXPINDEX_{ii}").quantity
        tau = getattr(self, f"EXPTAU_{ii}").quantity

        return -A * ffac**gamma * np.log(ffac) * np.exp(-dt / tau) * mask_factor

    def d_delay_d_exptau(self, toas: TOAs, param: str, acc_delay=None):
        ii = getattr(self, param).index

        f = self._parent.barycentric_radio_freq(toas)
        fref = 1400 * u.MHz
        ffac = f / fref

        eps = self.EXPEPS.quantity

        t0_mjd = getattr(self, f"EXPEP_{ii}").value
        dt = (toas["tdbld"] - t0_mjd) * u.day

        mask_factor = 1 / (1 + np.exp(-dt / eps))

        A = getattr(self, f"EXPPH_{ii}").quantity
        gamma = getattr(self, f"EXPINDEX_{ii}").quantity
        tau = getattr(self, f"EXPTAU_{ii}").quantity

        return -A * ffac**gamma * np.exp(-dt / tau) * (dt / tau / tau) * mask_factor

    def d_delay_d_expep(self, toas: TOAs, param: str, acc_delay=None):
        ii = getattr(self, param).index

        f = self._parent.barycentric_radio_freq(toas)
        fref = 1400 * u.MHz
        ffac = f / fref

        eps = self.EXPEPS.quantity

        t0_mjd = getattr(self, f"EXPEP_{ii}").value
        dt = (toas["tdbld"] - t0_mjd) * u.day

        mask_factor = 1 / (1 + np.exp(-dt / eps))

        A = getattr(self, f"EXPPH_{ii}").quantity
        gamma = getattr(self, f"EXPINDEX_{ii}").quantity
        tau = getattr(self, f"EXPTAU_{ii}").quantity

        return (
            -A
            * ffac**gamma
            * np.exp(-dt / tau)
            / tau
            * mask_factor
            * ((1 + (1 - tau / eps) * np.exp(-dt / eps)) / (1 + np.exp(-dt / eps)))
        )
