from copy import deepcopy
from typing import List, Tuple
from itertools import product as cartesian_product

import numpy as np
from astropy import units as u
from scipy.optimize import minimize
from numdifftools import Hessian
from pint.models.chromatic_model import ChromaticCM
from pint.models.cmwavex import cmwavex_setup
from pint.models.dispersion_model import DispersionDM
from pint.models.dmwavex import dmwavex_setup
from pint.models.phase_offset import PhaseOffset
from pint.models.timing_model import TimingModel
from pint.models.wavex import wavex_setup
from pint.toa import TOAs
from pint.utils import (
    akaike_information_criterion,
)


def _get_wx2pl_lnlike(
    model: TimingModel, component_name: str, ignore_fyr: bool = True
) -> float:
    from pint.models.noise_model import powerlaw
    from pint import DMconst

    assert component_name in {"WaveX", "DMWaveX", "CMWaveX"}
    prefix_dict = {"WaveX": "WX", "DMWaveX": "DMWX", "CMWaveX": "CMWX"}
    prefix = prefix_dict[component_name]

    idxs = np.array(model.components[component_name].get_indices())

    fs = np.array(
        [model[f"{prefix}FREQ_{idx:04d}"].quantity.to_value(u.Hz) for idx in idxs]
    )
    f0 = np.min(fs)
    fyr = (1 / u.year).to_value(u.Hz)

    assert np.allclose(
        np.diff(np.diff(fs)), 0
    ), "WaveX/DMWaveX/CMWaveX frequencies must be uniformly spaced for this conversion to work."

    if ignore_fyr:
        year_mask = np.abs(((fs - fyr) / f0)) > 0.5

        idxs = idxs[year_mask]
        fs = np.array(
            [model[f"{prefix}FREQ_{idx:04d}"].quantity.to_value(u.Hz) for idx in idxs]
        )
        f0 = np.min(fs)

    scaling_factor = (
        1
        if component_name == "WaveX"
        else (
            DMconst / (1400 * u.MHz) ** 2
            if component_name == "DMWaveX"
            else DMconst / 1400**model.TNCHROMIDX.value
        )
    )

    a = np.array(
        [
            (scaling_factor * model[f"{prefix}SIN_{idx:04d}"].quantity).to_value(u.s)
            for idx in idxs
        ]
    )
    da = np.array(
        [
            (scaling_factor * model[f"{prefix}SIN_{idx:04d}"].uncertainty).to_value(u.s)
            for idx in idxs
        ]
    )
    b = np.array(
        [
            (scaling_factor * model[f"{prefix}COS_{idx:04d}"].quantity).to_value(u.s)
            for idx in idxs
        ]
    )
    db = np.array(
        [
            (scaling_factor * model[f"{prefix}COS_{idx:04d}"].uncertainty).to_value(u.s)
            for idx in idxs
        ]
    )

    def powl_model(params: Tuple[float, float]) -> float:
        """Get the powerlaw spectrum for the WaveX frequencies for a given
        set of parameters. This calls the powerlaw function used by `PLRedNoise`/`PLDMNoise`/`PLChromNoise`.
        """
        gamma, log10_A = params
        return (powerlaw(fs, A=10**log10_A, gamma=gamma) * f0) ** 0.5

    def mlnlike(params: Tuple[float, ...]) -> float:
        """Negative of the likelihood function that acts on the
        `[DM/CM]WaveX` amplitudes."""
        sigma = powl_model(params)
        return 0.5 * float(
            np.sum(
                (a**2 / (sigma**2 + da**2))
                + (b**2 / (sigma**2 + db**2))
                + np.log(sigma**2 + da**2)
                + np.log(sigma**2 + db**2)
            )
        )

    return mlnlike


def plrednoise_from_wavex(model: TimingModel, ignore_fyr: bool = True) -> TimingModel:
    """Convert a `WaveX` representation of red noise to a `PLRedNoise`
    representation. This is done by minimizing a likelihood function
    that acts on the `WaveX` amplitudes over the powerlaw spectral
    parameters.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
        The timing model with a `WaveX` component.
    ignore_fyr: bool
        Whether to ignore the frequency bin containinf 1 yr^-1
        while fitting for the spectral parameters.

    Returns
    -------
    pint.models.timing_model.TimingModel
        The timing model with a converted `PLRedNoise` component.
    """
    from pint.models.noise_model import PLRedNoise

    mlnlike = _get_wx2pl_lnlike(model, "WaveX", ignore_fyr=ignore_fyr)

    result = minimize(mlnlike, [4, -13], method="Nelder-Mead")
    if not result.success:
        raise ValueError("Log-likelihood maximization failed to converge.")

    gamma_val, log10_A_val = result.x

    hess = Hessian(mlnlike)
    gamma_err, log10_A_err = np.sqrt(
        np.diag(np.linalg.pinv(hess((gamma_val, log10_A_val))))
    )

    tnredc = len(model.components["WaveX"].get_indices())

    model1 = deepcopy(model)
    model1.remove_component("WaveX")
    model1.add_component(PLRedNoise())
    model1.TNREDAMP.value = log10_A_val
    model1.TNREDGAM.value = gamma_val
    model1.TNREDC.value = tnredc
    model1.TNREDAMP.uncertainty_value = log10_A_err
    model1.TNREDGAM.uncertainty_value = gamma_err

    return model1


def pldmnoise_from_dmwavex(model: TimingModel, ignore_fyr: bool = False) -> TimingModel:
    """Convert a `DMWaveX` representation of red noise to a `PLDMNoise`
    representation. This is done by minimizing a likelihood function
    that acts on the `DMWaveX` amplitudes over the powerlaw spectral
    parameters.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
        The timing model with a `DMWaveX` component.

    Returns
    -------
    pint.models.timing_model.TimingModel
        The timing model with a converted `PLDMNoise` component.
    """
    from pint.models.noise_model import PLDMNoise

    mlnlike = _get_wx2pl_lnlike(model, "DMWaveX", ignore_fyr=ignore_fyr)

    result = minimize(mlnlike, [4, -13], method="Nelder-Mead")
    if not result.success:
        raise ValueError("Log-likelihood maximization failed to converge.")

    gamma_val, log10_A_val = result.x

    hess = Hessian(mlnlike)

    H = hess((gamma_val, log10_A_val))
    assert np.all(np.linalg.eigvals(H) > 0), "The Hessian is not positive definite!"

    Hinv = np.linalg.pinv(H)
    assert np.all(
        np.linalg.eigvals(Hinv) > 0
    ), "The inverse Hessian is not positive definite!"

    gamma_err, log10_A_err = np.sqrt(np.diag(Hinv))

    tndmc = len(model.components["DMWaveX"].get_indices())

    model1 = deepcopy(model)
    model1.remove_component("DMWaveX")
    model1.add_component(PLDMNoise())
    model1.TNDMAMP.value = log10_A_val
    model1.TNDMGAM.value = gamma_val
    model1.TNDMC.value = tndmc
    model1.TNDMAMP.uncertainty_value = log10_A_err
    model1.TNDMGAM.uncertainty_value = gamma_err

    return model1


def plchromnoise_from_cmwavex(
    model: TimingModel, ignore_fyr: bool = False
) -> TimingModel:
    """Convert a `CMWaveX` representation of red noise to a `PLChromNoise`
    representation. This is done by minimizing a likelihood function
    that acts on the `CMWaveX` amplitudes over the powerlaw spectral
    parameters.

    Parameters
    ----------
    model: pint.models.timing_model.TimingModel
        The timing model with a `CMWaveX` component.

    Returns
    -------
    pint.models.timing_model.TimingModel
        The timing model with a converted `PLChromNoise` component.
    """
    from pint.models.noise_model import PLChromNoise

    mlnlike = _get_wx2pl_lnlike(model, "CMWaveX", ignore_fyr=ignore_fyr)

    result = minimize(mlnlike, [4, -13], method="Nelder-Mead")
    if not result.success:
        raise ValueError("Log-likelihood maximization failed to converge.")

    gamma_val, log10_A_val = result.x

    hess = Hessian(mlnlike)

    H = hess((gamma_val, log10_A_val))
    assert np.all(np.linalg.eigvals(H) > 0), "The Hessian is not positive definite!"

    Hinv = np.linalg.pinv(H)
    assert np.all(
        np.linalg.eigvals(Hinv) > 0
    ), "The inverse Hessian is not positive definite!"

    gamma_err, log10_A_err = np.sqrt(np.diag(Hinv))

    tndmc = len(model.components["CMWaveX"].get_indices())

    model1 = deepcopy(model)
    model1.remove_component("CMWaveX")
    model1.add_component(PLChromNoise())
    model1.TNCHROMAMP.value = log10_A_val
    model1.TNCHROMGAM.value = gamma_val
    model1.TNCHROMC.value = tndmc
    model1.TNCHROMAMP.uncertainty_value = log10_A_err
    model1.TNCHROMGAM.uncertainty_value = gamma_err

    return model1


def find_optimal_nharms(
    model: TimingModel,
    toas: TOAs,
    include_components: List[str] = ["WaveX", "DMWaveX", "CMWaveX"],
    nharms_max: int = 45,
    chromatic_index: float = 4,
):
    assert len(set(include_components).intersection(set(model.components.keys()))) == 0

    idxs = list(
        cartesian_product(
            *np.repeat([np.arange(nharms_max + 1)], len(include_components), axis=0)
        )
    )

    aics = np.zeros(np.repeat(nharms_max, len(include_components)))
    for ii in idxs:
        aics[ii] = compute_aic(model, toas, include_components, ii, chromatic_index)

    assert all(np.isfinite(aics)), "Infs/NaNs found in AICs!"

    aics -= np.min(aics)

    return aics, np.unravel_index(np.argmin(aics), aics.shape)


def compute_aic(
    model: TimingModel,
    toas: TOAs,
    include_components: List[str],
    ii: np.ndarray,
    chromatic_index: float,
):
    model1 = prepare_model(model, toas, include_components, ii, chromatic_index)

    from pint.fitter import Fitter

    ftr = Fitter.auto(toas, model1)
    ftr.fit_toas(maxiter=10)

    return akaike_information_criterion(ftr.model, toas)


def prepare_model(
    model: TimingModel,
    toas: TOAs,
    include_components: List[str],
    nharms: np.ndarray,
    chromatic_index: float,
):
    model1 = deepcopy(model)

    Tspan = toas.get_Tspan()

    if "PhaseOffset" not in model1.components:
        model1.add_component(PhaseOffset())
        model1.PHOFF.frozen = False

    if "DMWaveX" in include_components:
        if "DispersionDM" not in model1.components:
            model1.add_component(DispersionDM())

        model1.DM.frozen = False
        if model.DM1.quantity is None:
            model.DM1.quantity = 0 * model.DM1.units
        model1.DM1.frozen = False

    if "CMWaveX" in include_components:
        if "ChromaticCM" not in model1.components:
            model1.add_component(ChromaticCM())
            model1.TNCHROMIDX.value = chromatic_index

        model1.CM.frozen = False
        if model.CM1.quantity is None:
            model.CM1.quantity = 0 * model.CM1.units
        model1.CM1.frozen = False

    for jj, comp in enumerate(include_components):
        if comp == "WaveX":
            nharms_wx = nharms[jj]
            if nharms_wx > 0:
                wavex_setup(model1, Tspan, n_freqs=nharms_wx, freeze_params=False)
        elif comp == "DMWaveX":
            nharms_dwx = nharms[jj]
            if nharms_dwx > 0:
                dmwavex_setup(model1, Tspan, n_freqs=nharms_dwx, freeze_params=False)
        elif comp == "CMWaveX":
            nharms_cwx = nharms[jj]
            if nharms_cwx > 0:
                cmwavex_setup(model1, Tspan, n_freqs=nharms_cwx, freeze_params=False)
        else:
            raise ValueError(f"Unsupported component {comp}.")

    return model1
