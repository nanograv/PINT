from copy import deepcopy
from typing import List, Optional, Tuple
from itertools import product as cartesian_product

from joblib import Parallel, cpu_count, delayed
import numpy as np
from astropy import units as u

from pint.models.chromatic_model import ChromaticCM
from pint.models.dispersion_model import DispersionDM
from pint.models.phase_offset import PhaseOffset
from pint.models.timing_model import TimingModel
from pint.toa import TOAs
from pint.logging import setup as setup_log
from pint.utils import (
    akaike_information_criterion,
    cmwavex_setup,
    dmwavex_setup,
    wavex_setup,
)


def find_optimal_nharms(
    model: TimingModel,
    toas: TOAs,
    include_components: List[str] = ["WaveX", "DMWaveX", "CMWaveX"],
    nharms_max: int = 45,
    chromatic_index: float = 4,
    num_parallel_jobs: Optional[int] = None,
) -> Tuple[tuple, np.ndarray]:
    """Find the optimal number of harmonics for `WaveX`/`DMWaveX`/`CMWaveX` using the
    Akaike Information Criterion.

    Parameters
    ----------
    model: `pint.models.timing_model.TimingModel`
        The timing model. Should not already contain `WaveX`/`DMWaveX` or `PLRedNoise`/`PLDMNoise`.
    toas: `pint.toa.TOAs`
        Input TOAs
    component: list[str]
        Component names; a non-empty sublist of ["WaveX", "DMWaveX", "CMWaveX"]
    nharms_max: int
        Maximum number of harmonics

    Returns
    -------
    aics: ndarray
        Array of AIC values.
    nharms_opt: tuple
        Optimal numbers of harmonics
    """
    assert len(set(include_components).intersection(set(model.components.keys()))) == 0
    assert len(include_components) > 0

    idxs = list(
        cartesian_product(
            *np.repeat([np.arange(nharms_max + 1)], len(include_components), axis=0)
        )
    )

    if num_parallel_jobs is None:
        num_parallel_jobs = cpu_count()

    aics_flat = Parallel(n_jobs=num_parallel_jobs)(
        delayed(
            lambda ii: compute_aic(model, toas, include_components, ii, chromatic_index)
        )(ii)
        for ii in idxs
    )

    aics = np.reshape(aics_flat, [nharms_max + 1] * len(include_components))

    assert np.isfinite(aics).all(), "Infs/NaNs found in AICs!"

    return aics, np.unravel_index(np.argmin(aics), aics.shape)


def compute_aic(
    model: TimingModel,
    toas: TOAs,
    include_components: List[str],
    ii: np.ndarray,
    chromatic_index: float,
):
    setup_log(level="WARNING")

    model1 = prepare_model(
        model, toas.get_Tspan(), include_components, ii, chromatic_index
    )

    from pint.fitter import Fitter

    # Downhill fitters don't work well here.
    # TODO: Investigate this.
    ftr = Fitter.auto(toas, model1, downhill=False)
    ftr.fit_toas(maxiter=10)

    return akaike_information_criterion(ftr.model, toas)


def prepare_model(
    model: TimingModel,
    Tspan: u.Quantity,
    include_components: List[str],
    nharms: np.ndarray,
    chromatic_index: float,
):
    model1 = deepcopy(model)

    for comp in ["PLRedNoise", "PLDMNoise", "PLCMNoise"]:
        if comp in model1.components:
            model1.remove_component(comp)

    if "PhaseOffset" not in model1.components:
        model1.add_component(PhaseOffset())
    model1.PHOFF.frozen = False

    for jj, comp in enumerate(include_components):
        if comp == "WaveX":
            nharms_wx = nharms[jj]
            if nharms_wx > 0:
                wavex_setup(model1, Tspan, n_freqs=nharms_wx, freeze_params=False)
        elif comp == "DMWaveX":
            nharms_dwx = nharms[jj]
            if nharms_dwx > 0:
                if "DispersionDM" not in model1.components:
                    model1.add_component(DispersionDM())

                model1["DM"].frozen = False

                if model1["DM1"].quantity is None:
                    model1["DM1"].quantity = 0 * model1["DM1"].units
                model1["DM1"].frozen = False

                if "DM2" not in model1.params:
                    model1.components["DispersionDM"].add_param(
                        model["DM1"].new_param(2)
                    )
                if model1["DM2"].quantity is None:
                    model1["DM2"].quantity = 0 * model1["DM2"].units
                model1["DM2"].frozen = False

                if model1["DMEPOCH"].quantity is None:
                    model1["DMEPOCH"].quantity = model1["PEPOCH"].quantity

                dmwavex_setup(model1, Tspan, n_freqs=nharms_dwx, freeze_params=False)
        elif comp == "CMWaveX":
            nharms_cwx = nharms[jj]
            if nharms_cwx > 0:
                if "ChromaticCM" not in model1.components:
                    model1.add_component(ChromaticCM())
                    model1["TNCHROMIDX"].value = chromatic_index

                model1["CM"].frozen = False
                if model1["CM1"].quantity is None:
                    model1["CM1"].quantity = 0 * model1["CM1"].units
                model1["CM1"].frozen = False

                if "CM2" not in model1.params:
                    model1.components["ChromaticCM"].add_param(
                        model1["CM1"].new_param(2)
                    )
                if model1["CM2"].quantity is None:
                    model1["CM2"].quantity = 0 * model1["CM2"].units
                model1["CM2"].frozen = False

                if model1["CMEPOCH"].quantity is None:
                    model1["CMEPOCH"].quantity = model1["PEPOCH"].quantity

                cmwavex_setup(model1, Tspan, n_freqs=nharms_cwx, freeze_params=False)
        else:
            raise ValueError(f"Unsupported component {comp}.")

    return model1
