from copy import deepcopy
from typing import List
from itertools import product as cartesian_product

import numpy as np
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
        aics[*ii] = compute_aic(model, toas, include_components, ii, chromatic_index)

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
