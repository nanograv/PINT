"""Test if the split basis and weights functions for EcorrNoise and PLRedNoise
give the same result as the old code."""

import numpy as np
import astropy.units as u
import pytest
import warnings
from pint.config import examplefile
from pint.models import get_model_and_toas
from pint.models.noise_model import (
    create_ecorr_quantization_matrix,
    create_fourier_design_matrix,
    get_rednoise_freqs,
    powerlaw,
)


@pytest.fixture()
def model_and_toas():
    parfile = examplefile("B1855+09_NANOGrav_9yv1.gls.par")
    timfile = examplefile("B1855+09_NANOGrav_9yv1.tim")
    model, toas = get_model_and_toas(parfile, timfile)
    return model, toas


def ecorr_basis_weight_pair_old(ecorrcomponent, toas):
    """Old version of the ecorr_basis_weight_pair function"""
    tbl = toas.table
    t = (tbl["tdbld"].quantity * u.day).to(u.s).value
    ecorrs = ecorrcomponent.get_ecorrs()
    umats = []
    for ec in ecorrs:
        mask = ec.select_toa_mask(toas)
        if np.any(mask):
            umats.append(create_ecorr_quantization_matrix(t[mask]))
        else:
            warnings.warn(f"ECORR {ec} has no TOAs")
            umats.append(np.zeros((0, 0)))
    nc = sum(u.shape[1] for u in umats)
    umat = np.zeros((len(t), nc))
    weight = np.zeros(nc)
    nctot = 0
    for ct, ec in enumerate(ecorrs):
        mask = ec.select_toa_mask(toas)
        nn = umats[ct].shape[1]
        umat[mask, nctot : nn + nctot] = umats[ct]
        weight[nctot : nn + nctot] = ec.quantity.to(u.s).value ** 2
        nctot += nn
    return (umat, weight)


def pl_rn_basis_weight_pair_old(plrncomponent, toas):
    """Old version of the pl_rn_basis_weight_pair function."""
    tbl = toas.table
    t = (tbl["tdbld"].quantity * u.day).to(u.s).value
    amp, gam, nf = plrncomponent.get_pl_vals()
    Fmat = create_fourier_design_matrix(t, nf)
    f = get_rednoise_freqs(t, nf)
    weight = powerlaw(f, amp, gam) * f[0]
    return (Fmat, weight)


def test_ecorrnoise_basis_weights(model_and_toas):
    model, toas = model_and_toas
    ecorrcomponent = model.components["EcorrNoise"]

    basis_old, weights_old = ecorr_basis_weight_pair_old(ecorrcomponent, toas)

    basis, weights = ecorrcomponent.ecorr_basis_weight_pair(toas)

    # The assertion expects an equality because the computations have not changed.
    # They have just been rearranged into different functions. Therefore, they must
    # produce identical results.
    assert np.all(basis_old == basis) and np.all(weights_old == weights)


def test_plrednoise_basis_weights(model_and_toas):
    model, toas = model_and_toas
    plrncomponent = model.components["PLRedNoise"]

    basis_old, weights_old = pl_rn_basis_weight_pair_old(plrncomponent, toas)

    basis, weights = plrncomponent.pl_rn_basis_weight_pair(toas)

    # The assertion expects an equality because the computations have not changed.
    # They have just been rearranged into different functions. Therefore, they must
    # produce identical results.
    assert np.all(basis_old == basis) and np.all(weights_old == weights)
