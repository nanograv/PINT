"""Test chi^2 gridding routines"""
import concurrent.futures
import io
import multiprocessing
import os

import astropy.units as u
import numpy as np

import pint.config
import pint.gridutils
from pint.fitter import WLSFitter
from pint.models.model_builder import get_model_and_toas


def test_grid_singleprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        15,
    )
    F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        17,
    )

    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1), ncpu=1)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_multiprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        15,
    )
    F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        17,
    )

    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1))

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_oneparam():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 5 * f.model.F0.uncertainty,
        f.model.F0.quantity + 5 * f.model.F0.uncertainty,
        21,
    )

    chi2grid = pint.gridutils.grid_chisq(f, ("F0",), (F0,))

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_oneparam_existingexecutor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 5 * f.model.F0.uncertainty,
        f.model.F0.quantity + 5 * f.model.F0.uncertainty,
        21,
    )
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    )
    chi2grid = pint.gridutils.grid_chisq(f, ("F0",), (F0,), executor=executor)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_oneparam_noexecutor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 5 * f.model.F0.uncertainty,
        f.model.F0.quantity + 5 * f.model.F0.uncertainty,
        21,
    )
    chi2grid = pint.gridutils.grid_chisq(f, ("F0",), (F0,), executor=None)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_3param():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        9,
    )
    F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        7,
    )
    DM = np.linspace(
        f.model.DM.quantity - 3 * f.model.DM.uncertainty,
        f.model.DM.quantity + 3 * f.model.DM.uncertainty,
        5,
    )
    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1", "DM"), (F0, F1, DM))

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_derived_singleprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        15,
    )
    tau = np.linspace(8.1, 8.3, 13) * 100 * u.Myr
    chi2grid_tau, params = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
        ncpu=1,
    )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)


def test_grid_derived_multiprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        15,
    )
    tau = np.linspace(8.1, 8.3, 13) * 100 * u.Myr
    chi2grid_tau, params = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
    )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)


def test_grid_derived_existingexecutor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        15,
    )
    tau = np.linspace(8.1, 8.3, 13) * 100 * u.Myr

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    )
    chi2grid_tau, params = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
        executor=executor,
    )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)


def test_grid_derived_noexecutor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        15,
    )
    tau = np.linspace(8.1, 8.3, 13) * 100 * u.Myr

    chi2grid_tau, params = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
        executor=None,
    )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)
