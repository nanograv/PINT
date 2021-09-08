"""Test chi^2 gridding routines"""
import concurrent.futures
import io
import multiprocessing
import os
import pytest

import astropy.units as u
import numpy as np

import pint.config
import pint.gridutils
import pint.models.parameter as param
from pint.fitter import WLSFitter
from pint.models.model_builder import get_model_and_toas

# for multi-core tests, don't use all available CPUs
ncpu = 2


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

    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1), ncpu=ncpu)

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

    chi2grid = pint.gridutils.grid_chisq(f, ("F0",), (F0,), ncpu=ncpu)

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
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu,)
    chi2grid = pint.gridutils.grid_chisq(f, ("F0",), (F0,), executor=executor)

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
    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1", "DM"), (F0, F1, DM), ncpu=ncpu)

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
        f, ("F0", "F1"), (lambda x, y: x, lambda x, y: -x / 2 / y), (F0, tau), ncpu=1,
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
        f, ("F0", "F1"), (lambda x, y: x, lambda x, y: -x / 2 / y), (F0, tau), ncpu=ncpu
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

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=ncpu)
    chi2grid_tau, params = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
        executor=executor,
    )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)


def test_grid_3param_prefix_singleprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    # add a F2 to the model
    modelcomponent = m.components["Spindown"]
    p = param.prefixParameter(
        parameter_type="float",
        name="F2",
        value=0,
        units=modelcomponent.F_unit(2),
        uncertainty=0,
        description=modelcomponent.F_description(2),
        longdouble=True,
        frozen=False,
    )
    modelcomponent.add_param(p, setup=True)
    m.validate()

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
    F2 = np.linspace(
        f.model.F2.quantity - 3 * f.model.F2.uncertainty,
        f.model.F2.quantity + 3 * f.model.F2.uncertainty,
        5,
    )
    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1", "F2"), (F0, F1, F2), ncpu=1)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_3param_prefix_multiprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    # add a F2 to the model
    modelcomponent = m.components["Spindown"]
    p = param.prefixParameter(
        parameter_type="float",
        name="F2",
        value=0,
        units=modelcomponent.F_unit(2),
        uncertainty=0,
        description=modelcomponent.F_description(2),
        longdouble=True,
        frozen=False,
    )
    modelcomponent.add_param(p, setup=True)
    m.validate()

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
    F2 = np.linspace(
        f.model.F2.quantity - 3 * f.model.F2.uncertainty,
        f.model.F2.quantity + 3 * f.model.F2.uncertainty,
        5,
    )
    chi2grid = pint.gridutils.grid_chisq(f, ("F0", "F1", "F2"), (F0, F1, F2), ncpu=ncpu)

    assert np.isclose(bestfit, chi2grid.min())
