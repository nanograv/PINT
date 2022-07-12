"""Test chi^2 gridding routines"""
import concurrent.futures
import io
import multiprocessing
import os

import astropy.units as u
import numpy as np
import pytest

import pint.config
import pint.gridutils
import pint.models.parameter as param
from pint.fitter import DownhillGLSFitter, DownhillWLSFitter, GLSFitter, WLSFitter
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
        5,
    )
    F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        7,
    )

    chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1), ncpu=1)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_extraparams_singleprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        5,
    )
    F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        7,
    )

    chi2grid, extraparams = pint.gridutils.grid_chisq(
        f, ("F0", "F1"), (F0, F1), ("DM",), ncpu=1
    )

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
        5,
    )
    F1 = np.linspace(
        f.model.F1.quantity - 3 * f.model.F1.uncertainty,
        f.model.F1.quantity + 3 * f.model.F1.uncertainty,
        7,
    )

    chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0", "F1"), (F0, F1), ncpu=ncpu)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_oneparam():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        5,
    )

    chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0",), (F0,), ncpu=ncpu)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_oneparam_extraparam():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        5,
    )

    chi2grid, extraparams = pint.gridutils.grid_chisq(
        f, ("F0",), (F0,), ("DM",), ncpu=ncpu
    )

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
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=ncpu,
    ) as executor:
        chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0",), (F0,), executor=executor)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_3param_singleprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        3,
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
    chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0", "F1", "DM"), (F0, F1, DM), ncpu=1)

    assert np.isclose(bestfit, chi2grid.min())


def test_grid_3param_multiprocessor():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        3,
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
    chi2grid, _ = pint.gridutils.grid_chisq(
        f, ("F0", "F1", "DM"), (F0, F1, DM), ncpu=ncpu
    )

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
    chi2grid_tau, params, _ = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
        ncpu=1,
    )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)


def test_grid_derived_extraparam_singleprocessor():
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
    chi2grid_tau, params, extraparams = pint.gridutils.grid_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        (F0, tau),
        ("DM",),
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
    chi2grid_tau, params, _ = pint.gridutils.grid_chisq_derived(
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
        chi2grid_tau, params, _ = pint.gridutils.grid_chisq_derived(
            f,
            ("F0", "F1"),
            (lambda x, y: x, lambda x, y: -x / 2 / y),
            (F0, tau),
            executor=executor,
        )
    assert np.isclose(bestfit, chi2grid_tau.min(), atol=1)


def test_grid_derived_extraparam_existingexecutor():
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=ncpu) as executor:
        chi2grid_tau, params, extraparams = pint.gridutils.grid_chisq_derived(
            f,
            ("F0", "F1"),
            (lambda x, y: x, lambda x, y: -x / 2 / y),
            (F0, tau),
            ("DM",),
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
        3,
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
    chi2grid, _ = pint.gridutils.grid_chisq(f, ("F0", "F1", "F2"), (F0, F1, F2), ncpu=1)

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
        3,
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
    chi2grid, _ = pint.gridutils.grid_chisq(
        f, ("F0", "F1", "F2"), (F0, F1, F2), ncpu=ncpu
    )

    assert np.isclose(bestfit, chi2grid.min())


@pytest.mark.parametrize(
    "fitter", [GLSFitter, WLSFitter, DownhillWLSFitter, DownhillGLSFitter]
)
def test_grid_fitters_singleprocessor(fitter):
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)
    f = fitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2
    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        7,
    )
    chi2grid, _ = pint.gridutils.grid_chisq(
        f, ("F0",), (F0,), printprogress=False, ncpu=1
    )

    assert np.isclose(bestfit, chi2grid.min())


@pytest.mark.parametrize(
    "fitter", [GLSFitter, WLSFitter, DownhillWLSFitter, DownhillGLSFitter]
)
def test_grid_fitters_multiprocessor(fitter):
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)
    f = fitter(t, m)
    f.fit_toas()
    bestfit = f.resids.chi2
    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        7,
    )
    chi2grid, _ = pint.gridutils.grid_chisq(
        f, ("F0",), (F0,), printprogress=False, ncpu=ncpu
    )

    assert np.isclose(bestfit, chi2grid.min())


def test_tuple_fit():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    # find the best-fit
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        25,
    )
    F1 = np.ones(len(F0)) * f.model.F1.quantity
    parnames = ("F0", "F1")
    parvalues = list(zip(F0, F1))

    chi2, extra = pint.gridutils.tuple_chisq(
        f, parnames, parvalues, extraparnames=("DM",)
    )
    f.model.F0.quantity = F0[3]
    f.model.F1.quantity = F1[3]
    f.model.F0.frozen = True
    f.model.F1.frozen = True
    f.fit_toas()
    assert np.isclose(chi2[3], f.resids.calc_chi2())
    assert np.isclose(chi2.min(), bestfit)
    assert np.isclose(extra["DM"][3], f.model.DM.quantity)


def test_derived_tuple_fit():
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    f = WLSFitter(t, m)
    # find the best-fit
    f.fit_toas()
    bestfit = f.resids.chi2

    F0 = np.linspace(
        f.model.F0.quantity - 3 * f.model.F0.uncertainty,
        f.model.F0.quantity + 3 * f.model.F0.uncertainty,
        53,
    )

    tau = np.linspace(8.1, 8.3, 53) * 100 * u.Myr
    parvalues = list(zip(F0, tau))
    chi2_tau, params, _ = pint.gridutils.tuple_chisq_derived(
        f,
        ("F0", "F1"),
        (lambda x, y: x, lambda x, y: -x / 2 / y),
        parvalues,  # ncpu=1,
    )
    assert np.isclose(bestfit, chi2_tau.min(), atol=3)
