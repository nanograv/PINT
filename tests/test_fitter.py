import os
from io import StringIO
from copy import deepcopy
from io import StringIO

import astropy.units as u

import pint.models as tm
from pint import fitter, toa, simulation
from pinttestdata import datadir
import pint.models.parameter as param
from pint import ls
from pint.models import get_model, get_model_and_toas


def test_fitter_basic():
    m = tm.get_model(os.path.join(datadir, "NGC6440E.par"))
    m.fit_params = ["F0", "F1"]
    e = 1 * u.us
    t = simulation.make_fake_toas_uniform(56000, 59000, 16, m, error=e)

    T = (t.last_MJD - t.first_MJD).to(u.s)

    dF0 = (e * m.F0.quantity / T).to(u.Hz)

    f_1 = fitter.WLSFitter(toas=t, model=m)
    f_1.fit_toas()
    assert abs(f_1.model.F0.quantity - m.F0.quantity) < dF0

    m_2 = deepcopy(m)
    m_2.F0.quantity += 2 * dF0
    assert abs(m_2.F0.quantity - m.F0.quantity) > dF0
    f_2 = fitter.WLSFitter(toas=t, model=m_2)
    f_2.fit_toas()
    assert abs(f_2.model.F0.quantity - m.F0.quantity) < dF0


def test_fitter():
    # Get model

    m = tm.get_model(os.path.join(datadir, "NGC6440E.par"))

    # Get TOAs
    t = toa.TOAs(os.path.join(datadir, "NGC6440E.tim"))
    t.apply_clock_corrections(include_bipm=False)
    t.compute_TDBs()
    try:
        planet_ephems = m.PLANET_SHAPIRO.value
    except AttributeError:
        planet_ephems = False
    t.compute_posvels(planets=planet_ephems)

    f = fitter.WLSFitter(toas=t, model=m)

    # Print initial chi2
    print("chi^2 is initially %0.2f" % f.resids.chi2)

    # Plot initial residuals
    yerr = t.get_errors() * 1e-6

    # Do a 4-parameter fit
    f.model.free_params = ("F0", "F1", "RAJ", "DECJ")
    f.fit_toas()

    # Check the number of degrees of freedom in the fit.
    # Fitting the 4 params above, plus the 1 implicit global offset = 5 free parameters.
    # NTOA = 62, so DOF = 62 - 5 = 57
    assert f.resids.dof == 57

    print("chi^2 is %0.2f after 4-param fit" % f.resids.chi2)

    # Make sure the summary printing works
    f.print_summary()

    # Try a few utils

    # Now perturb F1 and fit only that. This doesn't work, though tempo2 easily fits
    # it.
    f.model.F1.value = 1.1 * f.model.F1.value
    f.fit_toas()
    print("chi^2 is %0.2f after perturbing F1" % f.resids.chi2)

    f.model.free_params = ["F1"]
    f.fit_toas()
    print(
        'chi^2 is %0.2f after fitting just F1 with default method="Powell"'
        % f.resids.chi2
    )


def test_ftest_nb():
    """Test for narrowband fitter class F-test."""
    m = tm.get_model(os.path.join(datadir, "J0023+0923_ell1_simple.par"))
    t = simulation.make_fake_toas_uniform(
        56000.0, 56001.0, 10, m, freq=1400.0 * u.MHz, obs="AO"
    )
    f = fitter.WLSFitter(toas=t, model=m)
    f.fit_toas()
    # Test adding parameters
    F2 = param.prefixParameter(
        parameter_type="float",
        name="F2",
        value=0.0,
        units=u.Hz / u.s / u.s,
        frozen=False,
    )
    ft = f.ftest(F2, "Spindown", remove=False)
    assert isinstance(ft["ft"], (float, bool))
    # Test return the full output
    Ftest_dict = f.ftest(F2, "Spindown", remove=False, full_output=True)
    assert isinstance(Ftest_dict["ft"], (float, bool))
    # Test removing parameter
    F1 = param.prefixParameter(
        parameter_type="float", name="F1", value=0.0, units=u.Hz / u.s, frozen=False
    )
    ft = f.ftest(F1, "Spindown", remove=True)
    assert isinstance(ft["ft"], (float, bool))
    Ftest_dict = f.ftest(F1, "Spindown", remove=True, full_output=True)
    assert isinstance(Ftest_dict["ft"], (float, bool))


def test_ftest_wb():
    """Test for wideband fitter class F-test."""
    wb_m = tm.get_model(os.path.join(datadir, "J0023+0923_ell1_simple.par"))
    wb_t = simulation.make_fake_toas_uniform(
        56000.0, 56001.0, 10, wb_m, freq=1400.0 * u.MHz, obs="GBT", wideband=True
    )
    wb_f = fitter.WidebandTOAFitter(wb_t, wb_m)
    wb_f.fit_toas()
    # Parallax
    PX = param.floatParameter(
        parameter_type="float", name="PX", value=0.0, units=u.mas, frozen=False
    )
    PX_Component = "AstrometryEcliptic"
    # A1DOT
    A1DOT = param.floatParameter(
        parameter_type="float",
        name="A1DOT",
        value=0.0,
        units=ls / u.second,
        frozen=False,
    )
    A1DOT_Component = "BinaryELL1"
    # Test adding A1DOT
    Ftest_dict = wb_f.ftest(A1DOT, A1DOT_Component, remove=False, full_output=True)
    assert isinstance(Ftest_dict["ft"], (float, bool))
    # Test removing parallax
    Ftest_dict = wb_f.ftest(PX, PX_Component, remove=True, full_output=True)
    assert isinstance(Ftest_dict["ft"], (float, bool))


def test_fitsummary_binary():
    """Test fitter print_summary() when an ELL1 binary is fit"""
    par = os.path.join(datadir, "B1855+09_NANOGrav_12yv3.wb.gls.par")
    tim = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim")

    m, t = get_model_and_toas(par, tim)

    f = fitter.WLSFitter(t, m)
    f.model.free_params = ["PB", "A1", "SINI"]
    f.fit_toas()
    f.print_summary()


def test_fitselector():
    # WB
    par = os.path.join(datadir, "B1855+09_NANOGrav_12yv3.wb.gls.par")
    tim = os.path.join(datadir, "B1855+09_NANOGrav_12yv3.wb.tim")

    m, t = get_model_and_toas(par, tim)
    f = fitter.Fitter.auto(t, m, downhill=True)
    assert isinstance(f, fitter.WidebandDownhillFitter)
    f = fitter.Fitter.auto(t, m, downhill=False)
    assert isinstance(f, fitter.WidebandTOAFitter)

    # correlated errors
    m, t = get_model_and_toas(
        os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par"),
        os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.tim"),
    )
    f = fitter.Fitter.auto(t, m, downhill=True)
    assert isinstance(f, fitter.DownhillGLSFitter)
    f = fitter.Fitter.auto(t, m, downhill=False)
    assert isinstance(f, fitter.GLSFitter)

    # uncorrelated errors
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )
    f = fitter.Fitter.auto(t, m, downhill=True)
    assert isinstance(f, fitter.DownhillWLSFitter)
    f = fitter.Fitter.auto(t, m, downhill=False)
    assert isinstance(f, fitter.WLSFitter)


def test_unitless_ell1h_bug_1316():
    m = get_model(
        StringIO(
            """
    PSR J1234+5678
    ELAT 0
    ELONG 0
    PEPOCH 57000
    F0 1
    BINARY ELL1
    PB 1 1
    A1 4 1
    TASC 57000 1
    EPS1 0 1
    EPS2 0 1
    """
        )
    )
    t = simulation.make_fake_toas_uniform(56000, 59000, 16, m)
    f = fitter.Fitter.auto(t, m)
    f.get_derived_params()
