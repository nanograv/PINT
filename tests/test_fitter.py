#! /usr/bin/env python
import os
from copy import deepcopy

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pytest
import astropy.units as u

import pint.models as tm
from pint import fitter, toa
from pinttestdata import datadir
import pint.models.parameter as param
from pint import ls


@pytest.mark.xfail
def test_fitter_basic():
    m = tm.get_model(os.path.join(datadir, "NGC6440E.par"))
    m.fit_params = ["F0", "F1"]
    e = 1 * u.us
    t = toa.make_fake_toas(56000, 59000, 16, m, error=e)

    T = (t.last_MJD - t.first_MJD).to(u.s)

    dF0 = (e * m.F0.quantity / T).to(u.Hz)

    f_1 = fitter.WLSFitter(toas=t, model=m)
    f_1.fit_toas()
    assert abs(f_1.model.F0 - m.F0) < dF0

    m_2 = deepcopy(m)
    m_2.F0.quantity += 2 * dF0
    assert abs(m_2.F0 - m.F0) > dF0
    f_2 = fitter.WLSFitter(toas=t, model=m_2)
    f_2.fit_toas()
    assert abs(f_2.model.F0 - m.F0) < dF0


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
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
    xt = f.resids.toas.get_mjds().value
    yerr = t.get_errors() * 1e-6
    plt.close()
    p1 = plt.errorbar(
        xt,
        f.resids.toas.get_mjds().value,
        f.resids.time_resids.value,
        yerr.value,
        fmt="bo",
    )

    # Do a 4-parameter fit
    f.model.free_params = ("F0", "F1", "RAJ", "DECJ")
    f.fit_toas()

    # Check the number of degrees of freedom in the fit.
    # Fitting the 4 params above, plus the 1 implicit global offset = 5 free parameters.
    # NTOA = 62, so DOF = 62 - 5 = 57
    assert f.resids.dof == 57

    print("chi^2 is %0.2f after 4-param fit" % f.resids.chi2)
    p2 = plt.errorbar(xt, f.resids.time_resids.value, yerr.value, fmt="go")

    # Make sure the summary printing works
    f.print_summary()

    # Try a few utils

    # Now perturb F1 and fit only that. This doesn't work, though tempo2 easily fits
    # it.
    f.model.F1.value = 1.1 * f.model.F1.value
    f.fit_toas()
    print("chi^2 is %0.2f after perturbing F1" % f.resids.chi2)
    p3 = plt.errorbar(xt, f.resids.time_resids.value, yerr.value, fmt="ms")

    f.model.free_params = ["F1"]
    f.fit_toas()
    print(
        'chi^2 is %0.2f after fitting just F1 with default method="Powell"'
        % f.resids.chi2
    )
    p4 = plt.errorbar(xt, f.resids.time_resids.value, yerr.value, fmt="k^")

    # Try a different method. This works, apparently.
    # NOTE: Disable this part since the interface has changed.
    # f.fit_toas(method='Nelder-Mead')
    # print('chi^2 is %0.2f after fitting just F1 with method="Nelder-Mead"' % f.resids.chi2)
    # p5=plt.errorbar(xt,f.resids.time_resids.value,yerr,fmt='rs');
    #
    # # Perturb F1 again
    # f.model.F1.value=1.1*f.model.F1.value
    # f.fit_toas()
    # print('chi^2 is %0.2f after perturbing F1' % f.resids.chi2)
    #
    # # This method does not converge in 20 iterations when fitting four params
    # f.set_fitparams('F0','F1','RA','DEC')
    # f.fit_toas(method='Nelder-Mead')
    # print(f.fitresult)
    #
    # # Powell method does claim to converge, but clearly does not actually find
    # # global minimum.
    # f.fit_toas()
    # p6=plt.errorbar(xt,f.resids.time_resids.value,yerr,fmt='g^');
    # print(f.fitresult)
    # print('chi^2 is %0.2f after fitting F0,F1,RA,DEC with method="Powell"' % f.resids.chi2)
    #
    # plt.grid();
    # plt.legend([p1,p2,p3,p4,p5,p6],['Initial','4-param','Perturb F1',
    #                                 'Fit F1 with method="Powell"',
    #                                 'Fit F1 with method="Nelder-Mead"',
    #                                 'Fit F0,F1,RA,DEC with method="Powell"'],
    #            loc=3)
    # #plt.show()
    # plt.savefig(os.path.join(datadir,"test_fitter_plot.pdf"))


def test_ftest_nb():
    """Test for narrowband fitter class F-test."""
    m = tm.get_model(os.path.join(datadir, "J0023+0923_ell1_simple.par"))
    t = toa.make_fake_toas(56000.0, 56001.0, 10, m, freq=1400.0, obs="AO")
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
    assert isinstance(ft["ft"], float) or isinstance(ft["ft"], bool)
    # Test return the full output
    Ftest_dict = f.ftest(F2, "Spindown", remove=False, full_output=True)
    assert isinstance(Ftest_dict["ft"], float) or isinstance(Ftest_dict["ft"], bool)
    # Test removing parameter
    F1 = param.prefixParameter(
        parameter_type="float", name="F1", value=0.0, units=u.Hz / u.s, frozen=False
    )
    ft = f.ftest(F1, "Spindown", remove=True)
    assert isinstance(ft["ft"], float) or isinstance(ft["ft"], bool)
    Ftest_dict = f.ftest(F1, "Spindown", remove=True, full_output=True)
    assert isinstance(Ftest_dict["ft"], float) or isinstance(Ftest_dict["ft"], bool)


def test_ftest_wb():
    """Test for wideband fitter class F-test."""
    wb_m = tm.get_model(os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.gls.par"))
    wb_t = toa.get_TOAs(
        os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.tim"), usepickle=True
    )
    # wb_t = toa.make_fake_toas(56000.0, 56001.0, 10, wb_m, freq=1400.0, obs="GB")
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
    assert isinstance(Ftest_dict["ft"], float) or isinstance(Ftest_dict["ft"], bool)
    # Test removing parallax
    Ftest_dict = wb_f.ftest(PX, PX_Component, remove=True, full_output=True)
    assert isinstance(Ftest_dict["ft"], float) or isinstance(Ftest_dict["ft"], bool)
