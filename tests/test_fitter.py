#! /usr/bin/env python
import os

# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pytest
import astropy.units as u

import pint.models as tm
from pint import fitter, toa
from pinttestdata import datadir
import pint.models.parameter as param


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
    f.set_fitparams("F0", "F1", "RA", "DEC")
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

    f.set_fitparams("F1")
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


def test_ftest():
    """Test for fitter class F-test."""
    m = tm.get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    t = toa.get_TOAs(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"))
    f = fitter.WLSFitter(toas=t, model=m)
    f.fit_toas()
    # Test adding parameters
    FD4 = param.prefixParameter(
        parameter_type="float", name="FD4", value=0.0, units=u.s, frozen=False
    )
    ft = f.ftest(FD4, "FD", remove=False)
    assert isinstance(ft, float) or isinstance(ft, bool)
    # Test return the full output
    ft, r_rms, chi2, dof = f.ftest(FD4, "FD", remove=False, full_output=True)
    # Test removing parameter
    FD3 = param.prefixParameter(
        parameter_type="float", name="FD3", value=0.0, units=u.s, frozen=False
    )
    ft = f.ftest(FD3, "FD", remove=True)
    assert isinstance(ft, float) or isinstance(ft, bool)
    ft, r_rms, chi2, dof = f.ftest(FD3, "FD", remove=True, full_output=True)
