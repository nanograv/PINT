import logging

import numpy as np

from pinttestdata import datadir
from pint.templates import lcfitters, lcprimitives, lctemplate


def gauss(x, x0, s):
    return 1.0 / s / (2 * np.pi) ** 0.5 * np.exp(-0.5 * (x - x0) ** 2 / s ** 2)


def test_prim_gauss_definition():
    """ Make sure objects adequately implement mathematical intent."""

    # instantiate a basic gaussian with known parameters and make sure
    # it "looks" like a gaussian.  Remember that these are wrapped, so
    # use narrow gaussians here to be close to analytic form.

    g1 = lcprimitives.LCGaussian(p=[0.01, 0.5])
    # defined as 1/sigma/(2*pi)^0.5 * (-0.5*(x-x0)^2) WRAPPED
    # because this is a narrow gaussian, it's essentially the same as
    # an unwrapped version
    expected_val = gauss(0.50, 0.5, 0.01)
    assert abs(g1(0.50) - expected_val) < 1e-6
    expected_val = gauss(0.48, 0.5, 0.01)
    assert abs(g1(0.48) - expected_val) < 1e-6


def test_prim_gauss_wrapping():

    # default is to wrap a function by 10 times; make a fat pulse here and
    # check against a manually calculated wrap

    # explicitly test wrapping
    g1 = lcprimitives.LCGaussian(p=[0.5, 0.5])
    expected_val = gauss(0.5, 0.5, 0.5)
    for i in range(1, 11):
        expected_val += gauss(0.5 + i, 0.5, 0.5)
        expected_val += gauss(0.5 - i, 0.5, 0.5)
    assert abs(g1(0.5) - expected_val) < 1e-6


def test_template_basic_functionality():
    """ Make sure objects adequately implement mathematical intent."""

    # a template should *always* have total normalization set to 1
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.01
    )

    # check that instantiated template has same norm as input
    assert abs(lct.norm() - (0.25 + 0.35)) < 1e-10

    # check that that template correctly evaluates weighted sum of comps
    expected_val = (
        0.25 * gauss(0.49, 0.5, 0.01)
        + 0.35 * gauss(0.49, 0.48, 0.01)
        + (1 - 0.25 - 0.35)
    )
    assert abs(lct(0.49) - expected_val) < 1e-6


def test_template_string_representation():
    """ Exercise print functions."""
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.01
    )
    # add in an error manually
    lct.primitives[0].errors[0] = 0.01
    expected_val = r"""
Mixture Amplitudes
------------------
P1 : 0.2500 +\- 0.0000
P2 : 0.3500 +\- 0.0000
DC : 0.4000 +\- 0.0000

P1 -- Gaussian
------------------
Width   : 0.0100 +\- 0.0100
Location: 0.5000 +\- 0.0000

P2 -- Gaussian
------------------
Width   : 0.0100 +\- 0.0000
Location: 0.4800 +\- 0.0000

delta   : 0.4800 +\- 0.0000
Delta   : 0.0200 +\- 0.0000
"""

    assert str(lct).strip() == expected_val.strip()


def test_template_simulation():
    lct = lctemplate.get_gauss2()
    x = lct.random(100)
    assert (len(x) == 100) and (np.min(x) >= 0) and (np.max(x) < 1)


def test_simple_fit_unbinned():
    """ Make sure objects adequately implement mathematical intent."""

    lct = lctemplate.get_gauss2()

    # load in simulated phases
    ph = np.loadtxt(datadir + "/template_phases.asc")
    lcf = lcfitters.LCFitter(lct, ph)
    lcf.fit(unbinned=True, estimate_errors=True)
    expected_val = r"""
Log Likelihood for fit: 1091.04

Mixture Amplitudes
------------------
P1 : 0.5840 +\- 0.0220
P2 : 0.4160 +\- 0.0220
DC : 0.0000 +\- 0.0000

P1 -- Gaussian
------------------
Width   : 0.0103 +\- 0.0004
Location: 0.1007 +\- 0.0006

P2 -- Gaussian
------------------
Width   : 0.0211 +\- 0.0010
Location: 0.5493 +\- 0.0015

delta   : 0.1007 +\- 0.0006
Delta   : 0.4486 +\- 0.0016
"""
    assert expected_val.strip() == str(lcf).strip()


def test_simple_fit_binned():
    """ Make sure objects adequately implement mathematical intent."""

    lct = lctemplate.get_gauss2()

    # load in simulated phases
    ph = np.loadtxt(datadir + "/template_phases.asc")
    lcf = lcfitters.LCFitter(lct, ph)
    lcf.fit(unbinned=False, estimate_errors=True)
    expected_val = r"""
Log Likelihood for fit: 1080.31

Mixture Amplitudes
------------------
P1 : 0.5840 +\- 0.0220
P2 : 0.4160 +\- 0.0220
DC : 0.0000 +\- 0.0000

P1 -- Gaussian
------------------
Width   : 0.0106 +\- 0.0004
Location: 0.1007 +\- 0.0006

P2 -- Gaussian
------------------
Width   : 0.0213 +\- 0.0010
Location: 0.5493 +\- 0.0015

delta   : 0.1007 +\- 0.0006
Delta   : 0.4486 +\- 0.0016
"""

    assert expected_val.strip() == str(lcf).strip()
