import logging
import pytest

import numpy as np

from pinttestdata import datadir
from pint.templates import lcfitters, lcprimitives, lceprimitives, lcnorm, lcenorm, lctemplate


def gauss(x, x0, s):
    return 1.0 / s / (2 * np.pi) ** 0.5 * np.exp(-0.5 * (x - x0) ** 2 / s**2)


def test_prim_gauss_definition():
    """Make sure objects adequately implement mathematical intent."""

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
    """Make sure objects adequately implement mathematical intent."""

    # check keyword parsing
    with pytest.raises(ValueError):
        lcg = lcprimitives.LCGaussian(bogus=3)

    # a template should *always* have total normalization set to 1
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.01
    )

    assert(lct.num_parameters()==6)
    lct.freeze_parameters()
    assert(lct.num_parameters()==0)
    lct[0].free[0] = True
    assert(lct.num_parameters()==1)
    lct[-1].free[0] = True
    assert(lct.num_parameters()==2)
    assert(lct.num_parameters(free=False)==6)
    b = np.asarray(lct.get_bounds())
    assert(b.shape[0]==2)
    assert(b.shape[1]==2)
    b = np.asarray(lct.get_bounds(free=False))
    assert(b.shape[0]==6)
    assert(b.shape[1]==2)
    assert(lct.num_parameters()==len(lct.get_parameters()))
    lct.free_parameters()
    assert(lct.num_parameters()==6)

    assert(lct[-1].name=='NormAngles')

    # check that instantiated template has same norm as input
    assert(abs(lct.norm() - (0.25 + 0.35)) < 1e-10)

    # check that that template correctly evaluates weighted sum of comps
    expected_val = (
        0.25 * gauss(0.49, 0.5, 0.01)
        + 0.35 * gauss(0.49, 0.48, 0.01)
        + (1 - 0.25 - 0.35)
    )
    assert(abs(lct(0.49) - expected_val) < 1e-6)

    # check cdf
    assert(lct.cdf(1)==1)
    assert(lct.cdf(0)==0)

    # test sorting components
    lct.order_primitives(order=0) # sort by position, should swap them
    assert(lct[0].get_location()==0.48)
    assert(lct[1].get_location()==0.50)
    assert(abs(lct[-1]()[0]-0.35)<1e-10)
    assert(abs(lct[-1]()[1]-0.25)<1e-10)
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, 
        width1=0.01, width2=0.01
    )
    lct.order_primitives(order=1) # sort by amplitude, should swap them
    assert(lct[0].get_location()==0.48)
    assert(lct[1].get_location()==0.50)
    assert(abs(lct.norms()[0]-0.35)<1e-10)
    assert(abs(lct.norms()[1]-0.25)<1e-10)
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, 
        width1=0.01, width2=0.05
    )
    lct.order_primitives(order=1) # sort by amplitude, should do nothing
    assert(lct.primitives[0].get_location()==0.50)
    assert(lct.primitives[1].get_location()==0.48)
    assert(abs(lct.norms()[1]-0.35)<1e-10)
    assert(abs(lct.norms()[0]-0.25)<1e-10)
    lct.order_primitives(order=2) # sort by norm, should swap
    assert(lct.primitives[0].get_location()==0.48)
    assert(lct.primitives[1].get_location()==0.50)
    assert(abs(lct.norms()[0]-0.35)<1e-10)
    assert(abs(lct.norms()[1]-0.25)<1e-10)

    # test adding components
    # intention is that new component is "norm" of the pulsed flux, so it
    # should be norm*0.6 total, here, while the other components are scaled
    # to preserve the same pulsed flux, i.e. by (1-norm)
    lct_add = lct.add_primitive(
            lcprimitives.LCGaussian(p=[0.02,0.2]),norm=0.1)
    assert(len(lct.primitives)==2)
    assert(len(lct_add.primitives)==3)
    assert(abs(lct_add.norms().sum()-0.6)<1e-10)
    assert(abs(lct_add.norms()[-1]-0.1*0.6)<1e-10)
    assert(abs(lct_add.norms()[0]-0.35*(1-0.1))<1e-10)

    result = lct.add_primitive(
            lcprimitives.LCGaussian(p=[0.02,0.2]),norm=0.1,inplace=True)
    assert(result is None)
    assert(len(lct.primitives)==3)
    assert(abs(lct.norms().sum()-0.6)<1e-10)
    assert(abs(lct.norms()[-1]-0.1*0.6)<1e-10)
    assert(abs(lct.norms()[0]-0.35*(1-0.1))<1e-10)

    # test deletion -- should be back to original
    lct.delete_primitive(2,inplace=True)
    assert(abs(lct.norms().sum()-0.6)<1e-10)
    assert(abs(lct.norms()[0]-0.35)<1e-10)
    assert(abs(lct.norms()[1]-0.25)<1e-10)

def test_energy_dependence():
    """ Initial stab at testing energy-dependent profile modeling."""

    lcg = lcprimitives.LCGaussian(p=[0.03,0.5])
    lcg2 = lcprimitives.LCGaussian(p=[0.04,0.8])
    lct0 = lctemplate.LCTemplate([lcg,lcg2],[0.4,0.35])
    lct0.add_energy_dependence(0,slope_free=True)
    lct0[0].slope[:] = [0.02,0.1]
    lct0.add_energy_dependence(-1,slope_free=False)
    lct0[-1].slope[1] = 0.2
    lct0[-1].slope_free[1] = True

    lcg = lceprimitives.LCEGaussian(p=[0.03,0.5])
    lcg.slope[:] = [0.02,0.1]
    assert(not np.any(lcg.slope_free)) # default is False
    lcg.slope_free[:] = True
    lcg2 = lcprimitives.LCGaussian(p=[0.04,0.8])
    lcn = lcenorm.ENormAngles([0.4,0.35])
    assert(not np.any(lcn.slope_free)) # default is False
    lcn.slope_free[1] = True
    lcn.slope[1] = 0.2
    lct = lctemplate.LCTemplate([lcg,lcg2],lcn)

    assert(lct.check_gradient(quiet=True,seed=0))

    # set the seed to allow test in change in log likelihood
    np.random.seed(10)
    N=1000
    log10_ens = np.random.rand(N)*2 + 2
    we = np.ones(N)
    ph,comps = lct.random(N,weights=we,log10_ens=log10_ens,return_partition=True)

    assert(np.all(lct(ph,log10_ens=log10_ens)==lct0(ph,log10_ens=log10_ens)))

    lcf = lcfitters.LCFitter(lct,ph,weights=we,log10_ens=log10_ens)
    logl1 = lcf.loglikelihood(lct.get_parameters())
    lcf.fit(unbinned=True,use_gradient=True)
    logl2 = lcf.loglikelihood(lct.get_parameters())
    assert(abs(abs(logl2-logl1)-5.15)<0.05)



def test_template_gradient():
    """ Verify analytic gradient computation with numerical evaluation."""

    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.01
    )

    # check that analytic gradient is equal to numerical gradient
    assert(lct.check_gradient(seed=0))

    # check derivative too
    assert(lct.check_derivative())

    # check that nothing breaks when freezing a normalization angle
    lct.norms.free[0] = False
    assert(lct.check_gradient(seed=0))

    # check that nothing breaks when freezing a primitive parameter
    lct[0].free[0] = False
    assert(lct.check_gradient(seed=0))
    lct[0].free[:] = False
    assert(lct.check_gradient(seed=0))


def test_template_string_representation():
    """Exercise print functions."""
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
    # TODO -- add energy dependence
    lct = lctemplate.get_gauss2()
    x = lct.random(100)
    assert (len(x) == 100) and (np.min(x) >= 0) and (np.max(x) < 1)


def test_simple_fit_unbinned():
    """Make sure objects adequately implement mathematical intent."""

    # TODO -- add a small DC component to this to avoid negative values in
    # the log likelihood, which happen at floating point precision.  Another
    # option would be to filter those at the template level, but that's
    # probably expensive.
    lct = lctemplate.get_gauss2()

    # load in simulated phases
    ph = np.loadtxt(datadir / "template_phases.asc")
    lcf = lcfitters.LCFitter(lct, ph)
    lcf.fit(unbinned=True, estimate_errors=True, quiet=True)
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
    """Make sure objects adequately implement mathematical intent."""

    lct = lctemplate.get_gauss2()

    # load in simulated phases
    ph = np.loadtxt(datadir / "template_phases.asc")
    lcf = lcfitters.LCFitter(lct, ph, binned_bins=100)
    lcf.fit(unbinned=False, estimate_errors=True, quiet=True)
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
