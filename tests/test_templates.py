import pytest

import numpy as np
from scipy.integrate import quad
from scipy.special import i0, i1, erf

from pinttestdata import datadir
from pint.templates import (
    lcfitters,
    lcprimitives,
    lceprimitives,
    lcnorm,
    lcenorm,
    lctemplate,
)


def gauss(x, x0, s):
    return 1.0 / s / (2 * np.pi) ** 0.5 * np.exp(-0.5 * (x - x0) ** 2 / s**2)


def default_template():
    return lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.01
    )


def test_prim_gauss_definition():
    """Make sure objects adequately implement mathematical intent."""

    # instantiate a basic gaussian with known parameters and make sure
    # it "looks" like a gaussian.  Remember that these are wrapped, so
    # use narrow gaussians here to be close to analytic form.

    g1 = lcprimitives.LCGaussian(p=[0.01, 0.5])
    assert abs(g1(g1.get_location() + g1.hwhm()) * 2 - g1(g1.get_location())) < 1e-5
    # defined as 1/sigma/(2*pi)^0.5 * (-0.5*(x-x0)^2) WRAPPED
    # because this is a narrow gaussian, it's essentially the same as
    # an unwrapped version
    expected_val = gauss(0.50, 0.5, 0.01)
    assert abs(g1(0.50) - expected_val) < 1e-6
    expected_val = gauss(0.48, 0.5, 0.01)
    assert abs(g1(0.48) - expected_val) < 1e-6


def test_prim_io():
    """Test functionality to save a template and have it obey founds."""
    # This will fail unless we use loose bounds checking
    template1 = """
    # gauss
    -------------------------
    const = 0.95915 +/- 0.00000
    phas1 = 0.00000 +/- 0.00000
    fwhm1 = 0.02514 +/- 0.00000
    ampl1 = 0.03810 +/- 0.00000
    phas2 = 0.53059 +/- 0.00000
    fwhm2 = 0.01177 +/- 0.00000
    ampl2 = 0.00275 +/- 0.00000
    -------------------------
    """
    with pytest.raises(ValueError):
        prims, norms = lctemplate.prim_io(template1, bound_eps=0)

    # this should work and set the lower bound to 0.005
    prims, norms = lctemplate.prim_io(template1)
    assert prims[1].p[0] == prims[1].bounds[0][0]

    template2 = template1.replace("1177", "1178")
    prims, norms = lctemplate.prim_io(template2)


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
    lct = default_template()  # (x1=0.5,x2=0.48)
    assert lct.num_parameters() == 6
    lct.freeze_parameters()
    assert lct.num_parameters() == 0
    lct[0].free[0] = True
    assert lct.num_parameters() == 1
    lct[-1].free[0] = True
    assert lct.num_parameters() == 2
    assert lct.num_parameters(free=False) == 6
    b = np.asarray(lct.get_bounds())
    assert b.shape[0] == 2
    assert b.shape[1] == 2
    b = np.asarray(lct.get_bounds(free=False))
    assert b.shape[0] == 6
    assert b.shape[1] == 2
    assert lct.num_parameters() == len(lct.get_parameters())
    lct.free_parameters()
    assert lct.num_parameters() == 6

    assert lct[-1].name == "NormAngles"

    # check that instantiated template has same norm as input
    assert abs(lct.norm() - (0.25 + 0.35)) < 1e-10

    # check that that template correctly evaluates weighted sum of comps
    expected_val = (
        0.25 * gauss(0.49, 0.5, 0.01)
        + 0.35 * gauss(0.49, 0.48, 0.01)
        + (1 - 0.25 - 0.35)
    )
    assert abs(lct(0.49) - expected_val) < 1e-6

    # check rotation
    lct.rotate(-0.1)
    assert lct.primitives[0].get_location() == 0.4
    assert lct.primitives[1].get_location() == 0.38

    ## the optimal display point should roughly undo the above rotation
    assert lct.get_display_point() == 0.2

    # check wrap at 0/1
    lct.rotate(-0.4)
    assert lct.primitives[0].get_location() == 0.0
    assert lct.primitives[1].get_location() == 0.98
    assert lct(0) == lct(1)

    lct = lctemplate.get_gauss2()
    assert lct(0) > 0


def test_norms():
    n = np.asarray(
        [
            0.02683208,
            0.13441056,
            0.0236155,
            0.39370402,
            0.16328161,
            0.05283352,
            0.05245909,
            0.11335948,
        ]
    )
    lcn = lcnorm.NormAngles(n)
    assert np.allclose(lcn(), n)
    new_val = n[1] * (5.0 / 6)
    lcn.set_single_norm(1, new_val)
    assert abs(lcn()[1] - new_val) < 1e-10
    assert abs(1 - np.sum(lcn()) - np.cos(lcn.p[0]) ** 2) < 1e-10

    # this test inspired by a bug I caught when going from scalar to vector
    # there was a check that the sum of the norms <=1, but when it failed,
    # it was applied to every output!  That led all outputs to have norms=1.
    lcn = lcenorm.ENormAngles([0.55, 0.4], slope=[0.3, 0.0])
    q = lcn(log10_ens=np.linspace(2, 4.5, 101))
    assert np.any(q.sum(axis=0) <= 0.95)


def test_template_integration():
    # check integration / cdf
    lct = lctemplate.get_gauss2()
    assert (lct.get_display_point() - 0.15) < 1e-10
    assert np.all(lct.integrate([0.2, 0.3], [0.2, 0.3]) == 0)
    assert (
        np.abs(
            lct.integrate(0.8, 1.2) - (lct.integrate(0, 0.2) + lct.integrate(0.8, 1.0))
        )
        < 1e-10
    )
    assert np.abs(lct.integrate(0.2, 0.8) + lct.integrate(0.8, 0.2)) < 1e-10
    assert np.abs(lct.integrate(0.2, 0.8) - 0.4) < 1e-10
    assert np.abs(lct.integrate(0.0, 0.2) - 0.6) < 1e-10
    assert np.all(
        np.abs(lct.integrate([0.2, 0.0], [0.8, 0.2]) - np.asarray([0.4, 0.6])) < 1e-10
    )
    assert lct.cdf(1) == 1
    assert lct.cdf(0) == 0
    ph = np.linspace(0, 1, 101)
    lct.cdf(ph)


def test_template_copy():
    lct = lctemplate.get_gauss2()
    lct.set_cache_properties(ncache=347)
    # populate the cache
    lct(0.5, use_cache=True)
    assert lct.ncache == 347
    lct_copy = lct.copy()
    assert np.all(lct_copy._cache[0] == lct._cache[0])
    assert lct_copy.ncache == lct.ncache
    assert lct(0.47, use_cache=True) == lct_copy(0.47, use_cache=True)
    lct = lctemplate.get_gauss2()
    assert lct.ncache != lct_copy.ncache
    # should differ because of interpolation
    assert lct(0.47, use_cache=True) != lct_copy(0.47, use_cache=True)


def test_template_caching():
    # default is 1000 bins with linear interpolation
    lct = lctemplate.get_gauss2(
        width1=0.03, width2=0.05, x1=0.1, x2=0.5
    )  # default bin edges are at 0, 0.001, ...
    assert abs(lct.ph_edges[470] - 0.470) < 1e-15
    assert abs(lct(0.470, use_cache=True) - lct(0.470)) < 3e-15
    assert abs(lct(0.4705, use_cache=True) - 0.5 * (lct(0.470) + lct(0.471))) < 3e-15
    assert abs(lct(0.471, use_cache=True) - lct(0.471)) < 3e-15
    assert lct(0, use_cache=True) == lct(1, use_cache=True)
    assert abs(lct(0.471, use_cache=True) - lct(0.471)) < 3e-15
    lct.set_cache_properties(ncache=100)
    assert lct(0, use_cache=True) == lct(1, use_cache=True)
    v0 = lct(0, use_cache=False)
    v1 = lct(0.01, use_cache=False)
    expectation = v0 * 0.6 + v1 * 0.4
    assert abs(lct(0.004, use_cache=True) - expectation) < 1e-15
    expectation = v0 * 0.2 + v1 * 0.8
    assert abs(lct(0.008, use_cache=True) - expectation) < 1e-15
    expectation = v0 * 0.5 + v1 * 0.5
    assert abs(lct(0.005, use_cache=True) - expectation) < 1e-15

    # try an energy-dependent version
    lct = lctemplate.get_gauss2(
        width1=0.05, width2=0.05, x1=0.2, x2=0.5
    )  # default bin edges are at 0, 0.001, ...
    assert lct(0.1, log10_ens=3) == lct(0.1, log10_ens=2)
    lct.add_energy_dependence(0, slope_free=False)
    lct[0].slope[1] = 0.1
    assert abs(lct(0.1, log10_ens=2) - lct(0.2, log10_ens=3)) < 1e-7
    assert abs(lct(0.1, log10_ens=2) - lct(0.1, log10_ens=3)) > 1
    lct.set_cache_properties(ncache=100, en_edges=[2, 3, 4])
    assert lct.en_edges is not None
    assert lct.en_cens is not None
    v00 = lct(0.10, log10_ens=2)
    v01 = lct(0.10, log10_ens=3)
    v10 = lct(0.11, log10_ens=2)
    v11 = lct(0.11, log10_ens=3)
    expectation = 0.6 * v00 + 0.4 * v01
    assert abs(lct(0.1, log10_ens=2.4, use_cache=True) - expectation) < 1e-10
    expectation = 0.2 * v00 + 0.8 * v01
    assert abs(lct(0.1, log10_ens=2.8, use_cache=True) - expectation) < 1e-10
    expectation = 0.6 * (0.6 * v00 + 0.4 * v01) + 0.4 * (0.6 * v10 + 0.4 * v11)
    assert abs(lct(0.104, log10_ens=2.4, use_cache=True) - expectation) < 1e-10
    # assert(np.all(lct([0.1,0.2],log10_ens=[2.4,3.2],use_cache=True)==lct([0.1,0.2],log10_ens=[2.5,3.5])))


def test_component_manipulation():
    # test sorting components
    lct = default_template()
    lct.order_primitives(order=0)  # sort by position, should swap them
    assert lct[0].get_location() == 0.48
    assert lct[1].get_location() == 0.50
    assert abs(lct[-1]()[0] - 0.35) < 1e-10
    assert abs(lct[-1]()[1] - 0.25) < 1e-10
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.01
    )
    lct.order_primitives(order=1)  # sort by amplitude, should swap them
    assert lct[0].get_location() == 0.48
    assert lct[1].get_location() == 0.50
    assert abs(lct.norms()[0] - 0.35) < 1e-10
    assert abs(lct.norms()[1] - 0.25) < 1e-10
    lct = lctemplate.get_gauss2(
        pulse_frac=0.6, x1=0.5, x2=0.48, ratio=0.25 / 0.35, width1=0.01, width2=0.05
    )
    lct.order_primitives(order=1)  # sort by amplitude, should do nothing
    assert lct.primitives[0].get_location() == 0.50
    assert lct.primitives[1].get_location() == 0.48
    assert abs(lct.norms()[1] - 0.35) < 1e-10
    assert abs(lct.norms()[0] - 0.25) < 1e-10
    lct.order_primitives(order=2)  # sort by norm, should swap
    assert lct.primitives[0].get_location() == 0.48
    assert lct.primitives[1].get_location() == 0.50
    assert abs(lct.norms()[0] - 0.35) < 1e-10
    assert abs(lct.norms()[1] - 0.25) < 1e-10

    # test adding components
    # intention is that new component is "norm" of the pulsed flux, so it
    # should be norm*0.6 total, here, while the other components are scaled
    # to preserve the same pulsed flux, i.e. by (1-norm)
    lct_add = lct.add_primitive(lcprimitives.LCGaussian(p=[0.02, 0.2]), norm=0.1)
    assert len(lct.primitives) == 2
    assert len(lct_add.primitives) == 3
    assert abs(lct_add.norms().sum() - 0.6) < 1e-10
    assert abs(lct_add.norms()[-1] - 0.1 * 0.6) < 1e-10
    assert abs(lct_add.norms()[0] - 0.35 * (1 - 0.1)) < 1e-10

    result = lct.add_primitive(
        lcprimitives.LCGaussian(p=[0.02, 0.2]), norm=0.1, inplace=True
    )
    assert result is None
    assert len(lct.primitives) == 3
    assert abs(lct.norms().sum() - 0.6) < 1e-10
    assert abs(lct.norms()[-1] - 0.1 * 0.6) < 1e-10
    assert abs(lct.norms()[0] - 0.35 * (1 - 0.1)) < 1e-10

    # test deletion -- should be back to original
    lct.delete_primitive(2, inplace=True)
    assert abs(lct.norms().sum() - 0.6) < 1e-10
    assert abs(lct.norms()[0] - 0.35) < 1e-10
    assert abs(lct.norms()[1] - 0.25) < 1e-10

    print("TODO: test energy-dependent slope of normalizations when re-ordering")


def test_energy_dependence():
    """Initial stab at testing energy-dependent profile modeling."""

    lcg = lcprimitives.LCGaussian(p=[0.03, 0.5])
    lcg2 = lcprimitives.LCGaussian(p=[0.04, 0.8])
    lct0 = lctemplate.LCTemplate([lcg, lcg2], [0.4, 0.35])
    lct0.add_energy_dependence(0, slope_free=False)
    assert not np.any(lct0.primitives[0].slope_free)
    assert lct0.num_parameters(free=True) == 6
    assert lct0.num_parameters(free=False) == 8
    lct0[0].slope_free[:] = True
    lct0[0].slope[:] = [0.02, 0.1]
    assert lct0.num_parameters(free=True) == 8
    lct0.add_energy_dependence(-1, slope_free=True)
    assert np.all(lct0.norms.slope_free)
    assert lct0.num_parameters(free=True) == 10
    lct0[-1].slope[1] = 0.2
    lct0[-1].slope_free[0] = False
    assert lct0.num_parameters(free=True) == 9

    # check re-ordering
    g1 = lceprimitives.LCEGaussian(p=[0.03, 0.5], slope=[0.01, 0.02])
    g2 = lceprimitives.LCEGaussian(p=[0.04, 0.8], slope=[-0.01, -0.02])
    lct2 = lctemplate.LCTemplate([g2, g1], [0.4, 0.35])
    lct2.add_energy_dependence(-1)
    lct2.norms.slope[0] = 0.1
    lct2.norms.slope[1] = -0.1
    assert lct2.primitives[1].p[-1] < lct2.primitives[0].p[-1]
    dom = np.linspace(0, 1, 101)
    ens = np.linspace(2, 4, 101)
    vals0 = lct2(dom, log10_ens=ens)
    lct2.order_primitives()
    assert lct2.primitives[1].p[-1] > lct2.primitives[0].p[-1]
    assert np.allclose(lct2(dom, log10_ens=ens), vals0)

    lcg = lceprimitives.LCEGaussian(p=[0.03, 0.5])
    assert np.all(lcg.slope == 0)
    assert np.all(lcg.slope_free == False)
    lcg = lceprimitives.LCEGaussian(
        p=[0.03, 0.5], slope=[0.02, 0.1], slope_free=[True, True]
    )
    assert np.all(lcg.slope == np.asarray([0.02, 0.1]))
    assert np.all(lcg.slope_free == True)
    lcg2 = lcprimitives.LCGaussian(p=[0.04, 0.8])
    lcn = lcenorm.ENormAngles([0.4, 0.35])
    assert not np.any(lcn.slope_free)  # default is False
    lcn.slope_free[1] = True
    lcn.slope[1] = 0.2
    lct = lctemplate.LCTemplate([lcg, lcg2], lcn)

    assert lct.check_gradient(quiet=True, seed=0)

    # set the seed to allow test in change in log likelihood
    np.random.seed(10)
    N = 1000
    log10_ens = np.random.rand(N) * 2 + 2
    we = np.ones(N)
    ph, comps = lct.random(N, weights=we, log10_ens=log10_ens, return_partition=True)

    assert np.all(lct(ph, log10_ens=log10_ens) == lct0(ph, log10_ens=log10_ens))
    # check the evaluation of the template within the gradient function
    g, t = lct.gradient(ph, log10_ens=log10_ens, template_too=True)
    assert np.allclose(t, lct(ph, log10_ens=log10_ens))

    lcf = lcfitters.LCFitter(lct, ph, weights=we, log10_ens=log10_ens)
    logl1 = lcf.loglikelihood(lct.get_parameters())
    lcf.fit(unbinned=True, use_gradient=True, quiet=True)
    logl2 = lcf.loglikelihood(lct.get_parameters())
    assert abs(abs(logl2 - logl1) - 5.15) < 0.05


def test_template_gradient():
    """Verify analytic gradient computation with numerical evaluation."""

    lct = default_template()

    # check that analytic gradient is equal to numerical gradient
    assert lct.check_gradient(quiet=True, seed=0)

    # check derivative too
    assert lct.check_derivative(order=1)
    # assert(lct.check_derivative(order=2))

    # check that nothing breaks when freezing a normalization angle
    lct.norms.free[0] = False
    assert lct.check_gradient(quiet=True, seed=0)

    # check that nothing breaks when freezing a primitive parameter
    lct[0].free[0] = False
    assert lct.check_gradient(quiet=True, seed=0)
    lct[0].free[:] = False
    assert lct.check_gradient(quiet=True, seed=0)


def test_fitter_gradient():
    # TODO -- idea here is to use a known template and an exact set of
    # phases/weights to make sure the gradient comes out correctly/close
    # in both unbinned and (?) binned cases.
    print("need to implement fitter gradient!")


def test_template_ebinning():
    # TODO -- verify that the binned version of an energy-dependent template
    # comes out close enough -- though maybe this should be under validation
    pass


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
    assert np.all(lcf.weights == 1)
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

    # check position fitting
    offset, error = lcf.fit_position()
    assert abs(offset < 1e-7)
    assert abs(error - 0.00055513) < 1e-7

    # check that fitting with a bad parameter boundary fails
    p = lcf.template.primitives[0]
    p.p[0] = p.bounds[0][0] - 1e-4
    assert not (p.check_bounds())
    with pytest.raises(ValueError):
        lcf.fit_position()


def test_simple_fit_binned():
    """Make sure objects adequately implement mathematical intent."""

    lct = lctemplate.get_gauss2()

    # load in simulated phases
    ph = np.loadtxt(datadir / "template_phases.asc")
    lcf = lcfitters.LCFitter(lct, ph, binned_bins=100)
    assert np.all(lcf.weights == 1)
    lcf.fit(unbinned=False, estimate_errors=True, quiet=True)
    expected_val = r"""
Log Likelihood for fit: 1104.39

Mixture Amplitudes
------------------
P1 : 0.5840 +\- 0.0220
P2 : 0.4160 +\- 0.0220
DC : 0.0000 +\- 0.0000

P1 -- Gaussian
------------------
Width   : 0.0099 +\- 0.0004
Location: 0.1007 +\- 0.0006

P2 -- Gaussian
------------------
Width   : 0.0209 +\- 0.0010
Location: 0.5493 +\- 0.0015

delta   : 0.1007 +\- 0.0006
Delta   : 0.4486 +\- 0.0016
"""

    assert expected_val.strip() == str(lcf).strip()


def test_vonmises():
    with pytest.raises(AssertionError):
        lceprimitives.LCEVonMises(p=[0.0001, 0.5], slope_free=[True, True, True])
    with pytest.raises(ValueError):
        lcprimitives.LCVonMises(p=[0.0001, 0.5])
    p1 = lcprimitives.LCVonMises(p=[0.05, 0.1])
    assert abs(p1(0.1 + p1.hwhm()) * 2 - p1(0.1)) < 1e-5
    p2 = lcprimitives.LCVonMises(p=[0.05, 0.4])

    def vm(x, p):
        width, loc = p
        x = np.asarray(x)
        z = (2 * np.pi) * (x - loc)
        scale = 1.0 / width
        return np.exp(np.cos(z) * scale) * (1.0 / i0(scale))

    en = np.random.rand(1000) * 2 + 2  # 2-4 range
    ph = np.random.rand(1000)
    assert np.allclose(p1(ph, log10_ens=en), vm(ph, [0.05, 0.1]))

    lct = lctemplate.LCTemplate([p1, p2], [0.4, 0.6])
    assert lct.check_gradient(quiet=True, seed=0)
    assert lct.check_derivative(order=1)
    # assert(lct.check_derivative(order=2))
    p1.free[1] = False
    assert p1.gradient(ph[:10], free=True).shape[0] == 1

    p1 = lceprimitives.LCEVonMises(p=[0.05, 0.1], slope=[0, 0.1])
    p2 = lceprimitives.LCEVonMises(p=[0.05, 0.4], slope=[0.05, 0.05])
    assert np.all(p2.slope == 0.05)
    assert np.allclose(p1(ph, log10_ens=en), vm(ph, [0.05, 0.1 + (en - 3) * 0.1]))
    lct = lctemplate.LCTemplate([p1, p2], [0.4, 0.6])
    assert lct.is_energy_dependent()
    assert p1.gradient(ph[:10], free=True).shape[0] == 2
    assert p1.gradient(ph[:10], free=False).shape[0] == 4
    assert lct.check_gradient(quiet=True, seed=0)
    assert lct.check_derivative()

    # check integral
    ens = np.linspace(2, 4, 21)
    integrals = p1.integrate(0, 0.5, log10_ens=ens)
    quads = [
        quad(lambda x: p1(x, log10_ens=ens[i]), 0, 0.5)[0] for i in range(len(ens))
    ]
    assert np.allclose(quads, integrals)

    # check overflow
    p1 = lceprimitives.LCEVonMises()
    p1.p[0] = p1.bounds[0][0]
    assert not np.isnan(p1(p1.p[-1]))

    # check fast Bessel function
    q = np.logspace(-1, np.log10(700), 201)
    I0 = lcprimitives.FastBessel(order=0)
    I1 = lcprimitives.FastBessel(order=1)
    assert np.all(np.abs(I0(q) / i0(q) - 1) < 1e-2)
    assert np.all(np.abs(I1(q) / i1(q) - 1) < 1e-2)


def test_skewnorm():
    # check bounds and limits
    with pytest.raises(ValueError):
        lcprimitives.LCSkewGaussian(p=[0.0001, 0.1, 0.5])
    with pytest.raises(ValueError):
        lcprimitives.LCSkewGaussian(p=[0.0001, 200, 0.5])
    p1 = lcprimitives.LCSkewGaussian(p=[0.05, 0.0, 0.1])
    pg = lcprimitives.LCGaussian(p=[0.05, 0.1])
    assert p1(0.1) == pg(0.1)
    assert p1(0.5) == pg(0.5)

    # TODO -- check hwhm
    # assert(abs(p1(0.1+p1.hwhm())*2-p1(0.1))<1e-5)

    # check function evaluation
    p1 = lcprimitives.LCSkewGaussian(p=[0.05, -1.0, 0.5])

    def sn(x, p, index=0):
        width, shape, loc = p
        x = np.asarray(x) + index
        z = (x - loc) / width
        t1 = 1.0 / ((2 * np.pi) ** 0.5 * width) * np.exp(-0.5 * z**2)
        t2 = 1 + erf(shape * z / 2**0.5)
        return t1 * t2

    en = np.random.rand(1000) * 2 + 2  # 2-4 range
    ph = np.random.rand(1000)
    assert np.allclose(p1(ph, log10_ens=en), sn(ph, p1.p))

    p1 = lcprimitives.LCSkewGaussian(p=[0.05, -1.0, 0.1])
    p2 = lcprimitives.LCSkewGaussian(p=[0.05, 2.0, 0.4])
    lct = lctemplate.LCTemplate([p1, p2], [0.4, 0.6])
    assert lct.check_gradient(quiet=True, seed=0)
    assert lct.check_derivative()

    # @abhisrkckl : There was a return statement above which made the code below unreachable.
    # I have removed the return statement and commented out the below code.
    # The following tests don't actually work. I am not sure why.

    # p1.free[1] = False
    # assert p1.gradient(ph[:10], free=True).shape[0] == 1

    # p1 = lceprimitives.LCEVonMises(p=[0.05, 0.1], slope=[0, 0.1])
    # p2 = lceprimitives.LCEVonMises(p=[0.05, 0.4], slope=[0.05, 0.05])
    # assert np.all(p2.slope == 0.05)
    # assert np.allclose(p1(ph, log10_ens=en), vm(ph, [0.05, 0.1 + (en - 3) * 0.1]))
    # lct = lctemplate.LCTemplate([p1, p2], [0.4, 0.6])
    # assert lct.is_energy_dependent()
    # assert p1.gradient(ph[:10], free=True).shape[0] == 2
    # assert p1.gradient(ph[:10], free=False).shape[0] == 4
    # assert lct.check_gradient(quiet=True, seed=0)
    # assert lct.check_derivative()

    # # check integral
    # ens = np.linspace(2, 4, 21)
    # integrals = p1.integrate(0, 0.5, log10_ens=ens)
    # quads = [
    #     quad(lambda x: p1(x, log10_ens=ens[i]), 0, 0.5)[0] for i in range(len(ens))
    # ]
    # assert np.allclose(quads, integrals)

    # # check overflow
    # p1 = lceprimitives.LCEVonMises()
    # p1.p[0] = p1.bounds[0][0]
    # assert not np.isnan(p1(p1.p[-1]))

    # # check fast Bessel function
    # q = np.logspace(-1, np.log10(700), 201)
    # I0 = lcprimitives.FastBessel(order=0)
    # I1 = lcprimitives.FastBessel(order=1)
    # assert np.all(np.abs(I0(q) / i0(q) - 1) < 1e-2)
    # assert np.all(np.abs(I1(q) / i1(q) - 1) < 1e-2)
