import os
import pytest

import numpy as np
from scipy.stats import norm

import pint.models

# from pint.models.priors import *
from pint.models.priors import (
    Prior,
    UniformBoundedRV,
    GaussianBoundedRV,
    RandomInclinationPrior,
)
from pinttestdata import datadir


class TestPriors:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.m = pint.models.get_model("B1855+09_NANOGrav_dfg+12_modified.par")

    def test_uniform(self):
        print("test_uniform")
        # Call prior_pdf using parameter values set in model
        assert self.m.F0.prior_pdf() == 1.0
        assert self.m.F0.prior_pdf(logpdf=True) == 0.0

        # Now try overriding the parameter values
        assert self.m.F0.prior_pdf(10.0) == 1.0
        assert self.m.F0.prior_pdf(10.0, logpdf=True) == 0.0

    def test_uniform_bounded(self):
        print("test_uniform_bounded")
        self.m.ECC.prior = Prior(UniformBoundedRV(0.0, 1.0))
        assert self.m.ECC.prior_pdf() == 1.0
        assert self.m.ECC.prior_pdf(logpdf=True) == 0.0
        assert self.m.ECC.prior_pdf(-0.1) == 0.0
        assert self.m.ECC.prior_pdf(-0.1, logpdf=True) == -np.inf
        assert self.m.ECC.prior_pdf(1.1) == 0.0
        assert self.m.ECC.prior_pdf(1.1, logpdf=True) == -np.inf

    def test_gaussian(self):
        print("test_gaussian")
        v = -6.2e-16
        s = 1.0e-18
        self.m.F1.value
        self.m.F1.prior = Prior(norm(loc=v, scale=s))
        print(self.m.F1.prior_pdf(v))
        assert np.isclose(self.m.F1.prior_pdf(v), 1.0 / (s * np.sqrt(2.0 * np.pi)))

    def test_inclination_prior(self):
        self.m.SINI.prior = Prior(RandomInclinationPrior())
        v = 0.5
        # Check the prior PDF matches hand computation
        assert np.isclose(self.m.SINI.prior_pdf(v), 0.5773502691896258)
        # Check that np.exp(logpdf) == pdf
        assert np.isclose(self.m.SINI.prior.pdf(v), np.exp(self.m.SINI.prior.logpdf(v)))
        v = 0.9
        # Check the prior PDF matches hand computation
        assert np.isclose(self.m.SINI.prior_pdf(v), 2.0647416048350564)
        # Check that np.exp(logpdf) == pdf
        assert np.isclose(self.m.SINI.prior.pdf(v), np.exp(self.m.SINI.prior.logpdf(v)))

        v = np.array([0.5, 0.9])
        assert np.allclose(
            self.m.SINI.prior.pdf(v), np.exp(self.m.SINI.prior.logpdf(v))
        )

    def test_gaussian_bounded(self):
        print("test_gaussian_bounded")
        self.m.M2.prior = Prior(
            GaussianBoundedRV(loc=0.26, scale=0.10, lower_bound=0.0, upper_bound=0.6)
        )
        assert self.m.M2.prior_pdf(-0.1) == 0.0
        assert self.m.M2.prior_pdf(0.7) == 0.0
        assert self.m.M2.prior_pdf(-0.1, logpdf=True) == -np.inf
        assert self.m.M2.prior_pdf(0.7, logpdf=True) == -np.inf
        assert np.isclose(
            self.m.M2.prior_pdf(0.26 + 0.1) / self.m.M2.prior_pdf(0.26),
            0.60653065971263342,
        )
        # Test that integral is 1.0, not safe since using _rv private var
        assert np.isclose(self.m.M2.prior._rv.cdf(0.6), 1.0)
