import logging
import os
import unittest

import numpy as np

from pint.templates import lcprimitives
from pint.templates import lctemplate


def gauss(x, x0, s):
    return 1.0 / s / (2 * np.pi) ** 0.5 * np.exp(-0.5 * (x - x0) ** 2 / s ** 2)


class TestPrimitives(unittest.TestCase):
    """ Test instantiation and various properties of LCPrimitive objects.
    """

    @classmethod
    def setUpClass(cls):
        pass

    def test_basic_functionality(self):
        """ Make sure objects adequately implement mathematical intent."""

        # instantiate a basic gaussian with known parameters and make sure
        # it "looks" like a gaussian.  The main thing here is that these
        # functions are wrapped.

        g1 = lcprimitives.LCGaussian(p=[0.01, 0.5])
        # defined as 1/sigma/(2*pi)^0.5 * (-0.5*(x-x0)^2) WRAPPED
        # because this is a narrow gaussian, it's essentially the same as
        # an unwrapped version
        expected_val = gauss(0.50, 0.5, 0.01)
        assert abs(g1(0.50) - expected_val) < 1e-6
        expected_val = gauss(0.48, 0.5, 0.01)
        assert abs(g1(0.48) - expected_val) < 1e-6

        # explicitly test wrapping
        g1 = lcprimitives.LCGaussian(p=[0.5, 0.5])
        expected_val = gauss(0.5, 0.5, 0.5)
        for i in range(1, 11):
            expected_val += gauss(0.5 + i, 0.5, 0.5)
            expected_val += gauss(0.5 - i, 0.5, 0.5)
        assert abs(g1(0.5) - expected_val) < 1e-6


class TestTemplates(unittest.TestCase):
    """ Test instantiation and various properties of LCTemplate objects.
    """

    @classmethod
    def setUpClass(cls):
        pass

    def test_basic_functionality(self):
        """ Make sure objects adequately implement mathematical intent."""

        # a template should *always* have total normalization set to 1
        g2 = lcprimitives.LCGaussian(p=[0.01, 0.50])
        g1 = lcprimitives.LCGaussian(p=[0.01, 0.48])
        lct = lctemplate.LCTemplate([g1, g2], [0.25, 0.35])

        assert abs(lct.norm() - (0.25 + 0.35)) < 1e-10

        expected_val = (
            0.25 * gauss(0.49, 0.5, 0.01)
            + 0.35 * gauss(0.49, 0.48, 0.01)
            + (1 - 0.25 - 0.35)
        )
        assert abs(lct(0.49) - expected_val) < 1e-6
