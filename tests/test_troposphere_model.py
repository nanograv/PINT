#!/usr/bin/env python

import os
import unittest

import astropy.units as u
import numpy as np
import pint.models as models
import pint.observatory
import pint.toa as toa
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from pint.models.troposphere_delay import TroposphereDelay
from pint.observatory import get_observatory


class TestTroposphereDelay(unittest.TestCase):

    MIN_ALT = 5  # the minimum altitude in degrees for testing the delay model

    def setUp(self):
        self.toas = toa.get_TOAs(ngc + ".tim")
        self.model = pint.models.get_model(ngc + ".par")

        self.testAltitudes = np.arange(self.MIN_ALT, 90, 100) * u.deg
        self.testHeights = np.array([10, 100, 1000, 5000]) * u.m

        self.td = TroposphereDelay()

    def test_altitudes(self):
        # the troposphere delay should strictly decrease with increasing altitude above horizon
        delays = self.td.delay_model(self.testAltitudes, 0, 1000 * u.m, 0)

        for i in range(1, len(delays)):
            assert delays[i] < delays[i - 1]

    def test_heights(self):
        # higher elevation observatories should have less troposphere delay
        heightDelays = []  # list of the delays at each height
        for h in self.testHeights:
            heightDelays.append(self.td.delay_model(self.testAltitudes, 0, h, 0))

        for i in range(1, len(testHeights)):
            assert np.all(np.less(heightDelays[i], heightDelays[i - 1]))
