#! /usr/bin/env python
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import matplotlib.pyplot as plt
import astropy.units as u
import sys
import os
import unittest

from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'prefixtest.par')
timfile = os.path.join(datadir, 'prefixtest.tim')


class TestPrefix(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = pint.models.get_model(parfile)
        self.t = pint.toa.get_TOAs(timfile,ephem="DE405")
    def test_prefix(self):
        print "Test prefix parameter via a glitch model"
        rs = pint.residuals.resids(self.t, self.m).phase_resids
        # Now do the fit
        print "Fitting..."
        f = pint.fitter.fitter(self.t, self.m)
        f.call_minimize()
        emsg = "RMS of " + self.m.PSR.value + " is too big."
        assert f.resids.time_resids.std().to(u.us).value < 950.0, emsg


if __name__ == '__main__':
    pass
