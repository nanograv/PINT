#!/usr/bin/env python
import unittest
import numpy as np
import os
import pint.models
import pint.toa
from pint.residuals import resids
from pinttestdata import testdir, datadir

import astropy.units as u

parfile = os.path.join(datadir, 'withpn.par')
timfile = os.path.join(datadir, 'withpn.tim')

class TestPulseNumber(unittest.TestCase):
    def test_pulse_number(self):
        model = pint.models.get_model(parfile)
        toas = pint.toa.get_TOAs(timfile)
        track_resids = resids(toas, model).time_resids

        self.assertFalse(np.max(track_resids) < 0.2 * u.second)


        getattr(model, 'TRACK').value = '0'
        notrack_resids = resids(toas, model).time_resids

        self.assertTrue(np.max(notrack_resids) < 0.2 * u.second)
