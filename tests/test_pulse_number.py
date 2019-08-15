#!/usr/bin/python
from __future__ import absolute_import

import unittest
import numpy as np
import os
import pint.models
import pint.toa
from pint.residuals import Residuals
from pinttestdata import testdir, datadir

from astropy import units as u

parfile = os.path.join(datadir, 'withpn.par')
timfile = os.path.join(datadir, 'withpn.tim')

class TestPulseNumber(unittest.TestCase):
    def test_pulse_number(self):
        model = pint.models.get_model(parfile)
        toas = pint.toa.get_TOAs(timfile)
        print(toas.table['flags'])
        print(toas.table.colnames)
        #Make sure pn table column was added
        self.assertTrue('pn' in toas.table.colnames)

        #Tracking pn should result in runaway residuals
        track_resids = Residuals(toas, model).time_resids
        self.assertFalse(np.max(track_resids) < 0.2 * u.second)

        #Not tracking pn should keep residuals bounded
        getattr(model, 'TRACK').value = '0'
        notrack_resids = Residuals(toas, model).time_resids
        self.assertTrue(np.max(notrack_resids) < 0.2 * u.second)

        #Make sure Exceptions are thrown when trying to track nonexistent pn
        del toas.table['pn']
        getattr(model, 'TRACK').value = '-2'
        self.assertRaises(Exception, Residuals, toas, model)

        #Make sure pn can be added back by using the model
        self.assertTrue(toas.get_pulse_numbers() is None)
        toas.compute_pulse_numbers(model)
        self.assertTrue('pn' in toas.table.colnames)
