#!/usr/bin/env python
from __future__ import division, print_function
import sys, os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest
import numpy as np
import pint.scripts.photonphase as photonphase
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'J1513-5908_PKS_alldata_white.par')
eventfile = os.path.join(datadir, 'B1509_RXTE_short.fits')
orbfile = os.path.join(datadir, 'FPorbit_Day6223')

class TestPhotonPhase(unittest.TestCase):

    def test_result(self):
        saved_stdout, photonphase.sys.stdout = photonphase.sys.stdout, StringIO('_')
        cmd = '--plot --plotfile photontest.png --outfile photontest.fits {0} {1} --orbfile={2} '.format(eventfile,parfile,orbfile)
        photonphase.main(cmd.split())
        lines = photonphase.sys.stdout.getvalue()
        v = 999.0
        for l in lines.split('\n'):
            if l.startswith('Htest'):
                v = float(l.split()[2])
        # Check that H-test is greater than 725
        self.assertTrue(v>725)
        photonphase.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
