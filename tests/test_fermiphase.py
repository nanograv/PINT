#!/usr/bin/env python
from __future__ import division, print_function
import sys, os
from StringIO import StringIO
import unittest
import numpy as np
import pint.scripts.fermiphase as fermiphase
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'PSRJ0030+0451_psrcat.par')
eventfile = os.path.join(datadir, 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits')

# Can't easily test raw files since FT2 file is huge and can't be put in repo.
eventfileraw = os.path.joine(datadir, 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_weight.gt.0.4.fits')

class TestFermiPhase(unittest.TestCase):

    def test_result(self):
        saved_stdout, fermiphase.sys.stdout = fermiphase.sys.stdout, StringIO('_')
        cmd = '--plot --plotfile fermitest.png --outfile fermitest.fits {0} {1} CALC'.format(eventfile,parfile)
        fermiphase.main(cmd.split())
        lines = fermiphase.sys.stdout.getvalue()
        v = 999.0
        for l in lines.split('\n'):
            if l.startswith('Htest'):
                v = float(l.split()[2])
        # Check that H-test is greater than 1380
        self.assertTrue(v>1380)
        fermiphase.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
