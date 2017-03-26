#!/usr/bin/env python
from __future__ import division, print_function
import sys, os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest
import numpy as np
import pint.scripts.pintempo as pintempo
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'NGC6440E.par')
timfile = os.path.join(datadir, 'NGC6440E.tim')


class TestPintempo(unittest.TestCase):

    def test_result(self):
        saved_stdout, pintempo.sys.stdout = pintempo.sys.stdout, StringIO('_')
        cmd = '{0} {1}'.format(parfile,timfile)
        pintempo.main(cmd.split())
        lines = pintempo.sys.stdout.getvalue()
        v = 999.0
        for l in lines.split('\n'):
            if l.startswith('RMS in time is'):
                v = float(l.split()[4])
        # Check that RMS is less than 34 microseconds
        self.assertTrue(v<34.0)
        pintempo.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
