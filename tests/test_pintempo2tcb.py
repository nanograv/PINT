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

parfile = os.path.join(datadir, 'test_TCB.par')
timfile = os.path.join(datadir, 'test_TCB.simulate')

"""
Note: test_TCB.simulate has been produced using TEMPO2, with the fake plugin,
using:

tempo2 -gr fake -start 56990 -end 57355 -f test_TCB.par -ndobs 1 -nobsd 1 -ha 6 -randha n -rms 0.00000001
"""

class TestPintempo2tcb(unittest.TestCase):

    def test_result(self):
        saved_stdout, pintempo.sys.stdout = pintempo.sys.stdout, StringIO('_')
        cmd = '{0} {1}'.format(parfile,timfile)
        pintempo.main(cmd.split())
        lines = pintempo.sys.stdout.getvalue()
        v = 999.0
        for l in lines.split('\n'):
            if l.startswith('RMS in time is'):
                v = float(l.split()[4])
        # Check that RMS is less than 10 microseconds
        from astropy import log
        log.warning('%f' % v)
        print(v)
        self.assertTrue(v<10.0)
        pintempo.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
