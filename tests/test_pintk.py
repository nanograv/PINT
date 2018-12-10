#!/usr/bin/env python
from __future__ import division, print_function
import sys, os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest
import numpy as np
import pint.scripts.pintk as pintk
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'NGC6440E.par')
timfile = os.path.join(datadir, 'NGC6440E.tim')

class TestPintk(unittest.TestCase):

    def test_result(self):
        saved_stdout, pintk.sys.stdout = pintk.sys.stdout, StringIO('_')
        cmd = '--test {0} {1}'.format(parfile,timfile)
        pintk.main(cmd.split())
        lines = pintk.sys.stdout.getvalue()
        pintk.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
