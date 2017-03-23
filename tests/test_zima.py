#!/usr/bin/env python
from __future__ import division, print_function
import sys, os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest
import numpy as np
import pint.scripts.zima as zima
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'NGC6440E.par')
timfile = os.path.join(datadir, 'fake_testzima.tim')

class TestZima(unittest.TestCase):

    def test_result(self):
        saved_stdout, zima.sys.stdout = zima.sys.stdout, StringIO('_')
        cmd = '{0} {1}'.format(parfile,timfile)
        zima.main(cmd.split())
        lines = zima.sys.stdout.getvalue()
        zima.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
