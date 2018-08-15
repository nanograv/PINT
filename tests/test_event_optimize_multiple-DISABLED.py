#!/usr/bin/env python
# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO
from __future__ import division, print_function
import sys, os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest
import numpy as np
import pint.scripts.event_optimize_multiple as event_optimize_multiple
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'PSRJ0030+0451_psrcat.par')
eventfile = os.path.join(datadir, 'evtfiles.txt')

class TestEventOptimizeMultiple(unittest.TestCase):

    def test_result(self):
        saved_stdout, event_optimize_multiple.sys.stdout = event_optimize_multiple.sys.stdout, StringIO('_')
        cmd = '{0} {1} --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10'.format(eventfile,parfile)
        event_optimize_multiple.main(cmd.split())
        lines = event_optimize_multiple.sys.stdout.getvalue()
        # Need to add some check here.
        event_optimize_multiple.sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
