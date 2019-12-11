#!/usr/bin/env python
# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO
from __future__ import division, print_function

import os
import sys
import unittest

from six import StringIO

from pint.scripts import event_optimize
from pinttestdata import datadir

parfile = os.path.join(datadir, "PSRJ0030+0451_psrcat.par")
eventfile = os.path.join(
    datadir, "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
)
temfile = os.path.join(datadir, "templateJ0030.3gauss")


class TestEventOptimize(unittest.TestCase):
    def test_result(self):

        saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))
        cmd = "{0} {1} {2} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10".format(
            eventfile, parfile, temfile
        )
        event_optimize.main(cmd.split())
        lines = sys.stdout.getvalue()
        # Need to add some check here.
        sys.stdout = saved_stdout


if __name__ == "__main__":
    unittest.main()
