# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO
import os
import sys
import pytest

from io import StringIO

from pinttestdata import datadir

parfile = os.path.join(datadir, "PSRJ0030+0451_psrcat.par")
eventfile = os.path.join(datadir, "evtfiles.txt")


@pytest.mark.skip
class TestEventOptimizeMultiple:
    def test_result(self):
        # Delay import because of fftfit
        from pint.scripts import event_optimize_multiple

        saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))
        cmd = "{0} {1} --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10".format(
            eventfile, parfile
        )
        event_optimize_multiple.main(cmd.split())
        lines = sys.stdout.getvalue()
        # Need to add some check here.
        sys.stdout = saved_stdout
