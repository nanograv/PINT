# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO
import os
import pytest

from io import StringIO

from pinttestdata import datadir

parfile = os.path.join(datadir, "PSRJ0030+0451_psrcat.par")
eventfile = os.path.join(
    datadir, "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
)
temfile = os.path.join(datadir, "templateJ0030.3gauss")


# SMR skipped this test as event_optimize_MCMCFitter isn't used anywhere
# How/why is it different from event_optimize?
@pytest.mark.skip
class TestEventOptimizeMCMCFitter:
    def test_result(self):
        import pint.scripts.event_optimize_MCMCFitter as event_optimize

        saved_stdout, event_optimize.sys.stdout = (
            event_optimize.sys.stdout,
            StringIO("_"),
        )
        cmd = "{0} {1} {2} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10".format(
            eventfile, parfile, temfile
        )
        # cmd = '{0} {1} {2} --nwalkers=14 --nsteps=50 --burnin 10'.format(eventfile,parfile,temfile)
        event_optimize.main(cmd.split())
        lines = event_optimize.sys.stdout.getvalue()
        # Need to add some check here.
        event_optimize.sys.stdout = saved_stdout
