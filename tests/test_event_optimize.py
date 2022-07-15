#!/usr/bin/env python
# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO
# Actually it's not disabled? Unclear what the above is supposed to mean.
import os
import shutil
import sys
from io import StringIO
from pathlib import Path

import pytest
from pinttestdata import datadir

from pint.scripts import event_optimize


def test_result(tmp_path):
    parfile = datadir / "PSRJ0030+0451_psrcat.par"
    eventfile_orig = (
        datadir
        / "J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits"
    )
    temfile = datadir / "templateJ0030.3gauss"
    eventfile = tmp_path / "event.fits"
    # We will write a pickle next to this file, let's make sure it's not under tests/
    shutil.copy(eventfile_orig, eventfile)

    p = Path.cwd()
    saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))
    try:
        os.chdir(tmp_path)
        cmd = f"{eventfile} {parfile} {temfile} --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=10 --nsteps=50 --burnin 10"
        event_optimize.main(cmd.split())
        lines = sys.stdout.getvalue()
        # Need to add some check here.
    finally:
        os.chdir(p)
        sys.stdout = saved_stdout
