#!/usr/bin/env python
from __future__ import division, print_function

import os
import unittest

import pytest
from astropy import log
from six import StringIO

import pint.scripts.photonphase as photonphase
from pinttestdata import datadir

parfile = os.path.join(datadir, "J1513-5908_PKS_alldata_white.par")
eventfile = os.path.join(datadir, "B1509_RXTE_short.fits")
orbfile = os.path.join(datadir, "FPorbit_Day6223")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_result():
    "Test that processing RXTE data with orbit file gives correct result"
    saved_stdout, photonphase.sys.stdout = photonphase.sys.stdout, StringIO("_")
    cmd = "--plot --plotfile photontest.png --outfile photontest.fits {0} {1} --orbfile={2} ".format(
        eventfile, parfile, orbfile
    )
    photonphase.main(cmd.split())
    lines = photonphase.sys.stdout.getvalue()
    v = 0.0
    for l in lines.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is greater than 725
    log.warning("V:\t%f" % v)
    assert v > 725
    photonphase.sys.stdout = saved_stdout


parfile_nicer = os.path.join(datadir, "ngc300nicer.par")
parfile_nicerbad = os.path.join(datadir, "ngc300nicernoTZR.par")
eventfile_nicer = os.path.join(datadir, "ngc300nicer_bary.evt")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_nicer_result():
    "Check that barycentered NICER data is processed correctly."
    saved_stdout, photonphase.sys.stdout = photonphase.sys.stdout, StringIO("_")
    cmd = "{0} {1}".format(eventfile_nicer, parfile_nicer)
    photonphase.main(cmd.split())
    lines = photonphase.sys.stdout.getvalue()
    v = 0.0
    for l in lines.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is greater than 725
    log.warning("V:\t%f" % v)
    assert v > 200.0
    photonphase.sys.stdout = saved_stdout


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_AbsPhase_exception():
    "Verify that passing par file with no TZR* parameters raises exception"
    with pytest.raises(ValueError):
        saved_stdout, photonphase.sys.stdout = photonphase.sys.stdout, StringIO("_")
        cmd = "{0} {1}".format(eventfile_nicer, parfile_nicerbad)
        photonphase.main(cmd.split())


if __name__ == "__main__":
    unittest.main()
