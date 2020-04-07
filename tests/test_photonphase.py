#!/usr/bin/env python
from __future__ import division, print_function

import os
import unittest

import pytest

import pint.scripts.photonphase as photonphase
from pinttestdata import datadir

parfile = os.path.join(datadir, "J1513-5908_PKS_alldata_white.par")
eventfile = os.path.join(datadir, "B1509_RXTE_short.fits")
orbfile = os.path.join(datadir, "FPorbit_Day6223")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_result(capsys):
    "Test that processing RXTE data with orbit file gives correct result"
    cmd = "--plot --plotfile photontest.png --outfile photontest.fits {0} {1} --orbfile={2} ".format(
        eventfile, parfile, orbfile
    )
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is greater than 725
    assert v > 725


parfile_nicer = os.path.join(datadir, "ngc300nicer.par")
parfile_nicerbad = os.path.join(datadir, "ngc300nicernoTZR.par")
eventfile_nicer = os.path.join(datadir, "ngc300nicer_bary.evt")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_nicer_result(capsys):
    "Check that barycentered NICER data is processed correctly."
    cmd = "{0} {1}".format(eventfile_nicer, parfile_nicer)
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is greater than 725
    assert v > 200.0


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_AbsPhase_exception():
    "Verify that passing par file with no TZR* parameters raises exception"
    with pytest.raises(ValueError):
        cmd = "{0} {1}".format(eventfile_nicer, parfile_nicerbad)
        photonphase.main(cmd.split())


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_OrbPhase_exception():
    "Verify that trying to add ORBIT_PHASE column with no BINARY parameter in the par file raises exception"
    with pytest.raises(ValueError):
        cmd = "--addorbphase {0} {1}".format(eventfile_nicer, parfile_nicer)
        photonphase.main(cmd.split())


if __name__ == "__main__":
    unittest.main()
