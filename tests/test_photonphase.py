#!/usr/bin/env python
import os
import unittest

import pytest

import pint.scripts.photonphase as photonphase
from pinttestdata import datadir
from astropy.io import fits
import numpy as np

parfile = os.path.join(datadir, "J1513-5908_PKS_alldata_white.par")
eventfile = os.path.join(datadir, "B1509_RXTE_short.fits")
orbfile = os.path.join(datadir, "FPorbit_Day6223")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_rxte_result(capsys):
    "Test that processing RXTE data with orbit file gives correct result"
    cmd = "--minMJD 55576.640 --maxMJD 55576.645 --plot --plotfile photontest.png --outfile photontest.fits {0} {1} --orbfile={2} ".format(
        eventfile, parfile, orbfile
    )
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # H-test should be 87.5
    assert abs(v - 87.5) < 1


parfile_nicer = os.path.join(datadir, "ngc300nicer.par")
parfile_nicerbad = os.path.join(datadir, "ngc300nicernoTZR.par")
eventfile_nicer = os.path.join(datadir, "ngc300nicer_bary.evt")


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_nicer_result_bary(capsys):
    "Check that barycentered NICER data is processed correctly."
    cmd = "{0} {1}".format(eventfile_nicer, parfile_nicer)
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is 216.67
    assert abs(v - 216.67) < 1


parfile_nicer_topo = os.path.join(datadir, "sgr1830.par")
orbfile_nicer_topo = os.path.join(datadir, "sgr1830.orb")
eventfile_nicer_topo = os.path.join(datadir, "sgr1830kgfilt.evt")

# TODO -- should add explicit check for absolute phase
def test_nicer_result_topo(capsys):
    "Check that topocentric NICER data and orbit files are processed correctly."
    cmd = "--minMJD 59132.780 --maxMJD 59132.782 --orbfile {0} {1} {2}".format(
        orbfile_nicer_topo, eventfile_nicer_topo, parfile_nicer_topo
    )
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # H should be 183.21
    assert abs(v - 183.21) < 1


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


parfile_nicer_binary = os.path.join(datadir, "PSR_J0218+4232.par")
eventfile_nicer_binary = os.path.join(
    datadir, "J0218_nicer_2070030405_cleanfilt_cut_bary.evt"
)


@pytest.mark.skipif(
    "DISPLAY" not in os.environ, reason="Needs an X server, xvfb counts"
)
def test_OrbPhase_column():
    "Verify that the ORBIT_PHASE column is calculated and added correctly"

    outfile = "photonphase-test.evt"
    cmd = "--addorbphase --outfile {0} {1} {2}".format(
        outfile, eventfile_nicer_binary, parfile_nicer_binary
    )
    photonphase.main(cmd.split())

    # Check that output file got made
    assert os.path.exists(outfile)
    # Check that column got added
    hdul = fits.open(outfile)
    cols = hdul[1].columns.names
    assert "ORBIT_PHASE" in cols
    # Check that first and last entry have expected values and values are monotonic
    data = hdul[1].data
    orbphases = data["ORBIT_PHASE"]
    assert abs(orbphases[0] - 0.1763) < 0.0001
    assert abs(orbphases[-1] - 0.3140) < 0.0001
    assert np.all(np.diff(orbphases) > 0)

    hdul.close()
    os.remove(outfile)


if __name__ == "__main__":
    unittest.main()
