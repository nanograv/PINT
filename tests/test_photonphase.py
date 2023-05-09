import os

import pytest

import pint.scripts.photonphase as photonphase
from pinttestdata import datadir
from astropy.io import fits
import numpy as np

parfile = datadir / "J1513-5908_PKS_alldata_white.par"
eventfile = datadir / "B1509_RXTE_short.fits"
orbfile = datadir / "FPorbit_Day6223"


def test_rxte_result(capsys, tmp_path):
    "Test that processing RXTE data with orbit file gives correct result"

    plotfile = tmp_path / "photontest.png"
    outfile = tmp_path / "photontest.fits"
    cmd = f"--minMJD 55576.640 --maxMJD 55576.645 --plot --plotfile {plotfile} --outfile {outfile} {eventfile} {parfile} --orbfile={orbfile} "
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # H-test should be 87.5
    assert abs(v - 87.5) < 1


parfile_nicer = datadir / "ngc300nicer.par"
parfile_nicerbad = datadir / "ngc300nicernoTZR.par"
eventfile_nicer = datadir / "ngc300nicer_bary.evt"


def test_nicer_result_bary(capsys):
    "Check that barycentered NICER data is processed correctly."
    cmd = f" {eventfile_nicer} {parfile_nicer}"
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is 216.67
    assert abs(v - 216.67) < 1


parfile_nicer_topo = datadir / "sgr1830.par"
orbfile_nicer_topo = datadir / "sgr1830.orb"
eventfile_nicer_topo = datadir / "sgr1830kgfilt.evt"


# TODO -- should add explicit check for absolute phase
def test_nicer_result_topo(capsys):
    "Check that topocentric NICER data and orbit files are processed correctly."
    cmd = f"--minMJD 59132.780 --maxMJD 59132.782 --orbfile {orbfile_nicer_topo} {eventfile_nicer_topo} {parfile_nicer_topo}"
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # H should be 183.21
    assert abs(v - 183.21) < 1


def test_AbsPhase_exception():
    "Verify that passing par file with no TZR* parameters raises exception"
    with pytest.raises(ValueError):
        cmd = "{0} {1}".format(eventfile_nicer, parfile_nicerbad)
        photonphase.main(cmd.split())


def test_OrbPhase_exception():
    "Verify that trying to add ORBIT_PHASE column with no BINARY parameter in the par file raises exception"
    with pytest.raises(ValueError):
        cmd = f"--addorbphase {eventfile_nicer} {parfile_nicer}"
        photonphase.main(cmd.split())


parfile_nicer_binary = datadir / "PSR_J0218+4232.par"
eventfile_nicer_binary = datadir / "J0218_nicer_2070030405_cleanfilt_cut_bary.evt"


def test_OrbPhase_column(tmp_path):
    "Verify that the ORBIT_PHASE column is calculated and added correctly"

    outfile = tmp_path / "photonphase-test.evt"
    cmd = f"--addorbphase --outfile {outfile} {eventfile_nicer_binary} {parfile_nicer_binary}"
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


def test_nicer_result_bary_polyco(capsys, tmp_path):
    "Check that barycentered NICER data is processed correctly with --polyco."
    outfile = tmp_path / "polycos-photonphase-test.evt"
    cmd = f"--polycos --outfile {outfile} {eventfile_nicer} {parfile_nicer}"
    photonphase.main(cmd.split())
    out, err = capsys.readouterr()
    v = 0.0
    for l in out.split("\n"):
        if l.startswith("Htest"):
            v = float(l.split()[2])
    # Check that H-test is 216.67
    assert abs(v - 216.67) < 1

    # Check that the phases are the same with and without polycos
    assert os.path.exists(outfile)
    hdul = fits.open(outfile)
    cols = hdul[1].columns.names
    assert "PULSE_PHASE" in cols
    data = hdul[1].data
    phases = data["PULSE_PHASE"]

    outfile2 = tmp_path / "photonphase-test.evt"
    cmd2 = "--outfile {0} {1} {2}".format(outfile2, eventfile_nicer, parfile_nicer)
    photonphase.main(cmd2.split())
    assert os.path.exists(outfile2)
    hdul2 = fits.open(outfile2)
    cols2 = hdul2[1].columns.names
    assert "PULSE_PHASE" in cols2
    data2 = hdul2[1].data
    phases2 = data2["PULSE_PHASE"]

    assert (phases - phases2).std() < 0.00001
