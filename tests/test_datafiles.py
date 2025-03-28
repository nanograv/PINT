"""Test installation of PINT data files"""

import os
import pytest
import pint.config
import tempfile


@pytest.fixture
def getfiles():
    # get a list of par and tim files to make sure we can find
    files = """B1855+09_NANOGrav_9yv1.gls.par
    B1855+09_NANOGrav_9yv1.tim
    B1855+09_NANOGrav_dfg+12.tim
    B1855+09_NANOGrav_dfg+12_TAI.par
    J0613-sim.par
    J0613-sim.tim
    J1614-2230_NANOGrav_12yv3.wb.gls.par
    J1614-2230_NANOGrav_12yv3.wb.tim
    NGC6440E.par
    NGC6440E.tim
    PSRJ0030+0451_psrcat.par
    waves.par
    waves_withpn.tim"""

    return [f.strip() for f in files.split("\n")]


def test_datadir():
    datadir = pint.config.datadir()
    assert os.path.isdir(datadir)


def test_examplefiles(getfiles):
    files = getfiles
    for filename in files:
        assert os.path.exists(pint.config.examplefile(filename))


def test_examplefiles_tempdir(getfiles):
    files = getfiles
    curdir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as dir:
            os.chdir(dir)
            for filename in files:
                assert os.path.exists(pint.config.examplefile(filename))
    finally:
        os.chdir(curdir)


def test_examplefiles_tempdir_moveafter(getfiles):
    files = getfiles
    curdir = os.getcwd()
    # get the full names before moving directories
    filenames = [pint.config.examplefile(filename) for filename in files]
    try:
        with tempfile.TemporaryDirectory() as dir:
            os.chdir(dir)
            for filename in filenames:
                assert os.path.exists(filename)
    finally:
        os.chdir(curdir)
