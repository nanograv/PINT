import os
import unittest

from io import StringIO

from pint import toa
from pint.observatory import bipm_default
from pint.models import get_model, get_model_and_toas
from pinttestdata import datadir

# For this test, turn off the check for the age of the IERS A table
from astropy.utils.iers import conf

conf.auto_max_age = None

simplepar = """
PSR              1748-2021E
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""


class TestTOAReader(unittest.TestCase):
    def setUp(self):
        os.chdir(datadir)
        self.x = toa.TOAs("test1.tim")
        self.x.apply_clock_corrections()
        self.x.compute_TDBs()
        self.x.table.sort("index")

    def test_read_parkes(self):
        ts = toa.get_TOAs("parkes.toa")
        assert "arecibo" in ts.observatories
        assert ts.ntoas == 8

    def test_commands(self):
        assert len(self.x.commands) == 18

    def test_count(self):
        assert self.x.ntoas == 9

    def test_info(self):
        assert self.x.table[0]["flags"]["info"] == "test1"

    def test_jump(self):
        assert self.x.table[0]["flags"]["jump"] == 0

    def test_info_2(self):
        assert self.x.table[3]["flags"]["info"] == "test2"

    def test_time(self):
        assert self.x.table[3]["flags"]["to"] == 1.0

    def test_jump_2(self):
        assert "jump" not in self.x.table[4]["flags"]

    def test_time_2(self):
        assert "time" not in self.x.table[4]["flags"]

    def test_jump_3(self):
        assert self.x.table[-1]["flags"]["jump"] == 1

    def test_obs(self):
        assert self.x.table[1]["obs"] == "gbt"

    def test_ephem(self):
        assert self.x.ephem == "DE421"

    def test_planets(self):
        assert self.x.planets is False

    def test_clock(self):
        assert self.x.clock_corr_info["bipm_version"] == bipm_default


def test_model_override1():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    y = toa.get_TOAs("test1.tim", model=m)
    assert y.ephem == "DE436"
    assert y.planets is True
    assert y.clock_corr_info["bipm_version"] == "BIPM2017"


def test_model_override2():
    parstr = StringIO(simplepar)
    m, y = get_model_and_toas(parstr, "test1.tim")
    assert y.ephem == "DE436"
    assert y.planets is True
    assert y.clock_corr_info["bipm_version"] == "BIPM2017"


def test_model_override_override2():
    parstr = StringIO(simplepar)
    m, y = get_model_and_toas(parstr, "test1.tim", ephem="DE405")
    assert y.ephem == "DE405"


def test_model_override_override():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    y = toa.get_TOAs(
        "test1.tim",
        model=m,
        ephem="DE405",
        planets=False,
        include_bipm=True,
        bipm_version="BIPM2012",
    )
    assert y.ephem == "DE405"
    assert y.planets is False
    assert y.clock_corr_info["bipm_version"] == "BIPM2012"


def test_tttai():
    # This checks to make sure that CLOCK TT(TAI) is handled correctly
    parstr = StringIO(simplepar.replace("BIPM2017", "TAI"))
    m = get_model(parstr)
    y = toa.get_TOAs("test1.tim", model=m)
    assert y.clock_corr_info["include_bipm"] == False
