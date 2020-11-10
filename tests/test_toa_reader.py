import os
import unittest

from pint import toa
from pinttestdata import datadir

# For this test, turn off the check for the age of the IERS A table
from astropy.utils.iers import conf

conf.auto_max_age = None


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
