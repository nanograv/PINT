from __future__ import absolute_import, division, print_function
import os
import unittest
import pytest

from pint import toa
from pinttestdata import datadir
from astropy.time import Time


class TestRoundtripToFiles(unittest.TestCase):
    #   @classmethod
    #    def setUp(self):
    #        os.chdir(datadir)

    # Try writing and reading TEMPO and Tempo2 format .tim files
    # Use several observatories, including Topo and Bary
    def test_roundtrip_bary_toa_Tempo2format(self):
        # Create a barycentric TOA
        t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="tdb")
        t1 = toa.TOA(t1time, obs="Barycenter", freq=0.0)
        ts = toa.get_TOAs_list([t1], ephem="DE421")
        ts.write_TOA_file("testbary.tim", format="Tempo2")
        ts2 = toa.get_TOAs("testbary.tim")
        print(ts.table, ts2.table)
        assert ts.table["mjd"][0] - ts2.table["mjd"][0] < 1.0e-15
        assert ts.table["tdb"][0] - ts2.table["tdb"][0] < 1.0e-15

    def test_roundtrip_bary_toa_TEMPOformat(self):
        # Create a barycentric TOA
        t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="tdb")
        t1 = toa.TOA(t1time, obs="Barycenter", freq=0.0)
        ts = toa.get_TOAs_list([t1], ephem="DE421")
        ts.write_TOA_file("testbary.tim", format="TEMPO")
        ts2 = toa.get_TOAs("testbary.tim")
        print(ts.table, ts2.table)
        assert ts.table["mjd"][0] - ts2.table["mjd"][0] < 1.0e-15
        assert ts.table["tdb"][0] - ts2.table["tdb"][0] < 1.0e-15

    def test_roundtrip_topo_toa_Tempo2format(self):
        # Create a barycentric TOA
        t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
        t1 = toa.TOA(t1time, obs="gbt", freq=0.0)
        ts = toa.get_TOAs_list([t1], ephem="DE421")
        ts.write_TOA_file("testtopo.tim", format="Tempo2")
        ts2 = toa.get_TOAs("testtopo.tim")
        print(ts.table, ts2.table)
        assert ts.table["mjd"][0] - ts2.table["mjd"][0] < 1.0e-15
        assert ts.table["tdb"][0] - ts2.table["tdb"][0] < 1.0e-15

    def test_roundtrip_topo_toa_TEMPOformat(self):
        # Create a barycentric TOA
        t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
        t1 = toa.TOA(t1time, obs="gbt", freq=0.0)
        ts = toa.get_TOAs_list([t1], ephem="DE421")
        ts.write_TOA_file("testtopot1.tim", format="TEMPO")
        ts2 = toa.get_TOAs("testtopot1.tim")
        print(ts.table, ts2.table)
        assert ts.table["mjd"][0] - ts2.table["mjd"][0] < 1.0e-15
        assert ts.table["tdb"][0] - ts2.table["tdb"][0] < 1.0e-15

    # Comment out because TEMPO2 distro doesn't include gmrt2gps.clk so far
    # def test_roundtrip_gmrt_toa_Tempo2format(self):
    #     if os.getenv("TEMPO2") is None:
    #         pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
    #     # Create a barycentric TOA
    #     t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    #     t1 = toa.TOA(t1time, obs="gmrt", freq=0.0)
    #     ts = toa.get_TOAs_list([t1], ephem="DE421")
    #     ts.write_TOA_file("testgmrt.tim", format="Tempo2")
    #     ts2 = toa.get_TOAs("testgmrt.tim")
    #     print(ts.table, ts2.table)
    #     assert ts.table["mjd"][0] - ts2.table["mjd"][0] < 1.0e-15
    #     assert ts.table["tdb"][0] - ts2.table["tdb"][0] < 1.0e-15

    # def test_roundtrip_ncyobs_toa_Tempo2format(self):
    #     if os.getenv("TEMPO2") is None:
    #         pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
    #     # Create a barycentric TOA
    #     t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    #     t1 = toa.TOA(t1time, obs="ncyobs", freq=0.0)
    #     ts = toa.get_TOAs_list([t1], ephem="DE421")
    #     ts.write_TOA_file("testncyobs.tim", format="Tempo2")
    #     ts2 = toa.get_TOAs("testncyobs.tim")
    #     print(ts.table, ts2.table)
    #     assert ts.table["mjd"][0] - ts2.table["mjd"][0] < 1.0e-15
    #     assert ts.table["tdb"][0] - ts2.table["tdb"][0] < 1.0e-15

    # def test_roundtrip_ncyobs_toa_TEMPOformat(self):
    #     if os.getenv("TEMPO2") is None:
    #         pytest.skip("TEMPO2 evnironment variable is not set, can't run this test")
    #     # Create a barycentric TOA
    #     t1time = Time(58534.0, 0.0928602471130208, format="mjd", scale="utc")
    #     t1 = toa.TOA(t1time, obs="ncyobs", freq=0.0)
    #     ts = toa.get_TOAs_list([t1], ephem="DE421")
    #     # This is an observatory that can't be represented in TEMPO format
    #     # so it should raise an exception
    #     with pytest.raises(ValueError):
    #         ts.write_TOA_file("testncyobs.tim", format="TEMPO")
