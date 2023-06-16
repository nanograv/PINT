"""Various tests to assess the performance of the PINT position.
"""
import os
import pytest

import numpy as np

import pint.models.model_builder as mb
from pinttestdata import datadir


class TestPulsarPosition:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        # This uses ELONG and ELAT
        cls.m1 = mb.get_model("B1855+09_NANOGrav_9yv1.gls.par")
        # This uses RA and DEC
        cls.m2 = mb.get_model("B1855+09_NANOGrav_dfg+12_TAI_FB90.par")
        cls.t = 5000 * np.random.randn(100) + 49453.0

    def test_ssb_2_psr(self):
        PMELONG_v = self.m1.PMELONG.value
        PMELAT_v = self.m1.PMELAT.value
        PMRA = self.m2.PMRA.value
        PMDEC = self.m2.PMDEC.value
        # Switch off PM
        self.m1.PMELONG.value = 0.0
        self.m1.PMELAT.value = 0.0
        self.m2.PMRA.value = 0.0
        self.m2.PMDEC.value = 0.0

        p1 = self.m1.ssb_to_psb_xyz_ICRS(epoch=self.t)
        p2 = self.m2.ssb_to_psb_xyz_ICRS(epoch=self.t)

        assert np.max(np.abs(p1 - p2)) < 1e-6

        # Switch on PM
        self.m1.PMELONG.value = PMELONG_v
        self.m1.PMELAT.value = PMELAT_v
        self.m2.PMRA.value = PMRA
        self.m2.PMDEC.value = PMDEC

        p1 = self.m1.ssb_to_psb_xyz_ICRS(epoch=self.t)
        p2 = self.m2.ssb_to_psb_xyz_ICRS(epoch=self.t)

        assert np.max(np.abs(p1 - p2)) < 1e-7

    def test_parse_line(self):
        self.m1.ELONG.from_parfile_line(
            "LAMBDA   286.8634893301156  1  0.0000000165859"
        )
        self.m1.ELAT.from_parfile_line(
            "BETA      32.3214877555037  1   0.0000000273526"
        )
        self.m1.PMELONG.from_parfile_line("PMLAMBDA  -3.2701  1 0.0141")
        self.m1.PMELAT.from_parfile_line("PMBETA  -5.0982  1  0.0291")
        ELONG_v = self.m1.ELONG.value
        ELAT_v = self.m1.ELAT.value
        PMELONG_v = self.m1.PMELONG.value
        PMELAT_v = self.m1.PMELAT.value

        self.m1.ELONG.from_parfile_line("ELONG   286.8634893301156  1  0.0000000165859")
        self.m1.ELAT.from_parfile_line(
            "ELAT      32.3214877555037  1   0.0000000273526"
        )
        self.m1.PMELONG.from_parfile_line("PMELONG  -3.2701  1 0.0141")
        self.m1.PMELAT.from_parfile_line("PMELAT  -5.0982  1  0.0291")

        assert np.isclose(self.m1.ELONG.value, ELONG_v)
        assert np.isclose(self.m1.ELAT.value, ELAT_v)
        assert np.isclose(self.m1.PMELONG.value, PMELONG_v)
        assert np.isclose(self.m1.PMELAT.value, PMELAT_v)
