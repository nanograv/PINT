"""Various tests to assess the performance of the FD model."""
import copy
import os
import pytest
from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

import pint.models.model_builder as mb
import pint.residuals
import pint.toa as toa


class TestFD:
    @classmethod
    def setup_class(cls):
        cls.parf = os.path.join(datadir, "test_FD.par")
        cls.timf = os.path.join(datadir, "test_FD.simulate.pint_corrected")
        cls.FDm = mb.get_model(cls.parf)
        cls.toas = toa.get_TOAs(cls.timf, include_bipm=False)
        # libstempo result
        cls.ltres, cls.ltbindelay = np.genfromtxt(f"{cls.parf}.tempo_test", unpack=True)

    def test_fd(self):
        print("Testing FD module.")
        rs = (
            pint.residuals.Residuals(self.toas, self.FDm, use_weighted_mean=False)
            .time_resids.to(u.s)
            .value
        )
        resDiff = rs - self.ltres
        # NOTE : This precision is a lower then 1e-7 seconds level, due to some
        # early parks clock corrections are treated differently.
        # TEMPO2: Clock correction = clock0 + clock1 (in the format of general2)
        # PINT : Clock correction = toas.table['flags']['clkcorr']
        # Those two clock correction difference are causing the trouble.
        assert np.all(resDiff < 5e-6), "PINT and tempo Residual difference is too big. "

    def test_inf_freq(self):
        test_toas = copy.deepcopy(self.toas)
        test_toas.table["freq"][:5] = np.inf * u.MHz
        fd_delay = self.FDm.components["FD"].FD_delay(test_toas)
        assert np.all(
            np.isfinite(fd_delay)
        ), "FD component is not handling infinite frequency right."
        assert np.all(
            fd_delay[:5].value == 0.0
        ), "FD component did not compute infinite frequency delay right"
        d_d_d_fd = self.FDm.d_delay_FD_d_FDX(test_toas, "FD1")
        assert np.all(np.isfinite(d_d_d_fd)), (
            "FD component is not handling infinite frequency right when doning"
            + " derivatives."
        )


@pytest.fixture
def fd_sample():
    return mb.get_model(
        StringIO(
            """
        PSR J1234+5678
        F0 1
        PEPOCH 57000
        ELAT 0
        ELONG 0
        DM 10
        FD1 0
        FD2 -1
    """
        )
    )


def test_fd_frequency_infinite_no_effect(fd_sample):
    assert fd_sample.FD_delay_frequency(np.array([np.inf]) * u.MHz) == 0


def test_fd_frequency_ghz_no_effect(fd_sample):
    assert fd_sample.FD_delay_frequency(np.array([1]) * u.GHz) == 0


def test_fd_frequency_finite_varying(fd_sample):
    delays = fd_sample.FD_delay_frequency(np.array([1, 2]) * u.GHz)
    assert delays[0] != delays[1]


def test_fd_frequency_convexity_fd2(fd_sample):
    delays = fd_sample.FD_delay_frequency(np.array([0.5, 1, 2]) * u.GHz)
    # Because FD2 < 0 we expect
    assert delays[0] < delays[1]
    assert delays[1] > delays[2]
