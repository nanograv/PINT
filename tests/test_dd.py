"""Various tests to assess the performance of the DD model."""
import os
import unittest

import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir
from utils import verify_stand_alone_binary_parameter_updates

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()
    s.parfileB1855 = datadir / "B1855+09_NANOGrav_dfg+12_modified_DD.par"
    s.timB1855 = datadir / "B1855+09_NANOGrav_dfg+12.tim"
    s.toasB1855 = toa.get_TOAs(
        s.timB1855,
        ephem="DE405",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    s.modelB1855 = mb.get_model(s.parfileB1855)
    # libstempo result
    s.ltres, s.ltbindelay = np.genfromtxt(
        str(s.parfileB1855) + ".tempo_test", unpack=True
    )
    return s


def test_J1855_binary_delay(setup):
    # Calculate delays with PINT
    pint_binary_delay = setup.modelB1855.binarymodel_delay(setup.toasB1855, None)
    assert np.all(np.abs(pint_binary_delay.value + setup.ltbindelay) < 1e-11)


# TODO: PINT can still incresase the precision by adding more components
def test_B1855(setup):
    pint_resids_us = Residuals(
        setup.toasB1855, setup.modelB1855, use_weighted_mean=False
    ).time_resids.to(u.s)
    assert np.all(np.abs(pint_resids_us.value - setup.ltres) < 1e-7)
