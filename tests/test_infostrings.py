"""Tests for adding info strings to parfiles and tim files"""
import logging
import os
import unittest
import pytest
import copy
import io
import platform

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir
import pint


class SimpleSetup:
    def __init__(self, par, tim):
        self.par = par
        self.tim = tim
        self.m = mb.get_model(self.par)
        self.t = toa.get_TOAs(
            self.tim, ephem="DE405", planets=False, include_bipm=False
        )


@pytest.fixture
def setup_NGC6440E():
    os.chdir(datadir)
    return SimpleSetup("NGC6440E.par", "NGC6440E.tim")


def test_infostring_in_parfile(setup_NGC6440E):
    parfile = setup_NGC6440E.m.as_parfile(comment="test parfile writing")
    f = io.StringIO(parfile)
    newmodel = mb.get_model(f)
    for p in setup_NGC6440E.m.params:
        assert (
            getattr(newmodel, p).value == getattr(setup_NGC6440E.m, p).value
        ), f"Value of {p} does not match in new par file: ({getattr(newmodel, p).value} vs. {getattr(setup_NGC6440E.m, p).value})"


def test_contents_of_infostring_parfile(setup_NGC6440E):
    parfile = setup_NGC6440E.m.as_parfile(comment="test parfile writing")
    f = io.StringIO(parfile)
    lines = f.readlines()
    info = {}
    for line in lines:
        if line.startswith("# ") and ":" in line:
            data = line[2:].split(":")
            k = data[0]
            v = ":".join(data[1:])
            info[k] = v.strip()
    assert info["Comment"] == "test parfile writing"
    assert info["PINT_version"] == pint.__version__
    assert info["Host"] == platform.node()
    assert info["OS"] == platform.platform()


def test_infostring_in_timfile(setup_NGC6440E):
    f = io.StringIO()
    setup_NGC6440E.t.write_TOA_file(f, comment="test timfile writing")
    f.seek(0)
    newtoas = toa.get_TOAs(f)
    assert (setup_NGC6440E.t.get_mjds() == newtoas.get_mjds()).all()


def test_contents_of_infostring_timfile(setup_NGC6440E):
    f = io.StringIO()
    setup_NGC6440E.t.write_TOA_file(f, comment="test timfile writing")
    f.seek(0)
    lines = f.readlines()
    info = {}
    for line in lines:
        if line.startswith("C ") and ":" in line:
            data = line[2:].split(":")
            k = data[0]
            v = ":".join(data[1:])
            info[k] = v.strip()
    assert info["Comment"] == "test timfile writing"
    assert info["PINT_version"] == pint.__version__
    assert info["Host"] == platform.node()
    assert info["OS"] == platform.platform()
