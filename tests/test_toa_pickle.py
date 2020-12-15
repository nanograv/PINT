#!/usr/bin/env python
import os
import unittest
import shutil
import time

import pytest

from pint import toa
from pinttestdata import datadir


@pytest.fixture
def temp_tim(tmpdir):
    tt = os.path.join(tmpdir, "test.tim")
    shutil.copy(os.path.join(datadir, "test2.tim"), tt)
    tp = os.path.join(tmpdir, "test.tim.pickle.gz")
    return tt, tp


class TestTOAReader(unittest.TestCase):
    def setUp(self):
        os.chdir(datadir)
        # First, read the TOAs from the tim file.
        # This should also create the pickle file.
        try:
            os.remove("test1.tim.pickle.gz")
            os.remove("test1.tim.pickle")
        except OSError:
            pass
        tt = toa.get_TOAs("test1.tim", usepickle=False, include_bipm=False)
        self.numtoas = tt.ntoas
        del tt
        # Now read them from the pickle
        self.t = toa.get_TOAs("test1.tim", usepickle=True, include_bipm=False)

    def test_pickle(self):
        # Initially this just checks that the same number
        # of TOAs came out of the pickle as went in.
        assert self.t.ntoas == self.numtoas


def test_pickle_created(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)
    assert os.path.exists(tp)


def test_pickle_works(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)
    toa.get_TOAs(tt, usepickle=True)


def test_pickle_used(temp_tim, monkeypatch):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)

    def no(*args, **kwargs):
        raise ValueError

    monkeypatch.setattr(toa.TOAs, "read_pickle_file", no)
    with pytest.raises(ValueError):
        toa.get_TOAs(tt, usepickle=True)


def test_pickle_used_settings(temp_tim, monkeypatch):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True, ephem="de436")
    assert toa.get_TOAs(tt, usepickle=True).ephem == "de436"


def test_pickle_changed_ephem(temp_tim, monkeypatch):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True, ephem="de436")
    assert toa.get_TOAs(tt, usepickle=True, ephem="de421").ephem == "de421"


def test_pickle_changed_planets(temp_tim, monkeypatch):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True, planets=True)
    assert not toa.get_TOAs(tt, usepickle=True, planets=False).planets


@pytest.mark.parametrize(
    "k,v,wv",
    [
        ("bipm_version", "BIPM2019", "BIPM2018"),
        ("include_bipm", True, False),
        ("include_gps", True, False),
    ],
)
def test_pickle_invalidated_settings(temp_tim, monkeypatch, k, v, wv):
    tt, tp = temp_tim
    d = {}
    d[k] = v
    wd = {}
    wd[k] = wv
    toa.get_TOAs(tt, usepickle=True, **d)
    assert toa.get_TOAs(tt, usepickle=True, **wd).clock_corr_info[k] == wv


def test_pickle_invalidated_time(temp_tim, monkeypatch):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)

    rpf = toa.TOAs.read_pickle_file

    def change(self, *args, **kwargs):
        rpf(self, *args, **kwargs)
        self.was_pickled = True

    monkeypatch.setattr(toa.TOAs, "read_pickle_file", change)
    assert toa.get_TOAs(tt, usepickle=True).was_pickled

    time.sleep(1)
    with open(tt, "at") as f:
        f.write("\n")
    assert not hasattr(toa.get_TOAs(tt, usepickle=True), "was_pickled")
