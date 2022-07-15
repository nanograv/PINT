#!/usr/bin/env python
import os
import pickle
import shutil
import time
import unittest

import astropy.time
import astropy.units as u
import numpy as np
import pytest
from pinttestdata import datadir

import pint.models
import pint.toa
from pint import simulation, toa


@pytest.fixture
def temp_tim(tmpdir):
    tt = os.path.join(tmpdir, "test.tim")
    shutil.copy(os.path.join(datadir, "test2.tim"), tt)
    tp = os.path.join(tmpdir, "test.tim.pickle.gz")
    return tt, tp


def test_pickle_created(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)
    assert os.path.exists(tp)


def test_pickle_works(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)
    toa.get_TOAs(tt, usepickle=True)


def test_pickle_recovers_number_of_toas(temp_tim):
    tt, tp = temp_tim
    t1 = toa.get_TOAs(tt, usepickle=True)
    t2 = toa.get_TOAs(tt, usepickle=True)
    assert len(t1) == len(t2)


def test_pickle_used(temp_tim):
    tt, tp = temp_tim
    assert not toa.get_TOAs(tt, usepickle=True).was_pickled
    assert toa.get_TOAs(tt, usepickle=True).was_pickled


def test_pickle_used_settings(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True, ephem="de436")
    assert toa.get_TOAs(tt, usepickle=True).ephem == "de436"


def test_pickle_changed_ephem(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True, ephem="de436")
    assert toa.get_TOAs(tt, usepickle=True, ephem="de421").ephem == "de421"


def test_pickle_changed_planets(temp_tim):
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
def test_pickle_invalidated_settings(temp_tim, k, v, wv):
    tt, tp = temp_tim
    d = {}
    d[k] = v
    wd = {}
    wd[k] = wv
    toa.get_TOAs(tt, usepickle=True, **d)
    assert toa.get_TOAs(tt, usepickle=True, **wd).clock_corr_info[k] == wv


def test_pickle_invalidated_time(temp_tim):
    tt, tp = temp_tim
    toa.get_TOAs(tt, usepickle=True)
    assert toa.get_TOAs(tt, usepickle=True).was_pickled

    time.sleep(1)
    with open(tt, "at") as f:
        f.write("\n")
    assert not toa.get_TOAs(tt, usepickle=True).was_pickled


# SMR changed the behavior of this so that if the .filename
# in the picklefile does not exist, then the pickle is
# invalidated.  The reason is that the previous behavior
# caused pickling to always fail to be valid for .tim files
# that had INCLUDE statements to load other .tim files.
def test_pickle_moved(temp_tim):
    tt, tp = temp_tim
    # The following creates "test.tim.pickle.gz" from "test.tim"
    toa.get_TOAs(tt, usepickle=True, picklefilename=tp)
    # now remove "test.tim"
    os.remove(tt)
    # Should fail since the original file is gone
    with pytest.raises(FileNotFoundError):
        toa.get_TOAs(tt, usepickle=True, picklefilename=tp)


def test_pickle_dir_works(tmp_path):
    a = tmp_path / "a"
    a.mkdir()

    b = tmp_path / "b"
    b.mkdir()

    shutil.copy(datadir / "test2.tim", a)
    toa.get_TOAs(a / "test2.tim", usepickle=True, picklefilename=b)

    assert len(list(a.iterdir())) == 1
    assert len(list(b.iterdir())) == 1

    t = toa.get_TOAs(a / "test2.tim", usepickle=True, picklefilename=b)
    assert t.was_pickled
