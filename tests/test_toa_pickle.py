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
from pint import toa
from pint import simulation


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


def test_pickle_moved(temp_tim):
    tt, tp = temp_tim
    tt2 = tt + ".also.tim"
    shutil.copy(tt, tt2)
    toa.get_TOAs(tt, usepickle=True, picklefilename=tp)
    assert toa.get_TOAs(tt2, usepickle=True, picklefilename=tp).was_pickled
    with open(tt2, "at") as f:
        f.write("\n")
    assert not toa.get_TOAs(tt2, usepickle=True, picklefilename=tp).was_pickled


def test_group_survives_pickle(tmpdir):
    p = os.path.join(tmpdir, "test.pickle.gz")
    test_model = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
    test_toas = simulation.make_fake_toas_uniform(58000, 59000, 5, model=test_model)
    test_toas.adjust_TOAs(
        astropy.time.TimeDelta((np.random.uniform(0, 1, size=(5,)) * u.d))
    )

    pint.toa.save_pickle(test_toas, p)
    test_toas2 = pint.toa.load_pickle(p)
    test_toas2.adjust_TOAs(
        astropy.time.TimeDelta((np.random.uniform(0, 1, size=(5,)) * u.d))
    )


@pytest.mark.xfail(reason="astropy tables lose groupedness when pickled")
def test_astropy_group_survives_pickle(tmpdir):
    p = os.path.join(tmpdir, "test.pickle.gz")
    test_model = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
    test_toas = simulation.make_fake_toas_uniform(58000, 59000, 5, model=test_model)

    with open(p, "wb") as f:
        pickle.dump(test_toas.table, f)

    with open(p, "rb") as f:
        new_table = pickle.load(f)

    assert len(test_toas.table.groups.keys) == 1
    assert len(new_table.groups.keys) == 1


def test_group_survives_plain_pickle(tmpdir):
    test_model = pint.models.get_model(os.path.join(datadir, "NGC6440E.par"))
    test_toas = simulation.make_fake_toas_uniform(58000, 59000, 5, model=test_model)

    new_toas = pickle.loads(pickle.dumps(test_toas))

    assert len(test_toas.table.groups.keys) == 1
    assert len(new_toas.table.groups.keys) == 1
