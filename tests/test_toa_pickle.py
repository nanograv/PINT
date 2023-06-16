import contextlib
import os
import shutil
import time
import pytest
import copy

import pytest
from pinttestdata import datadir

from pint import toa


@pytest.fixture
def temp_tim(tmpdir):
    tt = os.path.join(tmpdir, "test.tim")
    shutil.copy(os.path.join(datadir, "test2.tim"), tt)
    tp = os.path.join(tmpdir, "test.tim.pickle.gz")
    return tt, tp


class TestTOAReader:
    def setup_method(self):
        os.chdir(datadir)
        # First, read the TOAs from the tim file.
        # This should also create the pickle file.
        with contextlib.suppress(OSError):
            os.remove("test1.tim.pickle.gz")
            os.remove("test1.tim.pickle")
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
    d = {k: v}
    wd = {k: wv}
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
# in the pickle file does not exist, then the pickle is
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


def test_save_pickle_exceptions(temp_tim):
    """Raise exceptions in save_pickle if output filename is not given."""

    tt, tp = temp_tim

    toas = toa.get_TOAs(tt, usepickle=True, picklefilename=tp)

    toas1 = toa.merge_TOAs(toas[:4], toas[4:])

    toas2 = copy.deepcopy(toas)
    toas2.filename = None

    # For merged TOAs.
    with pytest.raises(ValueError):
        toa.save_pickle(toas1)

    # For TOAs where filename is not set.
    with pytest.raises(ValueError):
        toa.save_pickle(toas2)
