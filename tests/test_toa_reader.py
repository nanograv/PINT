import os
import unittest
import pytest
import numpy as np
import shutil
from io import StringIO
from glob import glob

from hypothesis import given
from hypothesis.strategies import integers, floats, sampled_from
from hypothesis.extra.numpy import arrays
from pint import toa, simulation
from pint.observatory import bipm_default
from pint.models import get_model, get_model_and_toas
from pinttestdata import datadir

# For this test, turn off the check for the age of the IERS A table
from astropy.utils.iers import conf

conf.auto_max_age = None

os.chdir(datadir)

simplepar = """
PSR              1748-2021E
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""

os.chdir(datadir)


def check_indices_unique(toas):
    ix = toas.table["index"].copy()
    ix.sort()
    assert not np.any(np.diff(ix) == 0)
    assert toas.max_index >= ix[-1]


def check_indices_contiguous(toas):
    ix = toas.table["index"].copy()
    ix.sort()
    assert np.all(np.diff(ix) == 1)
    assert toas.max_index == ix[-1]
    assert ix[0] == 0


class TestTOAReader(unittest.TestCase):
    def setUp(self):
        self.x = toa.TOAs("test1.tim")
        self.x.apply_clock_corrections()
        self.x.compute_TDBs()
        self.x.table.sort("index")

    def test_read_parkes(self):
        ts = toa.get_TOAs("parkes.toa")
        assert "barycenter" in ts.observatories
        assert ts.ntoas == 8

    def test_commands(self):
        assert len(self.x.commands) == 18

    def test_count(self):
        assert self.x.ntoas == 9

    def test_info(self):
        assert self.x.table[0]["flags"]["info"] == "test1"

    def test_jump(self):
        assert self.x.table[0]["flags"]["jump"] == "1"

    def test_info_2(self):
        assert self.x.table[3]["flags"]["info"] == "test2"

    def test_time(self):
        assert float(self.x.table[3]["flags"]["to"]) == 1.0

    def test_jump_2(self):
        assert "jump" not in self.x.table[4]["flags"]

    def test_time_2(self):
        assert "time" not in self.x.table[4]["flags"]

    def test_jump_3(self):
        assert self.x.table[-1]["flags"]["jump"] == "2"

    def test_obs(self):
        assert self.x.table[1]["obs"] == "gbt"

    def test_ephem(self):
        assert self.x.ephem == "DE421"

    def test_planets(self):
        assert self.x.planets is False

    def test_clock(self):
        assert self.x.clock_corr_info["bipm_version"] == bipm_default


def test_model_override1():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    y = toa.get_TOAs("test1.tim", model=m)
    assert y.ephem == "DE436"
    assert y.planets is True
    assert y.clock_corr_info["bipm_version"] == "BIPM2017"


def test_model_override2():
    parstr = StringIO(simplepar)
    m, y = get_model_and_toas(parstr, "test1.tim")
    assert y.ephem == "DE436"
    assert y.planets is True
    assert y.clock_corr_info["bipm_version"] == "BIPM2017"


def test_model_override_override2():
    parstr = StringIO(simplepar)
    m, y = get_model_and_toas(parstr, "test1.tim", ephem="DE405")
    assert y.ephem == "DE405"


def test_model_override_override():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    y = toa.get_TOAs(
        "test1.tim",
        model=m,
        ephem="DE405",
        planets=False,
        include_bipm=True,
        bipm_version="BIPM2012",
    )
    assert y.ephem == "DE405"
    assert y.planets is False
    assert y.clock_corr_info["bipm_version"] == "BIPM2012"


def test_tttai():
    # This checks to make sure that CLOCK TT(TAI) is handled correctly
    parstr = StringIO(simplepar.replace("BIPM2017", "TAI"))
    m = get_model(parstr)
    y = toa.get_TOAs("test1.tim", model=m)
    assert y.clock_corr_info["include_bipm"] == False


def test_toa_check_hashes(tmpdir):
    tf = os.path.join(tmpdir, "file.tim")
    shutil.copy("NGC6440E.tim", tf)
    t = toa.get_TOAs(tf)
    assert t.check_hashes()
    with open(tf, "at") as f:
        f.write("\n")
    assert not t.check_hashes()


def test_toa_merge():
    filenames = ["NGC6440E.tim", "testtimes.tim", "parkes.toa"]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    ntoas = sum([tt.ntoas for tt in toas])
    nt = toa.merge_TOAs(toas)
    assert len(nt.observatories) == 3
    assert nt.table.meta["filename"] == filenames
    assert nt.ntoas == ntoas
    check_indices_contiguous(nt)
    assert nt.check_hashes()


def test_toa_merge_again():
    filenames = ["NGC6440E.tim", "testtimes.tim", "parkes.toa"]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    ntoas = sum([tt.ntoas for tt in toas])
    nt = toa.merge_TOAs(toas)
    # The following tests merging with and already merged TOAs
    other = toa.get_TOAs("test1.tim")
    nt2 = toa.merge_TOAs([nt, other])
    assert len(nt2.filename) == 5
    assert nt2.ntoas == ntoas + 9
    check_indices_contiguous(nt2)


def test_toa_merge_again():
    filenames = ["NGC6440E.tim", "testtimes.tim", "parkes.toa"]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    ntoas = sum([tt.ntoas for tt in toas])
    other = toa.get_TOAs("test1.tim")
    # check consecutive merging
    nt = toa.merge_TOAs(toas[:2])
    nt = toa.merge_TOAs([nt, toas[2]])
    nt = toa.merge_TOAs([nt, other])
    assert len(nt.filename) == 5
    assert nt.ntoas == ntoas + 9
    check_indices_contiguous(nt)
    assert nt.check_hashes()


def test_toa_merge_different_ephem():
    filenames = ["NGC6440E.tim", "testtimes.tim", "parkes.toa"]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    toas[0].ephem = "DE436"
    with pytest.raises(TypeError):
        nt = toa.merge_TOAs(toas)


def test_bipm_default():
    m, t = get_model_and_toas(
        StringIO(simplepar.replace("BIPM2017", "BIPM")), "test1.tim"
    )
    assert t.clock_corr_info["bipm_version"] == "BIPM"


def test_toas_comparison():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    x = toa.get_TOAs("test1.tim", model=m)
    y = toa.get_TOAs("test1.tim", model=m)
    assert x == y


def test_toas_comparison_unequal():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    x = toa.get_TOAs("test1.tim", model=m, ephem="de436")
    y = toa.get_TOAs("test1.tim", model=m)
    assert x != y


def test_toas_read_list():
    x = toa.get_TOAs("test1.tim")
    toas, commands = toa.read_toa_file("test1.tim")
    y = toa.get_TOAs_list(toas, commands=commands, filename=x.filename, hashes=x.hashes)
    assert x == y


@given(arrays(float, integers(1, 100), elements=floats(50000, 70000)))
def test_numpy_clusterss(t):
    gap = 1
    clusters = toa._cluster_by_gaps(t, gap)
    for i in range(np.amax(clusters) + 1):
        c = clusters == i
        in_cluster = np.sort(t[c])
        assert np.all(np.diff(in_cluster) < gap)

        for e in [in_cluster[0], in_cluster[-1]]:
            assert np.all(np.abs(t[~c] - e) >= gap)


def test_load_multiple(tmpdir):
    m = get_model(StringIO(simplepar))

    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 10, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 10, model=m, obs="@"),
    ]

    filenames = [os.path.join(tmpdir, f"t{i+1}.tim") for i in range(len(fakes))]

    for t, f in zip(fakes, filenames):
        t.write_TOA_file(f, format="tempo2")

    merged = toa.merge_TOAs([toa.get_TOAs(f, model=m) for f in filenames])

    assert merged == toa.get_TOAs(filenames, model=m)


def test_pickle_multiple(tmpdir):
    m = get_model(StringIO(simplepar))

    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 10, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 10, model=m, obs="@"),
    ]

    filenames = [os.path.join(tmpdir, f"t{i+1}.tim") for i in range(len(fakes))]
    picklefilename = os.path.join(tmpdir, "t.pickle.gz")

    for t, f in zip(fakes, filenames):
        t.write_TOA_file(f, format="tempo2")

    toa.get_TOAs(filenames, model=m, usepickle=True, picklefilename=picklefilename)
    assert os.path.exists(picklefilename)
    assert toa.get_TOAs(
        filenames, model=m, usepickle=True, picklefilename=picklefilename
    ).was_pickled
    with open(filenames[-1], "at") as f:
        f.write("\n")
    assert not toa.get_TOAs(
        filenames, model=m, usepickle=True, picklefilename=picklefilename
    ).was_pickled


def test_merge_indices():
    m = get_model(StringIO(simplepar))
    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 5, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 15, model=m, obs="@"),
    ]
    toas = toa.merge_TOAs(fakes)
    check_indices_contiguous(toas)


def test_merge_indices_excised():
    m = get_model(StringIO(simplepar))
    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 5, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 15, model=m, obs="@"),
    ]
    fakes_excised = [f[1:-1] for f in fakes]
    toas = toa.merge_TOAs(fakes)
    toas_excised = toa.merge_TOAs(fakes_excised)
    check_indices_unique(toas_excised)
    for i in range(len(toas_excised)):
        ix = toas_excised.table["index"][i]
        match = toas.table[toas.table["index"] == ix]
        assert len(match) == 1
        assert match[0]["tdbld"] == toas_excised.table["tdbld"][i]


def test_renumber_subset():
    m = get_model(StringIO(simplepar))
    toas = simulation.make_fake_toas_uniform(55000, 55500, 10, model=m, obs="ao")

    sub = toas[1:-1]
    assert 0 not in sub.table["index"]

    sub.renumber()
    assert np.all(sub.table["index"] == np.arange(len(sub)))


def test_renumber_order():
    m = get_model(StringIO(simplepar))
    toas = simulation.make_fake_toas_uniform(55000, 55500, 10, model=m, obs="ao")
    rev = toas[::-1]
    assert np.all(rev.table["index"] == np.arange(len(rev))[::-1])
    rev.renumber()
    assert np.all(rev.table["index"] == np.arange(len(rev))[::-1])
    rev.renumber(index_order=False)
    assert np.all(rev.table["index"] == np.arange(len(rev)))

    rev = toas[::-1]
    rev.renumber(index_order=True)
    assert np.all(rev.table["index"] == np.arange(len(rev))[::-1])


def test_renumber_subset_reordered():
    m = get_model(StringIO(simplepar))
    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 5, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 15, model=m, obs="@"),
    ]
    fakes_excised = [f[1:-1] for f in fakes]
    toas_excised = toa.merge_TOAs(fakes_excised)

    assert 0 not in toas_excised.table["index"]

    toas_excised.renumber()
    assert set(toas_excised.table["index"]) == set(range(len(toas_excised)))

    toas_excised.renumber(index_order=False)
    assert np.all(toas_excised.table["index"] == np.arange(len(toas_excised)))


loadable_tims = [
    tim for tim in glob("*.tim") if tim not in {"prefixtest.tim", "vela_wave.tim"}
]


@given(sampled_from(loadable_tims))
def test_contiguous_on_load(tim):
    check_indices_contiguous(toa.get_TOAs(tim))
