import os
import shutil
import pytest
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

# For this test, turn off the check for the age of the IERS A table
from astropy.utils.iers import conf
import astropy.table
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, sampled_from
from pinttestdata import datadir

from pint import simulation, toa
from pint.models import get_model, get_model_and_toas
from pint.observatory import bipm_default

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


class TestTOAReader:
    def setup_method(self):
        self.x = toa.TOAs(datadir / "test1.tim")
        self.x.apply_clock_corrections()
        self.x.compute_TDBs()
        self.x.table.sort("index")

    def test_read_parkes(self):
        ts = toa.get_TOAs(datadir / "parkes.toa")
        assert "barycenter" in ts.observatories
        assert ts.ntoas == 8

    def test_read_princeton(self):
        toaline = "6   11 1744-24 1666.0000 50852.5886574493316     25.6  8-FEB-98    0.0    1  1"
        mjd, d = toa._parse_TOA_line(toaline)
        assert np.isclose(mjd[0] + mjd[1], 50852.5886574493316)
        assert d["format"] == "Princeton"
        assert d["obs"] == "vla"
        assert d["freq"] == 1666.0
        assert d["error"] == 25.6

    def test_read_princeton_offset1(self):
        toaline = "6   26 1744-24 1667.0000 9064.9010416342947       6.2 26-OCT-90    0.1    1  1"
        mjd, d = toa._parse_TOA_line(toaline)
        assert np.isclose(mjd[0] + mjd[1], 48190.9010416342947)

    def test_read_princeton_offset2(self):
        toaline = "6   37 1744-24 1666.000010062.1379629521572       7.8 20-JUL-93   64.6    1  1"
        mjd, d = toa._parse_TOA_line(toaline)
        assert np.isclose(mjd[0] + mjd[1], 49188.1379629521572)

    def test_read_parkes_phaseoffset(self):
        # Fourth column contains non-zero phase offset
        toaline = " PUPPI_J2044+28_58852_652 432.3420  58852.7590686063892    1.00  120.75        @"

        # To be changed if/when phase offset for Parkes format is implemented
        with pytest.raises(ValueError):
            toa._parse_TOA_line(toaline)

    def test_commands(self):
        assert len(self.x.commands) == 19

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
    y = toa.get_TOAs(datadir / "test1.tim", model=m)
    assert y.ephem == "DE436"
    assert y.planets is True
    assert y.clock_corr_info["bipm_version"] == "BIPM2017"


def test_model_override2():
    parstr = StringIO(simplepar)
    m, y = get_model_and_toas(parstr, datadir / "test1.tim")
    assert y.ephem == "DE436"
    assert y.planets is True
    assert y.clock_corr_info["bipm_version"] == "BIPM2017"


def test_model_override_override2():
    parstr = StringIO(simplepar)
    m, y = get_model_and_toas(parstr, datadir / "test1.tim", ephem="DE405")
    assert y.ephem == "DE405"


def test_model_override_override():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    y = toa.get_TOAs(
        datadir / "test1.tim",
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
    y = toa.get_TOAs(datadir / "test1.tim", model=m)
    assert y.clock_corr_info["include_bipm"] == False


def test_toa_check_hashes(tmp_path):
    tf = tmp_path / "file.tim"
    shutil.copy(datadir / "NGC6440E.tim", tf)
    t = toa.get_TOAs(tf)
    assert t.check_hashes()
    with open(tf, "at") as f:
        f.write("\n")
    assert not t.check_hashes()


def test_toa_merge():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    ntoas = sum(tt.ntoas for tt in toas)
    nt = toa.merge_TOAs(toas)
    assert len(nt.observatories) == 3
    assert nt.table.meta["filename"] == filenames
    assert nt.ntoas == ntoas
    check_indices_contiguous(nt)
    assert nt.check_hashes()


def test_toa_merge_again():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    ntoas = sum(tt.ntoas for tt in toas)
    nt = toa.merge_TOAs(toas)
    # The following tests merging with and already merged TOAs
    other = toa.get_TOAs(datadir / "test1.tim")
    nt2 = toa.merge_TOAs([nt, other])
    assert len(nt2.filename) == 5
    assert nt2.ntoas == ntoas + 9
    check_indices_contiguous(nt2)


def test_toa_merge_again_2():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    ntoas = sum(tt.ntoas for tt in toas)
    other = toa.get_TOAs(datadir / "test1.tim")
    # check consecutive merging
    nt = toa.merge_TOAs(toas[:2])
    nt = toa.merge_TOAs([nt, toas[2]])
    nt = toa.merge_TOAs([nt, other])
    assert len(nt.filename) == 5
    assert nt.ntoas == ntoas + 9
    check_indices_contiguous(nt)
    assert nt.check_hashes()


def test_toa_merge_different_ephem():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    toas = [toa.get_TOAs(ff) for ff in filenames]
    toas[0].ephem = "DE436"
    with pytest.raises(TypeError):
        nt = toa.merge_TOAs(toas)


def test_toa_merge_different_columns_strict():
    # merge with strict=True should fail with an error about pulse_number
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    # add a pulse_number column.  then the merge should fail
    toas = [toa.get_TOAs(ff, model=model) for ff in filenames]
    toas[0].compute_pulse_numbers(model)
    with pytest.raises(astropy.table.np_utils.TableMergeError) as exc:
        nt = toa.merge_TOAs(toas, strict=True)
    assert "pulse_number" in str(exc)


def test_toa_merge_different_columns_notstrict():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    # add a pulse_number column.  then the merge should fail
    toas = [toa.get_TOAs(ff, model=model) for ff in filenames]
    toas[0].compute_pulse_numbers(model)
    nt = toa.merge_TOAs(toas, strict=False)


def test_toa_merge_different_columns_TDB_notstrict():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    toas = [toa.get_TOAs(ff, model=model) for ff in filenames]
    # remove the tdb column
    # merge should fail if strict
    del toas[1].table["tdb"]
    nt = toa.merge_TOAs(toas, strict=False)


def test_toa_merge_different_columns_TDB_strict():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    toas = [toa.get_TOAs(ff, model=model) for ff in filenames]
    # remove the tdb column
    # merge should fail if strict
    del toas[1].table["tdb"]
    with pytest.raises(astropy.table.np_utils.TableMergeError) as exc:
        nt = toa.merge_TOAs(toas, strict=True)
    assert "tdb" in str(exc)


def test_toa_merge_different_columns_posvel_notstrict():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    toas = [toa.get_TOAs(ff, model=model) for ff in filenames]
    # remove the ssb_obs_pos column
    # merge should fail if strict
    del toas[1].table["ssb_obs_pos"]
    nt = toa.merge_TOAs(toas, strict=False)


def test_toa_merge_different_columns_posvel_strict():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    toas = [toa.get_TOAs(ff, model=model) for ff in filenames]
    # remove the ssb_obs_pos column
    # merge should fail if strict
    del toas[1].table["ssb_obs_pos"]
    with pytest.raises(astropy.table.np_utils.TableMergeError) as exc:
        nt = toa.merge_TOAs(toas, strict=True)
    assert "ssb_obs_pos" in str(exc)


def test_toa_merge_different_columns_ignorepn_onread():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    # read and add pulse numbers
    t = toa.get_TOAs(filenames[0], model=model)
    t.compute_pulse_numbers(model)
    f = StringIO()
    t.write_TOA_file(f)
    f.seek(0)
    toas = [toa.get_TOAs(fname, include_pn=False) for fname in [f, filenames[1]]]
    nt = toa.merge_TOAs(toas)


def test_toa_merge_different_columns_ignorepn_onwrite():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
    ]
    model = get_model(datadir / "NGC6440E.par")
    # read and add pulse numbers
    t = toa.get_TOAs(filenames[0], model=model)
    t.compute_pulse_numbers(model)
    f = StringIO()
    t.write_TOA_file(f, include_pn=False)
    f.seek(0)
    toas = [toa.get_TOAs(fname, include_pn=True) for fname in [f, filenames[1]]]
    nt = toa.merge_TOAs(toas)


def test_toa_merge_different_bipm():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    inc_bipms = [True, True, False]
    toas = [
        toa.get_TOAs(ff, include_bipm=inc_bipm)
        for ff, inc_bipm in zip(filenames, inc_bipms)
    ]
    with pytest.raises(TypeError):
        nt = toa.merge_TOAs(toas)


def test_toa_merge_different_bipm_ver():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    bipm_vers = ["BIPM2015", "BIPM2015", "BIPM2017"]
    toas = [
        toa.get_TOAs(ff, include_bipm=True, bipm_version=bipm_ver)
        for ff, bipm_ver in zip(filenames, bipm_vers)
    ]
    with pytest.raises(TypeError):
        nt = toa.merge_TOAs(toas)


def test_toa_merge_different_gps():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    inc_gpss = [True, True, False]
    toas = [
        toa.get_TOAs(ff, include_gps=inc_gps)
        for ff, inc_gps in zip(filenames, inc_gpss)
    ]
    with pytest.raises(TypeError):
        nt = toa.merge_TOAs(toas)


def test_toa_merge_different_planets():
    filenames = [
        datadir / "NGC6440E.tim",
        datadir / "testtimes.tim",
        datadir / "parkes.toa",
    ]
    inc_planetss = [True, True, False]
    toas = [
        toa.get_TOAs(ff, planets=inc_planets)
        for ff, inc_planets in zip(filenames, inc_planetss)
    ]
    with pytest.raises(TypeError):
        nt = toa.merge_TOAs(toas)


def test_bipm_default():
    m, t = get_model_and_toas(
        StringIO(simplepar.replace("BIPM2017", "BIPM")), datadir / "test1.tim"
    )
    assert t.clock_corr_info["bipm_version"] == bipm_default


def test_toas_comparison():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    x = toa.get_TOAs(datadir / "test1.tim", model=m)
    y = toa.get_TOAs(datadir / "test1.tim", model=m)
    assert x == y


def test_toas_comparison_unequal():
    parstr = StringIO(simplepar)
    m = get_model(parstr)
    x = toa.get_TOAs(datadir / "test1.tim", model=m, ephem="de436")
    y = toa.get_TOAs(datadir / "test1.tim", model=m)
    assert x != y


def test_toas_read_list():
    x = toa.get_TOAs(datadir / "test1.tim")
    toas, commands = toa.read_toa_file(datadir / "test1.tim")
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


def test_load_multiple(tmp_path):
    m = get_model(StringIO(simplepar))

    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 10, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 10, model=m, obs="@"),
    ]

    filenames = [tmp_path / f"t{i+1}.tim" for i in range(len(fakes))]

    for t, f in zip(fakes, filenames):
        t.write_TOA_file(f, format="tempo2")

    merged = toa.merge_TOAs([toa.get_TOAs(f, model=m) for f in filenames])

    assert merged == toa.get_TOAs(filenames, model=m)


def test_pickle_multiple(tmp_path):
    m = get_model(StringIO(simplepar))

    fakes = [
        simulation.make_fake_toas_uniform(55000, 55500, 10, model=m, obs="ao"),
        simulation.make_fake_toas_uniform(56000, 56500, 10, model=m, obs="gbt"),
        simulation.make_fake_toas_uniform(57000, 57500, 10, model=m, obs="@"),
    ]

    filenames = [tmp_path / f"t{i+1}.tim" for i in range(len(fakes))]
    picklefilename = tmp_path / "t.pickle.gz"

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
    tim
    for tim in datadir.glob("*.tim")
    if tim not in {datadir / "prefixtest.tim", datadir / "vela_wave.tim"}
]


# This is slow and not very helpful
@given(sampled_from(loadable_tims))
@settings(max_examples=1)
def test_contiguous_on_load(tim):
    check_indices_contiguous(toa.get_TOAs(tim))


def test_include_directory(tmp_path):
    d = Path.cwd()
    try:
        os.chdir(datadir)
        # "test1.tim" has "INCLUDE test2.tim"
        # This should work
        toa.get_TOAs(str(Path(datadir) / "test1.tim"))
        os.chdir(tmp_path)
        # This should also work
        toa.get_TOAs(str(Path(datadir) / "test1.tim"))
    finally:
        os.chdir(d)


def test_chained_include_directories(tmp_path):
    t1 = tmp_path / "test.tim"
    b = tmp_path / "b"
    b.mkdir()
    t1.write_text(
        """
        FORMAT 1

        toa1 1404.000 55336.989701997555466   3.469  gbt  -fe Rcvr1_2 -be GASP -f Rcvr1_2_GASP -bw 4 -tobs 1740.3

        INCLUDE b/test2.tim
    """
    )
    (b / "test2.tim").write_text(
        """
        FORMAT 1

        toa2 1404.000 55337.989701997555466   3.469  gbt  -fe Rcvr1_2 -be GASP -f Rcvr1_2_GASP -bw 4 -tobs 1740.3

        INCLUDE ../test3.tim
    """
    )
    (tmp_path / "test3.tim").write_text(
        """
        FORMAT 1

        toa3 1404.000 55338.989701997555466   3.469  gbt  -fe Rcvr1_2 -be GASP -f Rcvr1_2_GASP -bw 4 -tobs 1740.3
    """
    )

    assert len(toa.get_TOAs(str(t1))) == 3


def test_read_itoa():
    # This test is put here only to ensure that the correct exception is raised.
    # This should be removed or replaced if/when ITOA support is implemented.
    timfile = datadir / "NGC6440E.itoa"
    with pytest.raises(RuntimeError):
        toa.get_TOAs(timfile)


def test_parse_toa_line_exceptions():
    # This should work.
    goodline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    toa._parse_TOA_line(goodline)

    # obs is given as a flag.
    badline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao -obs ao -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    with pytest.raises(ValueError):
        toa._parse_TOA_line(badline)

    # Flag without corresponding value (-a)
    badline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao -a -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    with pytest.raises(ValueError):
        toa._parse_TOA_line(badline)

    # Empty flag label (-)
    badline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao - a -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    with pytest.raises(ValueError):
        toa._parse_TOA_line(badline)

    # Garbage line
    garbage = "asdg skfgs dj"
    with pytest.raises(RuntimeError):
        toa._parse_TOA_line(garbage)
