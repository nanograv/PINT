"""Test polycos."""

import pytest
import numpy as np
from pint.polycos import Polycos
from pint.models import get_model
import pint.toa as toa
from pinttestdata import datadir
from pathlib import Path


@pytest.fixture
def polyco_file():
    return Path(datadir) / "B1855_polyco.dat"


@pytest.fixture
def par_file():
    return Path(datadir) / "B1855+09_polycos.par"


def test_polycos_basic(polyco_file):
    """Just run various features to make sure none of them throw errors.

    Except the ones that should.
    """
    p = Polycos.read(polyco_file)
    table = p.polycoTable
    entry = table["entry"][0]
    print(entry)

    for mjd in [55000.0, np.array([55000.0, 55001.0])]:
        p.eval_spin_freq(mjd)
        p.eval_abs_phase(mjd)
        p.eval_phase(mjd)
        p.find_entry(mjd)


def test_find_entry(polyco_file):
    """Check polycos find the correct entry and out of bounds throws an error."""
    p = Polycos.read(polyco_file)

    assert np.all(p.find_entry(55000) == 0)
    assert np.all(p.find_entry(55001) == 4)

    ts_2 = np.linspace(55000.3125, 55000.52, 1000)
    assert np.all(p.find_entry(ts_2) == 2)

    for t in [54999.8, 55000.8, 55001.8, np.linspace(54999, 55002, 1000)]:
        with pytest.raises(ValueError):
            _ = p.find_entry(t)


def test_read_write_round_trip(tmpdir, polyco_file):
    with open(polyco_file, "r") as f:
        p1 = f.read()

    output_polyco = tmpdir / "B1855_polyco_round_trip.dat"
    p = Polycos.read(polyco_file)
    p.write_polyco_file(str(output_polyco))
    with open(output_polyco, "r") as f:
        p2 = f.read()

    assert p1 == p2


def test_generate_polycos_maxha_error(par_file):
    model = get_model(str(par_file))
    with pytest.raises(ValueError):
        p = Polycos.generate_polycos(model, 55000, 55002, "ao", 144, 12, 400.0, maxha=8)


@pytest.mark.parametrize("obs", ["ao", "gbt", "@", "coe"])
@pytest.mark.parametrize("obsfreq", [1400.0, 400.0])
@pytest.mark.parametrize("nspan, ncoeff", [(144, 12), (72, 9)])
def test_generate_polycos(tmpdir, par_file, obs, obsfreq, nspan, ncoeff):
    output_polyco = tmpdir / "B1855_polyco_round_trip_from_par.dat"
    mjd_start, mjd_end = 55000.0, 55001.0

    model = get_model(str(par_file))

    p = Polycos.generate_polycos(model, mjd_start, mjd_end, obs, nspan, ncoeff, obsfreq)
    p.write_polyco_file(str(output_polyco))
    q = Polycos(str(output_polyco))

    mjds = np.linspace(mjd_start, mjd_end, 51)

    t = toa.get_TOAs_array(mjds, obs=obs, freqs=obsfreq, ephem=model.EPHEM.value)
    ph1 = p.eval_abs_phase(mjds)
    ph2 = q.eval_abs_phase(mjds)
    ph3 = model.phase(t, abs_phase=True)

    assert np.allclose(ph1.int.value[0], ph3.int.value[0])
    assert np.allclose(ph1.frac.value[0], ph3.frac.value[0])
    assert np.allclose(ph2.int.value[0], ph3.int.value[0])
    assert np.allclose(ph2.frac.value[0], ph3.frac.value[0])
