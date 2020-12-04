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
    p = Polycos()
    p.read_polyco_file(polyco_file)
    table = p.polycoTable
    entry = table["entry"][0]
    print(entry)

    # Test single float - arrays are tested later.
    mjd = 55000.0
    p.eval_spin_freq(mjd)
    p.eval_abs_phase(mjd)
    p.eval_phase(mjd)
    p.find_entry(mjd)

    with pytest.raises(ValueError):
        p.find_entry(200000.0)


def test_read_write_round_trip(tmpdir, polyco_file):
    with open(polyco_file, "r") as f:
        p1 = f.read()

    output_polyco = tmpdir / "B1855_polyco_round_trip.dat"
    p = Polycos()
    p.read_polyco_file(polyco_file)
    p.write_polyco_file(output_polyco)

    with open(output_polyco, "r") as f:
        p2 = f.read()

    assert p1 == p2


def test_generate_polycos(tmpdir, par_file):
    output_polyco = tmpdir / "B1855_polyco_round_trip_from_par.dat"

    model = get_model(str(par_file))

    p = Polycos()
    p.generate_polycos(model, 55000, 55002, "ao", 300, 12, 1400.0)
    p.write_polyco_file(output_polyco)
    q = Polycos()
    q.read_polyco_file(output_polyco)

    mjds = np.linspace(55000.0, 55002.0, 11)

    t = toa.get_TOAs_list([toa.TOA(mjd, obs="ao", freq=1400.0) for mjd in mjds])
    ph1 = p.eval_abs_phase(mjds)
    ph2 = q.eval_abs_phase(mjds)
    ph3 = model.phase(t)

    assert all(int(ph1.int.value[0]) == int(ph3.int.value[0]))
    assert np.allclose(ph1.frac.value[0], ph3.frac.value[0])

    # Loss of precision expected from writing to Polyco from par file.
    assert all(int(ph2.int.value[0]) == int(ph3.int.value[0]))
    assert np.allclose(ph2.frac.value[0], ph3.frac.value[0], rtol=1e-3)
