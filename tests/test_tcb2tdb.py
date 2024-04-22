"""Tests for `pint.models.tcb_conversion` and the `tcb2tdb` script."""

import os
from copy import deepcopy
from io import StringIO

import numpy as np
import pytest

from pint.models.model_builder import ModelBuilder
from pint.models.tcb_conversion import convert_tcb_tdb, IFTE_K
from pint.scripts import tcb2tdb

simplepar = """
PSR              PSRTEST
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
BINARY              BT
T0                  53750
A1                  100.0 1 0.1
ECC                 1.0
OM                  0.0
PB                  10.0
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TCB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""


def test_convert_units():
    with pytest.raises(ValueError):
        m = ModelBuilder()(StringIO(simplepar))

    m1 = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    m2 = ModelBuilder()(StringIO(simplepar), allow_tcb=True)

    assert m2.UNITS.value == "TDB"
    assert np.isclose(m2.F0.value / m1.F0.value, m1.PB.value / m2.PB.value)

    assert np.isclose(m1.F0.value / m2.F0.value, 1 / IFTE_K, rtol=1e-9)
    assert np.isclose(m1.DM.value / m2.DM.value, 1 / IFTE_K, rtol=1e-9)


def test_convert_units_roundtrip():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    m_ = deepcopy(m)
    convert_tcb_tdb(m, backwards=False)
    convert_tcb_tdb(m, backwards=True)

    for par in m.params:
        p = getattr(m, par)
        p_ = getattr(m_, par)
        if p.value is None:
            assert p_.value is None
        elif isinstance(p.value, str):
            assert getattr(m, par).value == getattr(m_, par).value
        else:
            assert np.isclose(getattr(m, par).value, getattr(m_, par).value, atol=1e-9)


def test_tcb2tdb(tmp_path):
    tmppar1 = tmp_path / "tmp1.par"
    tmppar2 = tmp_path / "tmp2.par"
    with open(tmppar1, "w") as f:
        f.write(simplepar)

    cmd = f"{tmppar1} {tmppar2}"
    tcb2tdb.main(cmd.split())

    assert os.path.isfile(tmppar2)

    m2 = ModelBuilder()(tmppar2)
    assert m2.UNITS.value == "TDB"
