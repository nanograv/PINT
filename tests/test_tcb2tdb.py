"""Tests for `pint.models.tcb_conversion` and the `tcb2tdb` script."""

import os
from copy import deepcopy
from io import StringIO

import numpy as np
import pytest

from pint import DMconst, dmu
from pint.models.model_builder import ModelBuilder, get_model
from pint.models.tcb_conversion import convert_tcb_tdb
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
FD1                 1e-3
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TCB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""


@pytest.mark.parametrize("backwards", [True, False])
def test_convert_units(backwards):
    with pytest.raises(ValueError):
        m = ModelBuilder()(StringIO(simplepar))

    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    f0_tcb = m.F0.value
    pb_tcb = m.PB.value
    convert_tcb_tdb(m, backwards=backwards)
    assert m.UNITS.value == ("TCB" if backwards else "TDB")
    assert np.isclose(m.F0.value / f0_tcb, pb_tcb / m.PB.value)


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
            assert np.isclose(getattr(m, par).value, getattr(m_, par).value)


def test_effective_dimensionality():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb=True)
    assert m.PEPOCH.effective_dimensionality == 1
    assert m.F0.effective_dimensionality == -1
    assert m.F1.effective_dimensionality == -2

    assert m.POSEPOCH.effective_dimensionality == 1
    assert m.RAJ.effective_dimensionality == 0
    assert m.DECJ.effective_dimensionality == 0
    assert m.PMRA.effective_dimensionality == -1
    assert m.PMDEC.effective_dimensionality == -1
    assert m.PX.effective_dimensionality == -1

    assert m.DM.effective_dimensionality == -1

    assert m.T0.effective_dimensionality == 1
    assert m.A1.effective_dimensionality == 1
    assert m.ECC.effective_dimensionality == 0
    assert m.OM.effective_dimensionality == 0
    assert m.PB.effective_dimensionality == 1

    assert m.NE_SW.effective_dimensionality == -2


def test_dm_scaling_factor():
    m = get_model(
        StringIO(
            """
            PSR         TEST
            F0          100
            F1          -1e-14
            PEPOCH      55000
            DMEPOCH     55000
            DM          12.5
            DM1         -0.001
            DM2         1e-5
            DMXR1_0001  51000
            DMXR2_0001  51000
            DMX_0001    0.002
            DMWXEPOCH   55000
            DMWXFREQ_0001   0.001
            DMWXSIN_0001    0.0003
            DMWXCOS_0001    0.0002
            """
        )
    )

    for param in m.params:
        par = m[param]

        if hasattr(par, "units") and par.units == dmu:
            assert not par.convert_tcb2tdb or par.tcb2tdb_scale_factor == DMconst


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
