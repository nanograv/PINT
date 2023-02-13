from pint.models.tcb_conversion import convert_tcb_to_tdb, IFTE_K
from pint.models import get_model
from pint.scripts import tcb2tdb
from io import StringIO
import pytest
import numpy as np
import os

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


def test_convert_to_tdb():
    with pytest.raises(ValueError):
        m = get_model(StringIO(simplepar))

    m = get_model(StringIO(simplepar), allow_tcb=True)
    f0_tcb = m.F0.value
    pb_tcb = m.PB.value
    convert_tcb_to_tdb(m)
    assert m.UNITS.value == "TDB"
    assert np.isclose(m.F0.value / f0_tcb, pb_tcb / m.PB.value)


def test_tcb2tdb(tmp_path):
    tmppar1 = tmp_path / "tmp1.par"
    tmppar2 = tmp_path / "tmp2.par"
    with open(tmppar1, "w") as f:
        f.write(simplepar)

    cmd = f"{tmppar1} {tmppar2}"
    tcb2tdb.main(cmd.split())

    assert os.path.isfile(tmppar2)

    m2 = get_model(tmppar2)
    assert m2.UNITS.value == "TDB"
