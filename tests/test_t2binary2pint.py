"""Tests for `pint.models.tcb_conversion` and the `tcb2tdb` script."""

import os
import numpy as np

import pytest

from pint.models.model_builder import ModelBuilder
from pint.models.model_builder import _binary_model_priority
from pint.models.timing_model import UnknownBinaryModel
from pint.models.timing_model import AllComponents
from pint.scripts import t2binary2pint


ddkpar = """
PSR            PSRTEST
ELONG          256.66869638  1
ELAT           30.700359811  1
F0             218.81184039  1
F1             -4.08278e-16  1
PEPOCH         55636
POSEPOCH       55636
DMEPOCH        58983
DM             15.988527064  1
PMELONG        5.2676082172  1
PMELAT         -3.441494631  1
PX             0.8981169694  1
SINI            KIN
#SINI          0.9509308076719
BINARY         T2
PB             67.825136364  1
T0             54303.634515  1
A1             32.342422610  1
OM             176.19952646  1
ECC            7.494087e-05  1
PBDOT          1.958980e-14  1
OMDOT          0.0001539624  1
M2             0.3016563544  1
KOM            92.824444242  1
KIN            70.946081736  1
EPHEM          DE436
EPHVER         5
CLK            TT(BIPM2020)
T2CMETHOD      IAU2000B
NE_SW          0.000
TIMEEPH        FB90
CORRECT_TROPOSPHERE  Y
"""

incompatiblepar = """
PSR            PSRTEST
ELONG          256.66869638  1
ELAT           30.700359811  1
F0             218.81184039  1
F1             -4.08278e-16  1
PEPOCH         55636
POSEPOCH       55636
SINI            KIN
#SINI          0.9509308076719
BINARY         T2
PB             67.825136364  1
T0             54303.634515  1
A1             32.342422610  1
OM             176.19952646  1
EPS1           0.1237542123  1
M2             0.3016563544  1
KOM            92.824444242  1
KIN            70.946081736  1
EPHEM          DE436
EPHVER         5
CLK            TT(BIPM2020)
T2CMETHOD      IAU2000B
NE_SW          0.000
TIMEEPH        FB90
CORRECT_TROPOSPHERE  Y
"""


def test_t2binary2pint(tmp_path):
    tmppar1 = tmp_path / "tmp1.par"
    tmppar2 = tmp_path / "tmp2.par"
    with open(tmppar1, "w") as f:
        f.write(ddkpar)

    cmd = f"{tmppar1} {tmppar2}"
    t2binary2pint.main(cmd.split())

    assert os.path.isfile(tmppar2)

    m2 = ModelBuilder()(tmppar2)
    assert m2.BINARY.value == "DDK"
    assert np.isclose(m2.KIN.value, 109.053918264)
    assert np.isclose(m2.KOM.value, -2.8244442419)


def test_binary_model_priority():
    all_components = AllComponents()
    binary_models = all_components.category_component_map["pulsar_system"]
    binary_models = [
        all_components.components[m].binary_model_name for m in binary_models
    ]

    msg = (
        "Please update _binary_model_priority in model_builder.py "
        "to include all binary models"
    )

    assert set(binary_models) - set(_binary_model_priority) == set(), msg


def test_binary_exception(tmp_path):
    tmppar = tmp_path / "tmp1.par"
    with open(tmppar, "w") as f:
        f.write(incompatiblepar)

    with pytest.raises(UnknownBinaryModel):
        m = ModelBuilder()(tmppar)
