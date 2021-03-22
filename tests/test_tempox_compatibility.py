from io import StringIO
import re

import astropy.units as u
import numpy as np
import pytest
from pint.models import get_model

par_basic = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
"""


@pytest.mark.parametrize(
    "name,want",
    [
        ("EFAC", "EFAC"),
        ("EQUAD", "EQUAD"),
        ("ECORR", "ECORR"),
        ("T2EFAC", "EFAC"),
        ("T2EQUAD", "EQUAD"),
        ("T2ECORR", "ECORR"),
    ],
)
def test_noise_parameter_aliases(name, want):
    parfile = "\n".join([par_basic, f"{name} -f bogus 1.234"])
    m = get_model(StringIO(parfile))
    assert getattr(m, want + "1").value == 1.234
    assert re.search(
        f"^{want}" + r"\s+-f\s+bogus\s+1.23\d+\s*$", m.as_parfile(), re.MULTILINE
    )
