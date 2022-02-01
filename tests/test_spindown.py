from io import StringIO

import pytest

from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

par_base = """
    PSR J1235+5678
    F0 1
    ELAT 0
    ELONG 0
    PEPOCH 57000
"""


def test_fn_removal():
    m = get_model(StringIO("\n".join([par_base, "F1 0", "F2 0"])))
    m.remove_param("F2")
    make_fake_toas_uniform(57000, 58000, 2, m)


def test_no_f1():
    get_model(StringIO(par_base))


def test_missing_f1():
    with pytest.raises(ValueError):
        m = get_model(StringIO("\n".join([par_base, "F2 0"])))


def test_removed_f1():
    m = get_model(StringIO("\n".join([par_base, "F1 0"])))
    assert m.F1.value == 0
    m.remove_param("F1")
    m.validate()
    make_fake_toas_uniform(57000, 58000, 2, m)


def test_missing_f2():
    with pytest.raises(ValueError):
        get_model(StringIO("\n".join([par_base, "F3 0"])))
