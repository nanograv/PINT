import io
import pytest
from astropy import units as u, constants as c
import numpy as np

import pint.models.parameter
from pint.models import get_model


base_par = """
PSR J1234+5678
ELAT 0
ELONG 10
F0 1
F1 
DM 10
PEPOCH 57000
"""


def test_funcpardefine():
    # just make sure we can define without an error
    p = pint.models.parameter.funcParameter(
        name="AGE",
        description="Spindown age",
        pars=("F0", "F1"),
        func=lambda f0, f1: f0 / 2 / f1,
        units="yr",
    )
    assert p.value is None


def test_funcpardefineandadd_undefined():

    m = get_model(io.StringIO(base_par))

    p = pint.models.parameter.funcParameter(
        name="AGE",
        description="Spindown age",
        pars=("F0", "F1"),
        func=lambda f0, f1: f0 / 2 / f1,
        units="yr",
    )
    m.components["Spindown"].add_param(p)
    # should still be None because F1 is not defined
    assert m.AGE.value is None


def test_funcpardefineandadd():

    m = get_model(io.StringIO(base_par))

    p = pint.models.parameter.funcParameter(
        name="AGE",
        description="Spindown age",
        pars=("F0", "F1"),
        func=lambda f0, f1: f0 / 2 / f1,
        units="yr",
    )
    m.components["Spindown"].add_param(p)
    m.F1.quantity = 3e-10 * u.Hz / u.s

    assert np.isclose(m.AGE.quantity, (1 * u.Hz / 2 / (3e-10 * u.Hz / u.s)).to(u.yr))


def test_funcpardefine_notquantity():

    m = get_model(io.StringIO(base_par))

    p = pint.models.parameter.funcParameter(
        name="AGE",
        description="Spindown age",
        pars=("F0", ("F1", "uncertainty")),
        func=lambda f0, f1: f0 / 2 / f1,
        units="yr",
    )
    m.components["Spindown"].add_param(p)
    m.F1.quantity = 3e-10 * u.Hz / u.s
    m.F1.uncertainty = 1e-10 * u.Hz / u.s

    assert np.isclose(m.AGE.quantity, (1 * u.Hz / 2 / (1e-10 * u.Hz / u.s)).to(u.yr))


def test_funcparfails():
    m = get_model(io.StringIO(base_par))

    # use F2 which doesn't exist
    # should raise an exception when figuring out parentage
    p = pint.models.parameter.funcParameter(
        name="AGE",
        description="Spindown age",
        pars=("F0", "F2"),
        func=lambda f0, f1: f0 / 2 / f1,
        units="yr",
    )
    m.components["Spindown"].add_param(p)
    m.F1.quantity = 3e-10 * u.Hz / u.s
    with pytest.raises(AttributeError):
        print(m.AGE)
