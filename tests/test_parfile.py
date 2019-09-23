#! /usr/bin/env python
import os
import sys
import tempfile

import pytest

import pint.models as tm
from pinttestdata import datadir, testdir

datadir = os.path.join(testdir, "datafile")
parfile = os.path.join(datadir, "J1744-1134.basic.par")


def demo_parfile():

    m = tm.get_model(parfile)

    print("model.param_help():")
    m.param_help()
    print()

    print("calling model.read_parfile():")
    m.read_parfile(parfile)
    print()

    print("print model:")
    print(m)
    print()

    print("model.as_parfile():")
    print(m.as_parfile())
    print()


params = tm.get_model(parfile).params


@pytest.fixture
def roundtrip():
    m = tm.get_model(parfile)
    with tempfile.NamedTemporaryFile("wt") as f:
        f.write(m.as_parfile())
        f.flush()
        m2 = tm.get_model(f.name)
    return m, m2


def test_roundtrip(roundtrip):
    m, m2 = roundtrip
    assert set(m.components.keys()) == set(m2.components.keys())
    assert set(m.params) == set(m2.params)


@pytest.mark.parametrize("p", params)
def test_roundtrip(roundtrip, p):
    m, m2 = roundtrip
    pm = getattr(m, p)
    pm2 = getattr(m2, p)
    assert type(pm) == type(pm2)
    assert pm.frozen == pm2.frozen
    assert pm.description == pm2.description
    if hasattr(pm, "units"):
        assert pm.units == pm2.units
    assert pm.uncertainty == pm2.uncertainty
    assert pm.value == pm2.value
