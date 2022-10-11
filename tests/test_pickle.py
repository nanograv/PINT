"""Test pickle-ability of timing models"""

import os

import pint.config
import pint.models.parameter as param
from pint.models.model_builder import get_model
from pinttestdata import datadir
import pickle


def test_pickle_prefixparameter():
    parfile = pint.config.examplefile("NGC6440E.par")
    m = get_model(parfile)

    modelcomponent = m.components["Spindown"]
    p = param.prefixParameter(
        parameter_type="float",
        name="F2",
        value=0,
        units=modelcomponent.F_unit(2),
        uncertainty=0,
        description=modelcomponent.F_description(2),
        longdouble=True,
        frozen=False,
    )
    modelcomponent.add_param(p, setup=True)
    m.validate()
    s = pickle.dumps(m)


def test_pickle_maskparameter():
    m = get_model(os.path.join(datadir, "J1713+0747_NANOGrav_11yv0.gls.par"))
    s = pickle.dumps(m)


def test_pickle_glitch():
    m = get_model(os.path.join(datadir, "prefixtest.par"))
    s = pickle.dumps(m)


def test_pickle_fd():
    m = get_model(os.path.join(datadir, "test_FD.par"))
    s = pickle.dumps(m)


def test_pickle_piecewise():
    m = get_model(os.path.join(datadir, "piecewise.par"))
    s = pickle.dumps(m)


def test_pickle_piecewise2():
    m = get_model(os.path.join(datadir, "piecewise_twocomps.par"))
    s = pickle.dumps(m)
