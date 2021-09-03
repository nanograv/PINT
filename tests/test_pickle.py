"""Test pickle-ability of timing models"""
import astropy.units as u
import numpy as np

import pint.config
import pint.models.parameter as param
from pint.models.model_builder import get_model
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
    m = get_model("tests/datafile/J1713+0747_NANOGrav_11yv0.gls.par",)
    s = pickle.dumps(m)
