#!/usr/bin/env python
import os

import astropy.units as u
from pinttestdata import datadir

import pint.fitter
import pint.models
import pint.residuals
import pint.toa
from pint.models.model_builder import get_model


def test_barytoa():
    os.chdir(datadir)
    # This par file has a very simple model in it
    m = get_model("slug.par")

    # This .tim file has TOAs at the barycenter, and at infinite frequency
    t = pint.toa.get_TOAs("slug.tim")

    rs = pint.residuals.Residuals(t, m).time_resids

    # Residuals should be less than 2.0 ms
    assert rs.std() < 2.0 * u.ms
