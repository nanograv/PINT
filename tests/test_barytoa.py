import os

import astropy.units as u

import pint.fitter
import pint.models
from pint.models.model_builder import get_model
import pint.residuals
import pint.toa
from pinttestdata import datadir


def test_barytoa():
    os.chdir(datadir)
    # This par file has a very simple model in it
    m = get_model("slug.par")

    # This .tim file has TOAs at the barycenter, and at infinite frequency
    t = pint.toa.get_TOAs("slug.tim")

    rs = pint.residuals.Residuals(t, m).time_resids

    # Residuals should be less than 2.0 ms
    assert rs.std() < 2.0 * u.ms
