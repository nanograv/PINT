#!/usr/bin/python
import os
import unittest

import astropy.units as u
import numpy as np
from pinttestdata import datadir

import pint.models
import pint.toa
from pint.residuals import Residuals

parfile = os.path.join(datadir, "NGC6440E_PHASETEST.par")
timfile = os.path.join(datadir, "NGC6440E_PHASETEST.tim")


def test_phase_commands(pickle_dir):
    model = pint.models.get_model(parfile)
    toas = pint.toa.get_TOAs(timfile, picklefilename=pickle_dir)
    # This TOA has PHASE -0.3. Check that
    assert np.isclose(float(toas.table[32]["flags"]["phase"]), -0.3)
    # This TOA should have PHASE 0.2 and -padd -0.2
    assert np.isclose(float(toas.table["flags"][9]["padd"]), -0.2)
    assert np.isclose(float(toas.table["flags"][9]["phase"]), 0.2)
    # The end result should be these residuals if the commands are respected
    res = Residuals(toas=toas, model=model)
    assert (res.rms_weighted() - 1602.0293 * u.us) < 0.1 * u.us
