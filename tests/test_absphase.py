#!/usr/bin/python
import os
import unittest

import numpy as np
from pinttestdata import datadir

import pint.models
import pint.toa

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "zerophase.tim")


def test_phase_zero(pickle_dir):
    # Check that model phase is 0.0 for a TOA at exactly the TZRMJD
    model = pint.models.get_model(parfile)
    toas = pint.toa.get_TOAs(timfile, picklefilename=pickle_dir)

    ph = model.phase(toas, abs_phase=True)
    # Check that integer and fractional phase values are very close to 0.0
    assert np.isclose(ph[0].value, 0.0)
    assert np.isclose(ph[1].value, 0.0)
