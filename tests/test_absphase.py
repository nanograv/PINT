#!/usr/bin/python
import os
import unittest
import io

import pint.models
import pint.toa
from astropy import units as u
from pinttestdata import datadir

parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "zerophase.tim")


class TestAbsPhase(unittest.TestCase):
    def test_phase_zero(self):
        # Check that model phase is 0.0 for a TOA at exactly the TZRMJD
        model = pint.models.get_model(parfile)
        toas = pint.toa.get_TOAs(timfile)

        ph = model.phase(toas, abs_phase=True)
        # Check that integer and fractional phase values are very close to 0.0
        self.assertAlmostEqual(ph.int.value, 0.0)
        self.assertAlmostEqual(ph.frac.value, 0.0)


def test_offset():
    par = """
    F0 100 1
    OFFSET 0.5 1
    """
    m = pint.models.get_model(io.StringIO(par))
    assert hasattr(m, "OFFSET") and m.OFFSET.quantity == u.Quantity(0.5, "rad")
