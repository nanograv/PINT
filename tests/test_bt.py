"""Various tests to assess the performance of the BT model."""
from __future__ import (print_function, division)
import pint.toa as toa
import numpy as np
import pint.models.bt as bt

class TestBT():
    """Various tests to assess the performance of the BT model."""

    def test_1955(self):
        """Compare delays from the BT model with libstempo and PINT"""

        parfile = 'tests/J1955.par'
        timfile = 'tests/J1955.tim'

        # Calculate delays with PINT
        toas = toa.get_TOAs(timfile, planets=False)
        newmodel = bt.BT()
        newmodel.read_parfile(parfile)
        pint_delays = newmodel.delay(toas.table)

        # Load delays calculated with libstempo
        _, lt_delays = np.genfromtxt('tests/J1955_ltdelays.dat', unpack=True)

        assert np.all(np.abs(pint_delays - lt_delays) < 1e-11), 'BT TEST FAILED'


if __name__ == '__main__':
    t = TestBT()

    t.test_1955()
