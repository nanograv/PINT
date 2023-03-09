import os
import pytest
import io
import copy

import astropy.units as u
import astropy.constants as c
import numpy as np

from pint.models import get_model
import pint.simulation
import pint.fitter
import pint.toa as toa
from pinttestdata import datadir

par = """PSR                            J1713+0747
EPHEM                               DE421
CLK                          TT(BIPM2015)
UNITS                                 TDB
START              53393.5600000000000001
FINISH             57387.6630000000000000
INFO                                   -f
TIMEEPH                              FB90
DILATEFREQ                              N
DMDATA                                  N
NTOA                                27571
CHI2                                  0.0
LAMBDA                256.668695240569491 1 0.00000000233510000000
BETA                   30.700360493706000 1 0.00000000414660000000
PMLAMBDA                           5.2671 1 0.0021
PMBETA                            -3.4428 1 0.0043
PX                                 0.8211 1 0.0258
ECL                              IERS2003
POSEPOCH           55391.0000000000000000
F0                    218.811843796082627 1 9.88e-14
F1                    -4.083888637248e-16 1 1.433249826455e-21
PEPOCH             55391.0000000000000000
CORRECT_TROPOSPHERE                         N
PLANET_SHAPIRO                          N
SOLARN0                               0.0
SWM                                   0.0
DM                              15.917131
BINARY                               ELL1
PB                      67.82512992243068 1 1.3525e-09
PBDOT                                 0.0
A1                           32.342422803 1 1.1e-07
A1DOT                                 0.0
TASC               55355.6398764915440722 1 0.00034922166676285590174
EPS1            -7.477570709820452097e-05 1 6.09373573558539801e-10
EPS2             4.968617143987452578e-06 1 1.7110028537492447898e-09
"""


class TestELL1plus:
    def setup(self):
        self.m = get_model(io.StringIO(par))
        self.mplus = get_model(
            io.StringIO(self.m.as_parfile().replace("ELL1", "ELL1+"))
        )
        self.t = pint.simulation.make_fake_toas_uniform(
            50000, 55000, 300, self.m, add_noise=True
        )

    def test_ell1plus_delay(self):
        ELL1_delay = self.m.binarymodel_delay(self.t, None)
        ELL1plus_delay = self.mplus.binarymodel_delay(self.t, None)
        ELL1_delayR = self.m.components["BinaryELL1"].binary_instance.delayR()
        ELL1plus_delayR = self.mplus.components[
            "BinaryELL1plus"
        ].binary_instance.delayR()
        Phi = self.m.components["BinaryELL1"].binary_instance.Phi()
        eps1 = self.m.components["BinaryELL1"].binary_instance.eps1()
        eps2 = self.m.components["BinaryELL1"].binary_instance.eps2()
        a1 = self.m.components["BinaryELL1"].binary_instance.a1()
        extra_delay = -(a1 / c.c / 8) * (
            5 * eps1**2 * np.sin(Phi)
            - 3 * eps1**2 * np.sin(3 * Phi)
            - 2 * eps1 * eps2 * np.cos(Phi)
            + 6 * eps1 * eps2 * np.cos(3 * Phi)
            + 3 * eps2**2 * np.sin(Phi)
            + 3 * eps2**2 * np.sin(3 * Phi)
        )
        assert np.allclose(ELL1_delayR - ELL1plus_delayR, -extra_delay)
        assert np.allclose(ELL1_delay - ELL1plus_delay, -extra_delay)

    def test_ell1plusfit(self):
        fplus = pint.fitter.Fitter.auto(self.t, self.mplus)
        fplus.fit_toas()
