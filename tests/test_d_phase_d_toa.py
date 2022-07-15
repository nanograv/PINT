import numpy as np
import pytest

try:
    from erfa import DJM0
except ImportError:
    from astropy._erfa import DJM0

from pinttestdata import datadir

import pint.toa as toa
from pint.models import model_builder as mb
from pint.polycos import Polycos


@pytest.fixture
def setup(pickle_dir):
    class Setup:
        pass

    s = Setup()
    s.parfileB1855 = datadir / "B1855+09_polycos.par"
    s.timB1855 = datadir / "B1855_polyco.tim"
    s.toasB1855 = toa.get_TOAs(
        s.timB1855,
        ephem="DE405",
        planets=False,
        include_bipm=False,
        picklefilename=pickle_dir,
    )
    s.modelB1855 = mb.get_model(s.parfileB1855)
    # Read tempo style polycos.
    s.plc = Polycos()
    s.plc.read_polyco_file(datadir / "B1855_polyco.dat", "tempo")
    return s


def testD_phase_d_toa(setup):
    pint_d_phase_d_toa = setup.modelB1855.d_phase_d_toa(setup.toasB1855)
    mjd = np.array(
        [
            np.longdouble(t.jd1 - DJM0) + np.longdouble(t.jd2)
            for t in setup.toasB1855["mjd"]
        ]
    )
    tempo_d_phase_d_toa = setup.plc.eval_spin_freq(mjd)
    diff = pint_d_phase_d_toa.value - tempo_d_phase_d_toa
    relative_diff = diff / tempo_d_phase_d_toa
    assert np.all(np.abs(relative_diff) < 1e-7), "d_phase_d_toa test failed."
