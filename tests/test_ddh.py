import numpy as np
from pint.models import get_model_and_toas, get_model
import pint.fitter
import pint.logging
import os
from pinttestdata import datadir
import pint.binaryconvert


def test_ddh():
    m, t = get_model_and_toas(
        os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"),
        os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim"),
    )
    m2 = pint.binaryconvert.convert_binary(m, "DDH")

    f = pint.fitter.Fitter.auto(t, m)
    f2 = pint.fitter.Fitter.auto(t, m2)
    f.fit_toas()
    f2.fit_toas()
    assert np.isclose(f.resids.calc_chi2(), f2.resids.calc_chi2())
