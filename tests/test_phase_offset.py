"Tests for the PhaseOffset component."

import numpy as np
import io

from pint.models import get_model_and_toas, get_model
from pint.models.phase_offset import PhaseOffset
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_uniform
from pint.fitter import WLSFitter

from pinttestdata import datadir

parfile = datadir / "NGC6440E.par"
timfile = datadir / "NGC6440E.tim"


def test_phase_offset():
    simplepar = """
    ELAT 5.6 1
    ELONG 3.2 1
    F0 100 1
    PEPOCH 50000
    PHOFF 0.2 1
    """
    m = get_model(io.StringIO(simplepar))

    assert hasattr(m, "PHOFF") and m.PHOFF.value == 0.2

    t = make_fake_toas_uniform(
        startMJD=50000,
        endMJD=50500,
        ntoas=100,
        model=m,
        add_noise=True,
    )

    res = Residuals(t, m)
    assert res.reduced_chi2 < 1.5

    ftr = WLSFitter(t, m)
    ftr.fit_toas(maxiter=3)
    assert ftr.resids.reduced_chi2 < 1.5


def test_phase_offset_designmatrix():
    m, t = get_model_and_toas(parfile, timfile)

    M1, pars1, units1 = m.designmatrix(t, incoffset=True)

    assert "Offset" in pars1

    offset_idx = pars1.index("Offset")
    M1_offset = M1[:, offset_idx]

    po = PhaseOffset()
    m.add_component(po)
    m.PHOFF.frozen = False

    M2, pars2, units2 = m.designmatrix(t, incoffset=True)

    assert "Offset" not in pars2
    assert "PHOFF" in pars2

    phoff_idx = pars2.index("PHOFF")
    M2_phoff = M2[:, phoff_idx]

    assert np.allclose(M1_offset, M2_phoff)
    assert M1.shape == M2.shape


def test_fit_real_data():
    m, t = get_model_and_toas(parfile, timfile)

    po = PhaseOffset()
    m.add_component(po)
    m.PHOFF.frozen = False

    ftr = WLSFitter(t, m)
    ftr.fit_toas(maxiter=3)

    assert ftr.resids.reduced_chi2 < 1.5

    assert abs(ftr.model.PHOFF.value) <= 0.5
