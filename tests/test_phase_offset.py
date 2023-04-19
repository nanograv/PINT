import numpy as np

from pint.models import get_model_and_toas
from pint.models.phase_offset import PhaseOffset

from pinttestdata import datadir

parfile = datadir / "NGC6440E.par"
timfile = datadir / "NGC6440E.tim"


def test_phase_offset():
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
