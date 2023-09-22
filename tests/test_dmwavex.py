from io import StringIO

import numpy as np

from pint.models import get_model
from pint.fitter import Fitter
from pint.simulation import make_fake_toas_uniform
from pint.utils import dmwavex_setup
from pint import dmu

import pytest
import astropy.units as u

par = """
    PSR              B1937+21
    LAMBDA   301.9732445337270
    BETA      42.2967523367957
    PMLAMBDA           -0.0175
    PMBETA             -0.3971
    PX                  0.1515
    POSEPOCH        55321.0000
    F0    641.9282333345536244  1  0.0000000000000132
    F1     -4.330899370129D-14  1  2.149749089617D-22
    PEPOCH        55321.000000
    DM               71.016633
    UNITS                  TDB
"""


def test_dmwavex():
    m = get_model(StringIO(par))

    with pytest.raises(ValueError):
        idxs = dmwavex_setup(m, 3600)

    idxs = dmwavex_setup(m, 3600, n_freqs=5)

    assert "DMWaveX" in m.components
    assert m.num_dmwavex_freqs == len(idxs)

    m.components["DMWaveX"].remove_dmwavex_component(5)
    assert m.num_dmwavex_freqs == len(idxs) - 1

    t = make_fake_toas_uniform(54000, 56000, 200, m, add_noise=True)

    ftr = Fitter.auto(t, m)
    ftr.fit_toas()

    assert ftr.resids.reduced_chi2 < 2


def test_dmwavex_badpar():
    with pytest.raises(ValueError):
        par = """
            PSR              B1937+21
            LAMBDA   301.9732445337270
            BETA      42.2967523367957
            PMLAMBDA           -0.0175
            PMBETA             -0.3971
            PX                  0.1515
            POSEPOCH        55321.0000
            F0    641.9282333345536244  1  0.0000000000000132
            F1     -4.330899370129D-14  1  2.149749089617D-22
            PEPOCH        55321.000000
            DM               71.016633
            UNITS                  TDB
            DMWXFREQ_0001         0.01
            DMWXSIN_0001             0
            DMWXSIN_0002             0
        """
        get_model(StringIO(par))

    with pytest.raises(ValueError):
        par = """
            PSR              B1937+21
            LAMBDA   301.9732445337270
            BETA      42.2967523367957
            PMLAMBDA           -0.0175
            PMBETA             -0.3971
            PX                  0.1515
            POSEPOCH        55321.0000
            F0    641.9282333345536244  1  0.0000000000000132
            F1     -4.330899370129D-14  1  2.149749089617D-22
            PEPOCH        55321.000000
            DM               71.016633
            UNITS                  TDB
            DMWXFREQ_0001         0.01
            DMWXCOS_0001             0
            DMWXCOS_0002             0
        """
        get_model(StringIO(par))

    with pytest.raises(ValueError):
        par = """
            PSR              B1937+21
            LAMBDA   301.9732445337270
            BETA      42.2967523367957
            PMLAMBDA           -0.0175
            PMBETA             -0.3971
            PX                  0.1515
            POSEPOCH        55321.0000
            F0    641.9282333345536244  1  0.0000000000000132
            F1     -4.330899370129D-14  1  2.149749089617D-22
            PEPOCH        55321.000000
            DM               71.016633
            UNITS                  TDB
            DMWXFREQ_0001         0.00
            DMWXCOS_0001             0
        """
        get_model(StringIO(par))


def test_add_dmwavex():
    m = get_model(StringIO(par))
    idxs = dmwavex_setup(m, 3600, n_freqs=5)

    with pytest.raises(ValueError):
        m.components["DMWaveX"].add_dmwavex_component(1, index=5, dmwxsin=0, dmwxcos=0)

    m.components["DMWaveX"].add_dmwavex_component(1, index=6, dmwxsin=0, dmwxcos=0)
    assert m.num_dmwavex_freqs == len(idxs) + 1

    m.components["DMWaveX"].add_dmwavex_component(
        1 / u.day, index=7, dmwxsin=0 * dmu, dmwxcos=0 * dmu
    )
    assert m.num_dmwavex_freqs == len(idxs) + 2

    m.components["DMWaveX"].add_dmwavex_component(2 / u.day)
    assert m.num_dmwavex_freqs == len(idxs) + 3

    m.components["DMWaveX"].add_dmwavex_components(
        np.array([3]) / u.day,
        dmwxsins=np.array([0]) * dmu,
        dmwxcoses=np.array([0]) * dmu,
    )
    assert m.num_dmwavex_freqs == len(idxs) + 4

    with pytest.raises(ValueError):
        m.components["DMWaveX"].add_dmwavex_components(
            [2 / u.day, 3 / u.day], dmwxsins=[0, 0], dmwxcoses=[0, 0, 0]
        )

    with pytest.raises(ValueError):
        m.components["DMWaveX"].add_dmwavex_components(
            [2 / u.day, 3 / u.day], dmwxsins=[0, 0, 0], dmwxcoses=[0, 0]
        )

    with pytest.raises(ValueError):
        m.components["DMWaveX"].add_dmwavex_components(
            [2 / u.day, 3 / u.day],
            dmwxsins=[0, 0],
            dmwxcoses=[0, 0],
            frozens=[False, False, False],
        )
