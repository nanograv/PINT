from io import StringIO

import numpy as np

from pint.models import get_model
from pint.fitter import Fitter
from pint.simulation import make_fake_toas_uniform
from pint.utils import cmwavex_setup
from pint.models.chromatic_model import cmu

import pytest
import astropy.units as u


def test_cmwavex():
    par = """`
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
        CM                     0.1
        TNCHROMIDX               4
        UNITS                  TDB
    """
    m = get_model(StringIO(par))

    with pytest.raises(ValueError):
        idxs = cmwavex_setup(m, 3600)

    idxs = cmwavex_setup(m, 3600, n_freqs=5)

    assert "CMWaveX" in m.components
    assert m.num_cmwavex_freqs == len(idxs)

    m.components["CMWaveX"].remove_cmwavex_component(5)
    assert m.num_cmwavex_freqs == len(idxs) - 1

    t = make_fake_toas_uniform(54000, 56000, 200, m, add_noise=True)

    ftr = Fitter.auto(t, m)
    ftr.fit_toas()

    assert ftr.resids.reduced_chi2 < 2


def test_cmwavex_badpar():
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
            CM                     0.1
            TNCHROMIDX               4
            UNITS                  TDB
            CMWXFREQ_0001         0.01
            CMWXSIN_0001             0
            CMWXSIN_0002             0
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
            CM                     0.1
            TNCHROMIDX               4
            UNITS                  TDB
            CMWXFREQ_0001         0.01
            CMWXCOS_0001             0
            CMWXCOS_0002             0
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
            CM                     0.1
            TNCHROMIDX               4
            UNITS                  TDB
            CMWXFREQ_0001         0.00
            CMWXCOS_0001             0
        """
        get_model(StringIO(par))


def test_add_cmwavex():
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
        CM                     0.1
        TNCHROMIDX               4
        UNITS                  TDB
    """
    m = get_model(StringIO(par))
    idxs = cmwavex_setup(m, 3600, n_freqs=5)

    with pytest.raises(ValueError):
        m.components["CMWaveX"].add_cmwavex_component(1, index=5, cmwxsin=0, cmwxcos=0)

    m.components["CMWaveX"].add_cmwavex_component(1, index=6, cmwxsin=0, cmwxcos=0)
    assert m.num_cmwavex_freqs == len(idxs) + 1

    m.components["CMWaveX"].add_cmwavex_component(
        1 / u.day, index=7, cmwxsin=0 * cmu, cmwxcos=0 * cmu
    )
    assert m.num_cmwavex_freqs == len(idxs) + 2

    m.components["CMWaveX"].add_cmwavex_component(2 / u.day)
    assert m.num_cmwavex_freqs == len(idxs) + 3

    m.components["CMWaveX"].add_cmwavex_components(
        np.array([3]) / u.day,
        cmwxsins=np.array([0]) * cmu,
        cmwxcoses=np.array([0]) * cmu,
    )
    assert m.num_cmwavex_freqs == len(idxs) + 4

    with pytest.raises(ValueError):
        m.components["CMWaveX"].add_cmwavex_components(
            [2 / u.day, 3 / u.day], cmwxsins=[0, 0], cmwxcoses=[0, 0, 0]
        )

    with pytest.raises(ValueError):
        m.components["CMWaveX"].add_cmwavex_components(
            [2 / u.day, 3 / u.day], cmwxsins=[0, 0, 0], cmwxcoses=[0, 0]
        )

    with pytest.raises(ValueError):
        m.components["CMWaveX"].add_cmwavex_components(
            [2 / u.day, 3 / u.day],
            cmwxsins=[0, 0],
            cmwxcoses=[0, 0],
            frozens=[False, False, False],
        )
