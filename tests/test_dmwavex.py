from io import StringIO

from pint.models import get_model
from pint.fitter import Fitter
from pint.simulation import make_fake_toas_uniform
from pint.utils import dmwavex_setup

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

    idxs = dmwavex_setup(m, 3600, n_freqs=5)

    assert "DMWaveX" in m.components
    assert m.num_dmwavex_freqs == len(idxs)

    m.components["DMWaveX"].remove_dmwavex_component(5)
    assert m.num_dmwavex_freqs == len(idxs) - 1

    t = make_fake_toas_uniform(54000, 56000, 200, m, add_noise=True)

    ftr = Fitter.auto(t, m)
    ftr.fit_toas()

    assert ftr.resids.reduced_chi2 < 2
