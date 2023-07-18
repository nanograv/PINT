from io import StringIO
import pytest
import numpy as np

from astropy import units as u
from pint.models import get_model, get_model_and_toas
from pint.models.timing_model import Component
from pint.fitter import Fitter
from pint.residuals import Residuals
from pint.toa import get_TOAs
from pint.simulation import make_fake_toas_uniform
import pint.utils
from pinttestdata import datadir

par1 = """
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

# Introduce a par file with WaveX already present

par2 = """
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
    WXEPOCH       55321.000000
    WXFREQ_0001            0.1
    WXSIN_0001              1
    WXCOS_0001              1 
    UNITS                  TDB
    """

wavex_par = """
    WXFREQ_0002            0.2
    WXSIN_0002              2
    WXCOS_0002              2 
    WXFREQ_0003            0.3
    WXSIN_0003              3
    WXCOS_0003              3 
"""


def wavex_delay(waves, toas):
    total_delay = np.zeros(toas.ntoas) * u.s
    wave_freqs = waves.get_prefix_mapping_component("WXFREQ_")
    wave_sins = waves.get_prefix_mapping_component("WXSIN_")
    wave_cos = waves.get_prefix_mapping_component("WXCOS_")

    base_phase = toas.table["tdbld"].value * u.d - waves.WXEPOCH.value * u.d
    for idx, param in wave_freqs.items():
        freq = getattr(waves, param).quantity
        wxsin = getattr(waves, wave_sins[idx]).quantity
        wxcos = getattr(waves, wave_cos[idx]).quantity
        arg = 2.0 * np.pi * freq * base_phase
        total_delay += wxsin * np.sin(arg.value) + wxcos * np.cos(arg.value)
    return total_delay


def test_wavex_resids_amp():
    # Check that the amplitude of residuals somewhat matches independent calculation of wave delay for a single component
    model = get_model(StringIO(par1))
    toas = make_fake_toas_uniform(55000, 55100, 100, model, obs="gbt")
    wave_model = get_model(StringIO(par2))
    rs = Residuals(toas, wave_model)
    injected_amp = np.sqrt(
        wave_model.components["WaveX"].WXSIN_0001.value ** 2
        + wave_model.components["WaveX"].WXCOS_0001.value ** 2
    )
    print(injected_amp, max(rs.resids.value))
    assert np.isclose(max(rs.resids.value), injected_amp, atol=1e-2)
    assert np.isclose(min(rs.resids.value), -injected_amp, atol=1e-2)


def test_multiple_wavex_resids_amp():
    # Check that residuals for multiple components match independent calculation
    model = get_model(StringIO(par1))
    toas = make_fake_toas_uniform(55000, 55100, 100, model, obs="gbt")
    wave_model = get_model(StringIO(par2 + wavex_par))
    rs = Residuals(toas, wave_model)
    wave_delays = wavex_delay(wave_model.components["WaveX"], toas)
    assert np.allclose(rs.resids, -wave_delays, atol=max(rs.resids.value) / 10.0)


def test_wavex_from_par():
    # Check that a par file with wavex components present produces expected indices
    model = get_model(StringIO(par2 + wavex_par))
    indices = model.components["WaveX"].get_indices()
    assert np.all(np.array(indices) == np.array([1, 2, 3]))


def test_add_wavex_to_par():
    # Add a wavex component to par file that has none and check against par file with some WaveX model
    model = get_model(StringIO(par1))
    toas = make_fake_toas_uniform(55000, 55100, 100, model, obs="gbt")
    model.add_component(Component.component_types["WaveX"]())
    index = model.components["WaveX"].get_indices()
    model.components["WaveX"].WXFREQ_0001.quantity = 0.1 * (1 / u.d)
    model.components["WaveX"].WXSIN_0001.quantity = 1 * u.s
    model.components["WaveX"].WXCOS_0001.quantity = 1 * u.s
    wavex_model = get_model(StringIO(par2))
    assert np.all(
        np.array(index) == np.array(wavex_model.components["WaveX"].get_indices())
    )
    assert np.all(
        model.components["WaveX"].wavex_delay(toas, 0.0 * u.s)
        == wavex_model.components["WaveX"].wavex_delay(toas, 0.0 * u.s)
    )


def test_add_then_remove_wavex():
    # Check that adding and then removing a wavex component actually gets rid of it
    model = get_model(StringIO(par2))
    model.components["WaveX"].add_wavex_component(0.2, index=2, wxsin=2, wxcos=2)
    indices = model.components["WaveX"].get_indices()
    model.components["WaveX"].remove_wavex_component(2)
    index = model.components["WaveX"].get_indices()
    assert np.all(np.array(len(indices)) != np.array(len(index)))


def test_multiple_wavex():
    # Check that when adding multiple wavex component pythonically is consistent with a par file with the same components
    model = get_model(StringIO(par2))
    toas = make_fake_toas_uniform(55000, 55100, 100, model, obs="gbt")
    wavex_model = get_model(StringIO(par2 + wavex_par))
    indices = model.components["WaveX"].add_wavex_components(
        [0.2, 0.3], indices=[2, 3], wxsins=[2, 3], wxcoses=[2, 3]
    )
    assert np.all(np.array(indices) == np.array([2, 3]))
    assert np.all(
        model.components["WaveX"].wavex_delay(toas, 0.0 * u.s)
        == wavex_model.components["WaveX"].wavex_delay(toas, 0.0 * u.s)
    )


def test_multiple_wavex_broadcast_frozens():
    # Check that when a single False is given for frozens, it gets broadcast to all the sine and cosine amplitudes
    model = get_model(StringIO(par2))
    indices = model.components["WaveX"].add_wavex_components(
        [0.2, 0.3],
        indices=[2, 3],
        wxsins=[2, 3],
        wxcoses=[2, 3],
        frozens=False,
    )
    for index in indices:
        assert getattr(model, f"WXSIN_{index:04d}").frozen == False
        assert getattr(model, f"WXCOS_{index:04d}").frozen == False


def test_multiple_wavex_wrong_cos_amps():
    # Check that code breaks when adding an extra cosine amplitude than there are frequencies, indices, and sine amplitudes for
    model = get_model(StringIO(par2))
    with pytest.raises(ValueError):
        indices = model.components["WaveX"].add_wavex_components(
            [0.2, 0.3], indices=[2, 3], wxsins=[2, 3], wxcoses=[2, 3, 4]
        )


def test_multiple_wavex_wrong_sin_amps():
    # Check that code breaks when adding an extra sine amplitude than there are frequencies, indices, and cosine amplitudes for
    model = get_model(StringIO(par2))
    with pytest.raises(ValueError):
        indices = model.components["WaveX"].add_wavex_components(
            [0.2, 0.3], indices=[2, 3], wxsins=[2, 3, 4], wxcoses=[2, 3]
        )


def test_multiple_wavex_wrong_freqs():
    # Check that code breaks when not adding enough frequencies for the number of indices, sine amps, and cosine amps given
    model = get_model(StringIO(par2))
    with pytest.raises(ValueError):
        indices = model.components["WaveX"].add_wavex_components(
            [0.2, 0.3], indices=[2, 3, 4], wxsins=[2, 3, 4], wxcoses=[2, 3, 4]
        )


def test_multiple_wavex_wrong_frozens():
    # Check that adding to many elements to frozens breaks code
    model = get_model(StringIO(par2))
    with pytest.raises(ValueError):
        indices = model.components["WaveX"].add_wavex_components(
            [0.2, 0.3],
            indices=[2, 3],
            wxsins=[2, 3],
            wxcoses=[2, 3],
            frozens=[False, False, False],
        )


def test_multiple_wavex_explicit_indices():
    # Check that adding specific indices is done correctly
    model = get_model(StringIO(par2))
    indices = model.components["WaveX"].add_wavex_components(
        [0.2, 0.3], indices=[3, 4], wxsins=[2, 3], wxcoses=[2, 3]
    )
    assert np.all(np.array(indices) == np.array([3, 4]))


def test_multiple_wavex_explicit_indices_duplicate():
    # Check that adding a duplicate index fails
    model = get_model(StringIO(par2))
    with pytest.raises(ValueError):
        indices = model.components["WaveX"].add_wavex_components(
            [0.2, 0.3], indices=[1, 3], wxsins=[2, 3], wxcoses=[2, 3]
        )
