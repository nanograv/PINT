"""Tests to ensure that ModelBuilder is able to read PLDMNoise 
component properly.

Tests that PLDMNoise reproduces same results as PLRedNoise using
monochromatic data

To do:
    Add test that fits out DM noise correctly in toy simulated dataset
    Add test that checks for same results between PINT and Tempo2/TempoNest
    Add test that checks for same results between PINT and enterprise
"""


import astropy.units as u
import contextlib
import io
import numpy as np
import pint.fitter as fitters
import pint.models.model_builder as mb
from pint.models.model_builder import get_model
from pint.models.timing_model import Component
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_fromMJDs
import pytest

parfile_contents = """
    PSRJ           J0023+0923
    RAJ             00:23:16.8790858         1  0.00002408141295805134   
    DECJ           +09:23:23.86936           1  0.00082010713730773120   
    F0             327.84702062954047136     1  0.00000000000295205483   
    F1             -1.2278326306812866375e-15 1  3.8219244605614075223e-19
    PEPOCH         56199.999797564144902       
    POSEPOCH       56199.999797564144902       
    DMEPOCH        56200                       
    DM             14.327978186774068347     1  0.00006751663559857748   
    BINARY         ELL1
    PB             0.13879914244858396754    1  0.00000000003514075083   
    A1             0.034841158415224894973   1  0.00000012173038389247   
    TASC           56178.804891768506529     1  0.00000007765191894742   
    EPS1           1.6508830631753595232e-05 1  0.00000477568412215803   
    EPS2           3.9656838708709247373e-06 1  0.00000458951091435993   
    CLK            TT(BIPM2015)
    MODE 1
    UNITS          TDB
    TIMEEPH        FB90
    DILATEFREQ     N
    PLANET_SHAPIRO N
    CORRECT_TROPOSPHERE  N
    EPHEM          DE436
    JUMP -fe 430 0 0
    TNEF -group 430_ASP 1.0389
    TNEQ -group 430_ASP -8.77109
    TNECORR -group 430_PUPPI 0.00490558
    TNRedAmp -13.3087
    TNRedGam 0.125393
    TNRedC 14
    TNDMAMP -14.2
    TNDMGAM 0.624
    TNDMC 70
"""


@pytest.fixture()
def modelJ0023p0923():
    return mb.get_model(io.StringIO(parfile_contents))


def test_read_PLDMNoise_component(modelJ0023p0923):
    assert "PLDMNoise" in modelJ0023p0923.components


def test_read_PLDMNoise_component_type(modelJ0023p0923):
    assert (
        modelJ0023p0923.components["PLDMNoise"] in modelJ0023p0923.NoiseComponent_list
    )


def test_read_PLDMNoise_params(modelJ0023p0923):
    params = ["TNDMAMP", "TNDMGAM", "TNDMC"]
    for param in params:
        assert (
            hasattr(modelJ0023p0923, param)
            and getattr(modelJ0023p0923, param).quantity is not None
        )


def test_PLRedNoise_recovery():
    # basic model, no EFAC or EQUAD
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0 1
        ELONG 0 1
        DM 10 0
        F0 1 1
        PEPOCH 58000
        POSEPOCH 58000
        PMELONG 0 1
        PMELAT 0 1
        PX 0 1
        TNEF mjd 57000 58000 1.0389
        TNEQ mjd 57000 58000 -8.77109
        TNRedAmp -11
        TNRedGam 3
        TNRedC 30
        """
        )
    )

    # simulate toas
    MJDs = np.linspace(57001, 58000, 150, dtype=np.longdouble) * u.d
    toas = make_fake_toas_fromMJDs(MJDs, model=model, error=1 * u.us, add_noise=True)

    # get red noise basis and weights
    rn_basis = model.components["PLRedNoise"].get_noise_basis(toas)
    rn_weights = model.components["PLRedNoise"].get_noise_weights(toas)

    # fit model with red noise
    f1 = fitters.DownhillGLSFitter(toas, model)
    f1.fit_toas()
    f1.model.validate()
    f1.model.validate_toas(toas)
    r1 = Residuals(toas, f1.model)

    # remove red noise
    A = model["TNREDAMP"].value
    gam = model["TNREDGAM"].value
    c = model["TNREDC"].value
    model.remove_component("PLRedNoise")

    # create and add PLDMNoise component
    all_components = Component.component_types
    noise_class = all_components["PLDMNoise"]
    noise = noise_class()  # Make the dispersion instance.
    model.add_component(noise, validate=False)
    model["TNDMAMP"].quantity = A
    model["TNDMGAM"].quantity = gam
    model["TNDMC"].value = c
    model.validate()

    # get DM noise basis and weights
    dm_basis = model.components["PLDMNoise"].get_noise_basis(toas)
    # DM basis will be off by a constant factor, depending on frequency
    # of the test data
    D = (1400 / toas.get_freqs().value) ** 2
    dm_basis_scaled = dm_basis / D[:, None]
    dm_weights = model.components["PLDMNoise"].get_noise_weights(toas)

    # refit model
    f2 = fitters.DownhillGLSFitter(toas, model)
    with contextlib.suppress(fitters.InvalidModelParameters, fitters.StepProblem):
        f2.fit_toas()
    f2.model.validate()
    f2.model.validate_toas(toas)
    r2 = Residuals(toas, f2.model)

    # check weights and basis are equivalent within error
    basis_diff = rn_basis - dm_basis_scaled
    weights_diff = rn_weights - dm_weights
    assert np.all(np.isclose(basis_diff, 0, atol=1e-3))
    assert np.all(np.isclose(weights_diff, 0))

    # check residuals are equivalent within error
    rs_diff = r2.time_resids.value - r1.time_resids.value
    assert np.all(np.isclose(rs_diff, 0, atol=1e-6))
