"""Test for pint solar wind module"""

import os
import copy
import warnings
from io import StringIO
import pytest
import numpy as np
from numpy.testing import assert_allclose

from astropy.time import Time
from astropy import units as u
from pint.models import get_model, get_model_and_toas
from pint.fitter import Fitter
from pint.simulation import make_fake_toas_uniform
from pint.models.solar_wind_dispersion import SolarWindDispersionX, SolarWindProxyRegression
import pint.utils
from pinttestdata import datadir


par = """
PSR J1234+5678
F0 1
DM 10
ELAT 0
ELONG 0
PEPOCH 54000
"""

# a model with ELAT > 0
par2 = """
PSR J1234+5678
F0 1
DM 10
ELAT 10
ELONG 0
PEPOCH 54000
"""

march_equinox = Time("2015-03-20 22:45:00").mjd
year = 365.25  # in mjd


@pytest.mark.parametrize("frac", [0, 0.25, 0.5, 0.123])
def test_sun_angle_ecliptic(frac):
    model = get_model(StringIO(par))
    toas = make_fake_toas_uniform(
        march_equinox, march_equinox + 2 * year, 10, model=model, obs="gbt"
    )
    # Sun longitude, from Astronomical Almanac
    sun_n = toas.get_mjds().value - 51544.5
    sun_L = 280.460 + 0.9856474 * sun_n
    sun_g = 357.528 + 0.9856003 * sun_n
    sun_longitude = (
        sun_L
        + 1.915 * np.sin(np.deg2rad(sun_g))
        + 0.020 * np.sin(2 * np.deg2rad(sun_g))
    )
    sun_longitude = (sun_longitude + 180) % 360 - 180
    angles = np.rad2deg(model.sun_angle(toas).value)
    assert_allclose(angles, np.abs(sun_longitude), atol=1)


def test_conjunction():
    model = get_model(StringIO(par))
    t0, elongation = pint.utils.get_conjunction(
        model.get_psr_coords(),
        Time(march_equinox, format="mjd") - 2 * u.d,
        precision="high",
    )
    assert np.isclose(t0.mjd, 57102.1792, atol=1e-4)
    assert np.isclose(elongation, 0 * u.deg, atol=0.1 * u.deg)


def test_solar_wind_delays_positive():
    model = get_model(StringIO("\n".join([par, "NE_SW 1"])))
    toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")
    assert np.all(model.components["SolarWindDispersion"].solar_wind_dm(toas) > 0)


def test_solar_wind_generalmodel():
    # default model
    model = get_model(StringIO("\n".join([par, "NE_SW 1"])))
    # model with general power-law index
    model2 = get_model(StringIO("\n".join([par, "NE_SW 1\nSWM 1"])))
    toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")

    assert np.allclose(
        model2.components["SolarWindDispersion"].solar_wind_delay(toas),
        model.components["SolarWindDispersion"].solar_wind_delay(toas),
    )


def test_solar_wind_generalmodel_deriv():
    # default model
    model = get_model(StringIO("\n".join([par, "NE_SW 1"])))
    # model with general power-law index but the default is p==2 (same as SWM==0)
    model2 = get_model(StringIO("\n".join([par, "NE_SW 1\nSWM 1"])))
    toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")

    assert np.allclose(
        model2.components["SolarWindDispersion"].d_dm_d_ne_sw(toas, "NE_SW").to(u.cm),
        model.components["SolarWindDispersion"].d_dm_d_ne_sw(toas, "NE_SW").to(u.cm),
    )


def test_solar_wind_swm2():
    # should fail for SWM != 0 or 1
    model = get_model(StringIO("\n".join([par, "NE_SW 1\nSWM 2"])))
    with pytest.raises(NotImplementedError):
        toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")


def test_solar_wind_generalmodel_p1():
    # model with general power-law index
    model = get_model(StringIO("\n".join([par, "NE_SW 1\nSWM 1\nSWP 1"])))
    with pytest.raises(NotImplementedError):
        toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")


def test_swx_frozen():
    # SWX model with a single segment to match the default model
    model = get_model(
        StringIO(
            "\n".join(
                [par2, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
            )
        )
    )
    assert model.SWXP_0001.frozen is True
    assert model.SWXDM_0001.frozen is True
    model = get_model(
        StringIO(
            "\n".join(
                [
                    par2,
                    "SWXDM_0001 1\nSWXP_0001 2 1\nSWXR1_0001 53999\nSWXR2_0001 55000",
                ]
            )
        )
    )
    assert model.SWXP_0001.frozen is False
    model.add_swx_range(54000, 54180, swxdm=10, swxp=1.5, frozen=False)
    assert model.SWXP_0002.frozen is True
    assert model.SWXDM_0002.frozen is False


def test_swx_empty():
    model = get_model(
        StringIO(
            "\n".join(
                [
                    par2,
                    "SWXDM_0001 1\nSWXP_0001 2 1\nSWXR1_0001 54000\nSWXR2_0001 55000",
                ]
            )
        )
    )
    model.add_swx_range(55000, 55500, swxdm=10, swxp=1.5, frozen=False)
    model.add_swx_range(55500, 55501, swxdm=10, swxp=1.5, frozen=False)
    t = make_fake_toas_uniform(54000, 56000, 100, model)
    tofreeze = model.find_empty_masks(t)
    assert "SWXDM_0003" in tofreeze


def test_swx_minmax():
    # default model
    model = get_model(StringIO("\n".join([par2, "NE_SW 1"])))
    # SWX model with a single segment to match the default model
    model2 = get_model(
        StringIO(
            "\n".join(
                [par2, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
            )
        )
    )
    # because of the way SWX is scaled, scale the input
    scale = model2.get_swscalings()[0]
    model2.SWXDM_0001.quantity = model.get_max_dm() * scale
    assert np.isclose(model.get_max_dm(), model2.get_max_dms()[0] / scale)
    assert np.isclose(model.get_min_dm(), model2.get_min_dms()[0] / scale)


def test_swx_dm():
    # default model
    model = get_model(StringIO("\n".join([par2, "NE_SW 1"])))
    # SWX model with a single segment to match the default model
    model2 = get_model(
        StringIO(
            "\n".join(
                [par2, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
            )
        )
    )
    # because of the way SWX is scaled, scale the input
    scale = model2.get_swscalings()[0]
    model2.SWXDM_0001.quantity = model.get_max_dm() * scale
    toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")
    assert np.allclose(
        model2.swx_dm(toas) + model.get_min_dm(), model.solar_wind_dm(toas)
    )


def test_swx_delay():
    # default model
    model = get_model(StringIO("\n".join([par2, "NE_SW 1"])))
    # SWX model with a single segment to match the default model
    model2 = get_model(
        StringIO(
            "\n".join(
                [par2, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
            )
        )
    )
    scale = model2.get_swscalings()[0]
    model2.SWXDM_0001.quantity = model.get_max_dm() * scale
    # change the DM to handle the minimum
    model2.DM.quantity = model.get_min_dm()
    toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")
    assert np.allclose(
        model2.swx_delay(toas)
        + model2.components["DispersionDM"].dispersion_type_delay(toas),
        model.solar_wind_delay(toas),
    )


# def test_swx_derivs():
#     # default model
#     model = get_model(StringIO("\n".join([par2, "NE_SW 1"])))
#     # SWX model with a single segment to match the default model
#     model2 = get_model(
#         StringIO(
#             "\n".join(
#                 [par2, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
#             )
#         )
#     )
#     model2.SWXDM_0001.quantity = model.get_max_dm() * model2.get_swscalings()[0]
#     toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")
#     assert np.allclose(
#         model2.d_dm_d_param(toas, "SWXDM_0001")
#         * model2.conjunction_solar_wind_geometry(model2.SWXP_0001.value),
#         model.d_dm_d_param(toas, "NE_SW"),
#     )
#     assert np.allclose(
#         model2.d_delay_d_param(toas, "SWXDM_0001")
#         * model2.fiducial_solar_wind_geometry(model2.SWXP_0001.value),
#         model.d_delay_d_param(toas, "NE_SW"),
#     )


def test_swx_getset():
    # default model
    model = get_model(StringIO("\n".join([par2, "NE_SW 1"])))
    # SWX model with a single segment to match the default model
    model2 = get_model(
        StringIO(
            "\n".join(
                [par2, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
            )
        )
    )
    model2.SWXDM_0001.quantity = model.get_max_dm()
    toas = make_fake_toas_uniform(54000, 54000 + year, 13, model=model, obs="gbt")
    assert np.isclose(model2.get_ne_sws()[0], model.NE_SW.quantity)
    model2.set_ne_sws(model.NE_SW.quantity * 2)
    assert np.isclose(model2.get_ne_sws()[0], 2 * model.NE_SW.quantity)
    model2.set_ne_sws(np.array([model.NE_SW.value * 2]) * model.NE_SW.quantity.unit)
    assert np.isclose(model2.get_ne_sws()[0], 2 * model.NE_SW.quantity)
    with pytest.raises(ValueError):
        model2.set_ne_sws(
            np.array([model.NE_SW.value * 2, model.NE_SW.value])
            * model.NE_SW.quantity.unit
        )


def test_swfit_swm1():
    # default model and TOAs
    model0, t = get_model_and_toas(
        os.path.join(datadir, "2145_swfit.par"), os.path.join(datadir, "2145_swfit.tim")
    )
    # default SWM=0 model
    model1 = copy.deepcopy(model0)
    model1.SWM.value = 0
    # SWM=1 model with p=2
    model2 = copy.deepcopy(model0)
    model2.SWM.value = 1
    model2.SWP.value = 2
    model2.SWP.frozen = True

    f1 = Fitter.auto(t, model1)
    f2 = Fitter.auto(t, model2)

    f1.fit_toas()
    f2.fit_toas()
    assert np.isclose(f1.model.NE_SW.value, f2.model.NE_SW.value)


def test_swfit_swx():
    # default model and TOAs
    model0, t = get_model_and_toas(
        os.path.join(datadir, "2145_swfit.par"), os.path.join(datadir, "2145_swfit.tim")
    )
    # default SWM=0 model
    model1 = copy.deepcopy(model0)
    model1.SWM.value = 0
    # SWX model
    model3 = copy.deepcopy(model0)
    model3.remove_component("SolarWindDispersion")
    model3.add_component(SolarWindDispersionX())
    model3.SWXR1_0001.value = t.get_mjds().min().value
    model3.SWXR2_0001.value = t.get_mjds().max().value
    model3.SWXP_0001.value = 2
    model3.SWXDM_0001.value = 1e-3
    model3.SWXDM_0001.frozen = False

    f1 = Fitter.auto(t, model1)
    f3 = Fitter.auto(t, model3)

    f1.fit_toas()
    f3.fit_toas()
    # make sure to scale SWX model for different definition
    assert np.isclose(
        f1.model.NE_SW.value,
        f3.model.get_ne_sws()[0].value / f3.model.get_swscalings()[0],
    )


def test_swp_fit():
    model = get_model(StringIO("\n".join([par2, "NE_SW 1\nSWM 1\nSWP 2"])))
    toas = make_fake_toas_uniform(54000, 54000 + 365, 153, model=model, obs="gbt")
    for param in model.free_params:
        getattr(model, param).frozen = True
    model.SWP.value = 2.5
    model.SWP.frozen = False
    f = Fitter.auto(toas, model)
    f.fit_toas()
    assert np.isclose(f.model.SWP.value, 2, atol=0.5)


def test_swxp_fit():
    model = get_model(StringIO(par2))
    model.add_component(SolarWindDispersionX())
    model.SWXR1_0001.value = 54000
    model.SWXR2_0001.value = 54365
    model.SWXDM_0001.value = 1e-3
    model.SWXP_0001.value = 2
    toas = make_fake_toas_uniform(54000, 54000 + 365, 153, model=model, obs="gbt")
    for param in model.free_params:
        getattr(model, param).frozen = True
    model.SWXP_0001.value = 2.5
    model.SWXP_0001.frozen = False
    f = Fitter.auto(toas, model)
    f.fit_toas()
    assert np.isclose(f.model.SWXP_0001.value, 2, atol=0.1)


def test_overlapping_swx():
    model = get_model(StringIO(par2))
    model.add_component(SolarWindDispersionX())
    model.SWXR1_0001.value = 54000
    model.SWXR2_0001.value = 54365
    model.SWXDM_0001.value = 10
    model.SWXP_0001.value = 5
    toas = make_fake_toas_uniform(54000, 54000 + 365, 150, model=model, obs="gbt")

    dm = model.components["SolarWindDispersionX"].swx_dm(toas)

    model2 = copy.deepcopy(model)
    model3 = copy.deepcopy(model)

    model2.add_swx_range(54000, 54180, swxdm=10, swxp=1.5)
    dm2 = model2.components["SolarWindDispersionX"].swx_dm(toas)

    model3.SWXR2_0001.value = 54180
    model3.SWXP_0001.value = 1.5
    dm3 = model3.components["SolarWindDispersionX"].swx_dm(toas)
    dm3[toas.get_mjds() >= 54180 * u.d] = 0
    assert np.allclose(dm2 - dm, dm3)


def test_nesw_derivatives():
    par = """
        PSRJ            J1744-1134
        ELONG           266.119458498                 6.000e-09
        ELAT            11.80517508                   3.000e-08
        DM              3.1379                        4.000e-04
        PEPOCH          55000
        F0              245.4261196898081             5.000e-13
        F1              -5.38156E-16                  3.000e-21
        POSEPOCH        59150
        DMEPOCH         55000
        CLK             TT(BIPM2018)
        EPHEM           DE436
        RM              1.62                          9.000e-02
        PX              2.61                          9.000e-02
        DM1             -7.0E-5                       3.000e-05
        PMELONG         19.11                         2.000e-02
        DM2             1.7E-5                        4.000e-06
        PMELAT          -9.21                         1.300e-01
        UNITS           TDB        
        NE_SW           8           1
        NE_SW1          0           1
        NE_SW2          0           1
        SWEPOCH         55000
    """
    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(54500, 55500, 1000, m, add_noise=True)
    ftr = Fitter.auto(t, m)
    ftr.fit_toas()

    assert (
        m.NE_SW.value - ftr.model.NE_SW.value
    ) / ftr.model.NE_SW.uncertainty_value < 3
    assert (
        m.NE_SW1.value - ftr.model.NE_SW1.value
    ) / ftr.model.NE_SW1.uncertainty_value < 3
    assert (
        m.NE_SW2.value - ftr.model.NE_SW2.value
    ) / ftr.model.NE_SW2.uncertainty_value < 3

    m.components["SolarWindDispersion"]


def test_expression():
    par = """
        PSRJ            J1744-1134
        ELONG           266.119458498                 6.000e-09
        ELAT            11.80517508                   3.000e-08
        DM              3.1379                        4.000e-04
        PEPOCH          55000
        F0              245.4261196898081             5.000e-13
        F1              -5.38156E-16                  3.000e-21
        POSEPOCH        59150
        DMEPOCH         55000
        CLK             TT(BIPM2018)
        EPHEM           DE436
        PX              2.61                          9.000e-02
        PMELONG         19.11                         2.000e-02
        PMELAT          -9.21                         1.300e-01
        UNITS           TDB        
        NE_SW           8           1
        NE_SW1          1           1
        SWEPOCH         55000
    """
    m = get_model(StringIO(par))
    t = make_fake_toas_uniform(54500, 55500, 10, m, add_noise=True)

    t0 = m.SWEPOCH.value * u.day
    t1 = t["tdbld"][-1] * u.day
    dt = t1 - t0

    assert np.isclose(
        (
            m.components["SolarWindDispersion"].solar_wind_dm(t)
            / m.components["SolarWindDispersion"].solar_wind_geometry(t)
        )[-1],
        (m.NE_SW.quantity + dt * m.NE_SW1.quantity),
    )

# ---------------------------------------------------------------------------
# SolarWindProxyRegression tests
# ---------------------------------------------------------------------------

# Par string shared by proxy tests.  ELAT != 0 so S(t) is not degenerate.
_PROXY_PAR = """
PSR J1234+5678
F0 1
DM 10
ELAT 10
ELONG 0
PEPOCH 54000
"""

_TSTART = 54000
_TEND = 54000 + 365.25 * 2  # two years

_NE_SW_TRUE = 7.9  # cm^-3
_BETA1_TRUE = 2.1  # cm^-3 (per unit normalised proxy)


def _make_proxy(toas):
    """Return (proxy_mjd, proxy_vals) — a sinusoidal solar-cycle proxy."""
    mjds = np.linspace(_TSTART - 100, _TEND + 100, 2000)
    vals = 5.0 + 3.0 * np.sin(2 * np.pi * (mjds - _TSTART) / (11 * 365.25))
    return mjds, vals


def _proxy_model(ne_sw=_NE_SW_TRUE, beta1=_BETA1_TRUE, swm=0):
    """Build a SolarWindProxyRegression model with one proxy loaded."""
    base = get_model(StringIO(_PROXY_PAR))
    comp = SolarWindProxyRegression()
    base.add_component(comp)
    base.NE_SW.value = ne_sw
    base.SWPRBETA1.value = beta1
    base.SWM.value = swm
    toas = make_fake_toas_uniform(_TSTART, _TEND, 50, base, obs="gbt")
    proxy_mjd, proxy_vals = _make_proxy(toas)
    base.components["SolarWindProxyRegression"].set_proxy(proxy_mjd, proxy_vals)
    return base, toas, proxy_mjd, proxy_vals


def test_sw_proxy_instantiation():
    """Component can be programmatically added and has the expected parameters."""
    model = get_model(StringIO(_PROXY_PAR))
    model.add_component(SolarWindProxyRegression())
    assert "SolarWindProxyRegression" in model.components
    assert hasattr(model, "NE_SW")
    assert hasattr(model, "SWPRBETA1")
    assert hasattr(model, "SWPRLAG1")


def test_sw_proxy_set_proxy_normalisation():
    """After set_proxy the stored values are zero-mean and unit-variance."""
    model = get_model(StringIO(_PROXY_PAR))
    model.add_component(SolarWindProxyRegression())
    comp = model.components["SolarWindProxyRegression"]

    rng = np.random.default_rng(0)
    mjds = np.linspace(53000, 57000, 500)
    vals = 5.0 + 2.0 * rng.standard_normal(500)
    comp.set_proxy(mjds, vals)

    stored = comp._proxy_data[1]["vals"]
    assert_allclose(np.mean(stored), 0.0, atol=1e-12)
    assert_allclose(np.std(stored), 1.0, atol=1e-12)


def test_sw_proxy_set_proxy_no_normalisation():
    """With normalize=False the raw values are stored unchanged."""
    model = get_model(StringIO(_PROXY_PAR))
    model.add_component(SolarWindProxyRegression())
    comp = model.components["SolarWindProxyRegression"]

    mjds = np.linspace(53000, 57000, 100)
    vals = np.linspace(3.0, 9.0, 100)
    comp.set_proxy(mjds, vals, normalize=False)

    assert_allclose(comp._proxy_data[1]["vals"], vals)


def test_sw_proxy_dm_zero_when_ne_zero():
    """DM is zero when both NE_SW and BETA1 are zero."""
    model, toas, _, _ = _proxy_model(ne_sw=0.0, beta1=0.0)
    dm = model.components["SolarWindProxyRegression"].solar_wind_dm(toas)
    assert np.all(dm.value == 0.0)


def test_sw_proxy_dm_matches_parent_when_beta1_zero():
    """With BETA1=0 the proxy model gives the same DM as the parent."""
    model_parent = get_model(StringIO("\n".join([_PROXY_PAR, f"NE_SW {_NE_SW_TRUE}"])))

    model_proxy, toas, proxy_mjd, proxy_vals = _proxy_model(
        ne_sw=_NE_SW_TRUE, beta1=0.0
    )

    dm_parent = model_parent.components["SolarWindDispersion"].solar_wind_dm(toas)
    dm_proxy = model_proxy.components["SolarWindProxyRegression"].solar_wind_dm(toas)

    assert_allclose(dm_proxy.value, dm_parent.value, rtol=1e-10)


def test_sw_proxy_dm_positive():
    """DM values are positive when NE_SW > 0 with a loaded proxy."""
    model, toas, _, _ = _proxy_model()
    dm = model.components["SolarWindProxyRegression"].solar_wind_dm(toas)
    assert np.all(dm.value > 0)


def test_sw_proxy_dm_additivity():
    """DM = (NE_SW + BETA1 * x_p) * S, checkable against manual computation."""
    model, toas, _, _ = _proxy_model()
    comp = model.components["SolarWindProxyRegression"]

    geom = comp.solar_wind_geometry(toas)
    xp = comp._get_proxy_at_toas(toas, index=1, lag_days=0.0)
    ne_expected = (_NE_SW_TRUE * u.cm**-3 + _BETA1_TRUE * u.cm**-3 * xp)
    dm_expected = (ne_expected * geom).to(u.pc / u.cm**3)

    dm_computed = comp.solar_wind_dm(toas)
    assert_allclose(dm_computed.value, dm_expected.value, rtol=1e-10)


def test_sw_proxy_d_dm_d_beta1_units():
    """d_dm_d_beta1 has units pc cm^-3 / cm^-3 = pc and correct shape."""
    model, toas, _, _ = _proxy_model()
    comp = model.components["SolarWindProxyRegression"]

    d = comp.d_dm_d_swprbeta(toas, "SWPRBETA1")
    assert d.shape == (len(toas),)
    assert d.unit.is_equivalent(u.pc / u.cm**3 / (u.cm**-3))


def test_sw_proxy_d_dm_d_beta1_matches_finite_difference():
    """Analytic d_dm_d_beta1 matches numerical finite difference."""
    model, toas, _, _ = _proxy_model()
    comp = model.components["SolarWindProxyRegression"]

    eps = 1e-4  # cm^-3
    orig = comp.SWPRBETA1.value

    comp.SWPRBETA1.value = orig + eps
    dm_hi = comp.solar_wind_dm(toas).value
    comp.SWPRBETA1.value = orig - eps
    dm_lo = comp.solar_wind_dm(toas).value
    comp.SWPRBETA1.value = orig

    fd = (dm_hi - dm_lo) / (2 * eps)
    analytic = comp.d_dm_d_swprbeta(toas, "SWPRBETA1").value

    assert_allclose(analytic, fd, rtol=1e-5)


def test_sw_proxy_d_dm_d_lag1_shape():
    """d_dm_d_lag1 has the right shape."""
    model, toas, _, _ = _proxy_model()
    # Give the lag a non-trivial value for a meaningful derivative
    model.SWPRLAG1.value = 5.0
    comp = model.components["SolarWindProxyRegression"]
    d = comp.d_dm_d_swprlag(toas, "SWPRLAG1")
    assert d.shape == (len(toas),)


def test_sw_proxy_missing_proxy_raises():
    """Evaluating solar_wind_dm with BETA1 != 0 but no proxy loaded raises ValueError."""
    model = get_model(StringIO(_PROXY_PAR))
    model.add_component(SolarWindProxyRegression())
    model.NE_SW.value = 5.0
    model.SWPRBETA1.value = 2.0
    toas = make_fake_toas_uniform(_TSTART, _TEND, 20, model, obs="gbt")
    with pytest.raises(ValueError, match="not loaded"):
        model.components["SolarWindProxyRegression"].solar_wind_dm(toas)


def test_sw_proxy_validate_warns_missing_proxy():
    """validate() issues a warning when BETA1 is non-zero but no proxy is loaded."""
    model = get_model(StringIO(_PROXY_PAR))
    model.add_component(SolarWindProxyRegression())
    model.SWPRBETA1.value = 2.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.components["SolarWindProxyRegression"].validate()
    messages = [str(w.message) for w in caught]
    assert any("SWPRBETA1" in m and "not loaded" in m for m in messages)


def test_sw_proxy_print_par_includes_proxy_params():
    """print_par output includes SWPRBETA1 and SWPRLAG1 lines."""
    model, _, _, _ = _proxy_model()
    par_text = model.components["SolarWindProxyRegression"].print_par()
    assert "SWPRBETA1" in par_text
    assert "SWPRLAG1" in par_text
    assert "NE_SW" in par_text


def test_sw_proxy_delay_positive():
    """solar_wind_delay is positive (delays are positive for dispersion)."""
    model, toas, _, _ = _proxy_model()
    delay = model.components["SolarWindProxyRegression"].solar_wind_delay(toas)
    assert np.all(delay.to_value(u.s) > 0)


def test_sw_proxy_matches_parent_swm1_swp2():
    """With SWM=1, SWP=2 and BETA1=0 the DM matches SolarWindDispersion SWM=1."""
    model_parent = get_model(
        StringIO("\n".join([_PROXY_PAR, f"NE_SW {_NE_SW_TRUE}\nSWM 1"]))
    )
    model_proxy, toas, _, _ = _proxy_model(ne_sw=_NE_SW_TRUE, beta1=0.0, swm=1)

    dm_parent = model_parent.components["SolarWindDispersion"].solar_wind_dm(toas)
    dm_proxy = model_proxy.components["SolarWindProxyRegression"].solar_wind_dm(toas)

    assert_allclose(dm_proxy.value, dm_parent.value, rtol=1e-10)


def test_sw_proxy_fit_ne_sw_and_beta1():
    """Fit NE_SW and BETA1 against fake TOAs injected with the proxy model."""
    model_inj, toas, _, _ = _proxy_model(ne_sw=_NE_SW_TRUE, beta1=_BETA1_TRUE)

    # Build a fresh proxy model to fit with slightly perturbed starting values
    model_fit, _, proxy_mjd, proxy_vals = _proxy_model(ne_sw=5.0, beta1=0.5)
    model_fit.NE_SW.frozen = False
    model_fit.SWPRBETA1.frozen = False

    # Use the injected residuals as data
    fake_toas = make_fake_toas_uniform(
        _TSTART, _TEND, 50, model_inj, obs="gbt", add_noise=False
    )
    # Load proxy into the fitting model
    model_fit.components["SolarWindProxyRegression"].set_proxy(proxy_mjd, proxy_vals)

    fitter = Fitter.auto(fake_toas, model_fit)
    fitter.fit_toas()

    assert np.isclose(fitter.model.NE_SW.value, _NE_SW_TRUE, atol=1.0)
    assert np.isclose(fitter.model.SWPRBETA1.value, _BETA1_TRUE, atol=1.0)


def test_sw_proxy_multiple_proxy_slots():
    """Two independent proxy slots produce additive contributions."""
    model = get_model(StringIO(_PROXY_PAR))
    from pint.models.parameter import prefixParameter
    import astropy.constants as const
    from pint import DMconst

    model.add_component(SolarWindProxyRegression())
    comp = model.components["SolarWindProxyRegression"]

    toas = make_fake_toas_uniform(_TSTART, _TEND, 30, model, obs="gbt")
    proxy_mjd, proxy_vals = _make_proxy(toas)

    # Slot 1
    comp.set_proxy(proxy_mjd, proxy_vals, index=1)
    model.NE_SW.value = _NE_SW_TRUE
    model.SWPRBETA1.value = _BETA1_TRUE

    # Add a second proxy slot with the same series but opposite sign
    model.add_param_from_top(
        prefixParameter(
            name="SWPRBETA1",
            index=2,
            units="cm^-3",
            value=-_BETA1_TRUE,
            description="slot 2",
            unit_template=lambda n: "cm^-3",
            description_template=lambda n: f"slot {n}",
            type_match="float",
            tcb2tdb_scale_factor=(const.c * DMconst),
        ),
        "SolarWindProxyRegression",
    )
    comp.set_proxy(proxy_mjd, proxy_vals, index=2)

    # The two slopes cancel; result should equal NE_SW-only DM
    dm_combined = comp.solar_wind_dm(toas)
    dm_ne_sw_only = (
        _NE_SW_TRUE * u.cm**-3 * comp.solar_wind_geometry(toas)
    ).to(u.pc / u.cm**3)

    assert_allclose(dm_combined.value, dm_ne_sw_only.value, rtol=1e-10)
