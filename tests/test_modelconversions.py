import io
import pytest

import astropy.units as u
import numpy as np

import pint.residuals
import pint.simulation
from pint.fitter import WLSFitter
from pint.models.model_builder import get_model
from pint.pulsar_ecliptic import OBL

modelstring_ECL = """
PSR              B1855+09
LAMBDA   286.8634893301156  1     0.0000000165859
BETA      32.3214877555037  1     0.0000000273526
PMLAMBDA           -3.2701  1              0.0141
PMBETA             -5.0982  1              0.0291
F0    186.4940812707752116  1  0.0000000000328468
F1     -6.205147513395D-16  1  1.379566413719D-19
PEPOCH        54978.000000
POSEPOCH        54978.000000
START            53358.726
FINISH           56598.873
DM               13.299393
"""

modelstring_ICRS = """
PSRJ           1855+09
RAJ             18:57:36.3932884         1  0.00002602730280675029
DECJ           +09:43:17.29196           1  0.00078789485676919773
F0             186.49408156698235146     1  0.00000000000698911818
F1             -6.2049547277487420583e-16 1  1.7380934373573401505e-20
PEPOCH        54978.000000
POSEPOCH        54978.000000
START            53358.726
FINISH           56598.873
DM             13.29709
PMRA           -2.5054345161030380639    1  0.03104958261053317181
PMDEC          -5.4974558631993817232    1  0.06348008663748286318
"""

MJDStart = 57000
MJDStop = 58000
NTOA = 20


def test_ICRS_to_ECL():
    # start with ICRS model, get residuals with ECL model, compare
    model_ICRS = get_model(io.StringIO(modelstring_ICRS))

    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ICRS, error=1 * u.us, add_noise=True
    )
    r_ICRS = pint.residuals.Residuals(toas, model_ICRS)
    r_ECL = pint.residuals.Residuals(toas, model_ICRS.as_ECL())
    assert np.allclose(r_ECL.resids, r_ICRS.resids)
    # assert model_ICRS.as_ECL(ecl).ECL.value == ecl


def test_ECL_to_ICRS():
    # start with ECL model, get residuals with ICRS model, compare
    model_ECL = get_model(io.StringIO(modelstring_ECL))

    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ECL, error=1 * u.us, add_noise=True
    )
    r_ECL = pint.residuals.Residuals(toas, model_ECL)
    r_ICRS = pint.residuals.Residuals(toas, model_ECL.as_ICRS())
    assert np.allclose(r_ECL.resids, r_ICRS.resids)


def test_ICRS_to_ECL_nouncertainties():
    # start with ICRS model with no pm uncertainties, get residuals with ECL model, compare
    model_ICRS = get_model(io.StringIO(modelstring_ICRS))
    model_ICRS.PMRA._uncertainty = None
    model_ICRS.PMRA.frozen = True
    model_ICRS.PMDEC._uncertainty = None
    model_ICRS.PMDEC.frozen = True
    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ICRS, error=1 * u.us, add_noise=True
    )
    r_ICRS = pint.residuals.Residuals(toas, model_ICRS)
    converted = model_ICRS.as_ECL()
    r_ECL = pint.residuals.Residuals(toas, converted)
    assert np.allclose(r_ECL.resids, r_ICRS.resids)
    assert converted.PMELONG.uncertainty is None
    assert converted.PMELAT.uncertainty is None
    # assert model_ICRS.as_ECL(ecl).ECL.value == ecl


def test_ECL_to_ICRS_nouncertainties():
    # start with ECL model with no pm uncertainties, get residuals with ICRS model, compare
    model_ECL = get_model(io.StringIO(modelstring_ECL))
    model_ECL.PMELONG._uncertainty = None
    model_ECL.PMELONG.frozen = True
    model_ECL.PMELAT._uncertainty = None
    model_ECL.PMELAT.frozen = True
    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ECL, error=1 * u.us, add_noise=True
    )
    r_ECL = pint.residuals.Residuals(toas, model_ECL)
    converted = model_ECL.as_ICRS()
    r_ICRS = pint.residuals.Residuals(toas, converted)
    assert np.allclose(r_ECL.resids, r_ICRS.resids)
    assert converted.PMRA.uncertainty is None
    assert converted.PMDEC.uncertainty is None


def test_ECL_to_ECL():
    # start with ECL model, get residuals with ECL model with different obliquity, compare
    model_ECL = get_model(io.StringIO(modelstring_ECL))

    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ECL, error=1 * u.us, add_noise=True
    )
    r_ECL = pint.residuals.Residuals(toas, model_ECL)
    r_ECL2 = pint.residuals.Residuals(toas, model_ECL.as_ECL(ecl="IERS2003"))
    assert np.allclose(r_ECL.resids, r_ECL2.resids)
    assert model_ECL.as_ECL(ecl="IERS2003").ECL.value == "IERS2003"


def test_ECL_to_ICRS_uncertainties():
    # start with ECL model, fit with both models
    # compare parameter values and uncertainties
    model_ECL = get_model(io.StringIO(modelstring_ECL))

    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ECL, error=1 * u.us, add_noise=True
    )
    fit_ECL = WLSFitter(toas, model_ECL)
    fit_ICRS = WLSFitter(toas, model_ECL.as_ICRS())
    fit_ECL.fit_toas()
    fit_ICRS.fit_toas()

    m1 = fit_ECL.model
    m2 = fit_ICRS.model.as_ECL()

    for p in ("ELONG", "ELAT", "PMELONG", "PMELAT"):
        assert np.isclose(m1.__getitem__(p).value, m2.__getitem__(p).value)
        # do a generous test here since the uncertainties could change
        assert np.isclose(
            m1.__getitem__(p).uncertainty, m2.__getitem__(p).uncertainty, rtol=0.5
        )


def test_ICRS_to_ECL_uncertainties():
    # start with ICRS model, fit with both models
    # compare parameter values and uncertainties
    model_ICRS = get_model(io.StringIO(modelstring_ICRS))

    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ICRS, error=1 * u.us, add_noise=True
    )
    fit_ICRS = WLSFitter(toas, model_ICRS)
    fit_ECL = WLSFitter(toas, model_ICRS.as_ECL())
    fit_ECL.fit_toas()
    fit_ICRS.fit_toas()

    m1 = fit_ICRS.model
    m2 = fit_ECL.model.as_ICRS()

    for p in ("RAJ", "DECJ", "PMRA", "PMDEC"):
        tol = 1e-12 if p in ["RAJ", "DECJ"] else 1e-3
        assert np.isclose(
            m1.__getitem__(p).value, m2.__getitem__(p).value, atol=tol
        ), f"Paramter {p} with values {m1.__getitem__(p).value}, {m2.__getitem__(p).value} was too far apart (difference={m1.__getitem__(p).value-m2.__getitem__(p).value})"
        # do a generous test here since the uncertainties could change
        assert np.isclose(
            m1.__getitem__(p).uncertainty, m2.__getitem__(p).uncertainty, rtol=0.5
        )


def test_ECL_to_ECL_uncertainties():
    # start with ECL model, fit with both models
    # compare parameter values and uncertainties
    model_ECL = get_model(io.StringIO(modelstring_ECL))

    toas = pint.simulation.make_fake_toas_uniform(
        MJDStart, MJDStop, NTOA, model=model_ECL, error=1 * u.us, add_noise=True
    )
    fit_ECL = WLSFitter(toas, model_ECL)
    fit_ECL2 = WLSFitter(toas, model_ECL.as_ECL(ecl="IERS2003"))
    fit_ECL.fit_toas()
    fit_ECL2.fit_toas()

    m1 = fit_ECL.model
    m2 = fit_ECL2.model.as_ECL(ecl="IERS2010")

    for p in ("ELONG", "ELAT", "PMELONG", "PMELAT"):
        assert np.isclose(m1.__getitem__(p).value, m2.__getitem__(p).value)
        # do a generous test here since the uncertainties could change
        assert np.isclose(
            m1.__getitem__(p).uncertainty, m2.__getitem__(p).uncertainty, rtol=0.5
        )


@pytest.mark.parametrize("ecl", OBL.keys())
def test_ECL_to_allECL(ecl):
    model_ECL = get_model(io.StringIO(modelstring_ECL))
    model_ECL2 = model_ECL.as_ECL(ecl=ecl)
    coords_ECL2 = model_ECL2.get_psr_coords()
    assert model_ECL2.ECL.value == ecl
    # note that coord.separation() will transform between frames when needed
    assert np.isclose(model_ECL.get_psr_coords().separation(coords_ECL2).arcsec, 0)


# High-|ELAT| ecliptic astrometry: the old signed-vector uncertainty transform
# returned a negative ELONG σ on the ICRS→ECL leg (PPTA J0711-6830 / J1017-7156).
modelstring_ECL_high_lat = """
PSR J0711-6830
ELONG  204.06115135931555  1  5.698305032154e-08
ELAT   -82.88863150271472  1  5.66560917129e-09
PMELONG -12.03829529527527 1  0.003414889229688632
PMELAT  -17.28185806109565 1  0.003068280695947412
F0 182.11723462004737 1
PEPOCH 55636
POSEPOCH 55636
ECL IERS2010
"""


def test_ecliptic_obliquity_uncertainty_roundtrip():
    """ECL→ECL obliquity change must preserve diagonal uncertainties.

    The IERS2003/IERS2010 obliquities differ by only 1e-4 arcsec, so the
    tangent-frame rotation is tiny, so the induced correlation discarded by
    the parameter-only representation is negligible and the input marginal
    uncertainties return to numerical precision. This is not an assertion
    that a general frame round trip preserves a full covariance matrix.
    """
    model = get_model(io.StringIO(modelstring_ECL_high_lat))
    roundtrip = model.as_ECL(ecl="IERS2003").as_ECL(ecl="IERS2010")
    for name in ("ELONG", "ELAT", "PMELONG", "PMELAT"):
        assert np.isclose(
            getattr(model, name).uncertainty.to_value(
                getattr(roundtrip, name).uncertainty.unit
            ),
            getattr(roundtrip, name).uncertainty.value,
            rtol=1e-12,
            atol=0,
        ), name


def test_icrs_to_ecliptic_uncertainties_match_covariance_rotation():
    """Check one-way marginals against an analytic tangent-plane rotation."""
    model = get_model(
        io.StringIO(
            """
PSR TEST
RAJ 00:00:00
DECJ +00:00:00
PMRA 1
PMDEC 1
F0 100
PEPOCH 55000
POSEPOCH 55000
"""
        )
    )
    model.RAJ.uncertainty = 1 * u.mas
    model.DECJ.uncertainty = 100 * u.mas
    model.PMRA.uncertainty = 2 * u.mas / u.yr
    model.PMDEC.uncertainty = 50 * u.mas / u.yr

    converted = model.as_ECL(ecl="IERS2010")
    obliquity = OBL["IERS2010"]

    def expected(sigma_lon, sigma_lat):
        return (
            np.hypot(
                np.cos(obliquity) * sigma_lon,
                np.sin(obliquity) * sigma_lat,
            ),
            np.hypot(
                np.sin(obliquity) * sigma_lon,
                np.cos(obliquity) * sigma_lat,
            ),
        )

    elong, elat = expected(1 * u.mas, 100 * u.mas)
    pmelong, pmelat = expected(2 * u.mas / u.yr, 50 * u.mas / u.yr)
    assert u.isclose(converted.ELONG.uncertainty, elong)
    assert u.isclose(converted.ELAT.uncertainty, elat)
    assert u.isclose(converted.PMELONG.uncertainty, pmelong)
    assert u.isclose(converted.PMELAT.uncertainty, pmelat)


def test_as_ecl_uncertainties_use_requested_epoch_frame():
    """Rotate uncertainties in the tangent frame used for converted values."""
    par = """
PSR TEST
RAJ 00:00:00 1 0.001
DECJ +70:00:00 1 0.01
PMRA 100000 1 10
PMDEC -50000 1 20
F0 100
PEPOCH 55000
POSEPOCH 55000
"""
    model = get_model(io.StringIO(par))
    target_epoch = 65000
    propagated = model.coords_as_ICRS(epoch=target_epoch)

    # Construct the equivalent source model directly at the requested epoch.
    at_epoch = get_model(io.StringIO(par))
    at_epoch.RAJ.quantity = propagated.ra
    at_epoch.DECJ.quantity = propagated.dec
    at_epoch.PMRA.quantity = propagated.pm_ra_cosdec
    at_epoch.PMDEC.quantity = propagated.pm_dec
    at_epoch.POSEPOCH.value = target_epoch

    converted = model.as_ECL(epoch=target_epoch)
    expected = at_epoch.as_ECL()
    for name in ("ELONG", "ELAT", "PMELONG", "PMELAT"):
        assert u.allclose(
            getattr(converted, name).uncertainty,
            getattr(expected, name).uncertainty,
        ), name


def test_as_ecl_uncertainties_are_nonnegative():
    """ICRS→ECL must not assign a signed/negative uncertainty.

    High-|ELAT| sources (e.g. PPTA J0711-6830) previously failed here with
    ``Uncertainties cannot be negative`` because ``as_ECL`` transformed
    ``(σ_lon, σ_lat)`` as a signed vector while ``as_ICRS`` wrapped the
    reverse transform in ``np.abs``.
    """
    model = get_model(io.StringIO(modelstring_ECL_high_lat))
    ecl = model.as_ICRS().as_ECL(ecl="IERS2010")
    for name in ("ELONG", "ELAT", "PMELONG", "PMELAT"):
        unc = getattr(ecl, name).uncertainty
        assert unc is not None
        assert unc.value >= 0, name
    # The old signed-vector path produced |σ_ELONG| ≈ 6.68e-8 deg here;
    # covariance rotation must stay near the input 5.70e-8 deg.
    assert np.isclose(
        model.ELONG.uncertainty.to_value(u.deg),
        ecl.ELONG.uncertainty.to_value(u.deg),
        rtol=0.02,
    )
