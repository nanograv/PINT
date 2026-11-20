"""Tests for pint.solar_indices.

The parser tests run against small fixtures written here rather than against the
network, so they are fast and deterministic. The one test that does touch the
network is marked and skipped by default.
"""

import os
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pint import solar_indices as si
from pint.utils import compute_hash


DRAO_TEXT = """fluxdate    fluxtime    fluxjulian    fluxcarrington  fluxobsflux  fluxadjflux  fluxursi
----------  ----------  ------------  --------------  -----------  -----------  ----------
20041028    170000      02453307.229  002022.605      000132.7     000130.9     000117.8
20041028    200000      02453307.354  002022.610      000135.8     000134.0     000120.6
20041028    230000      02453307.479  002022.615      000134.0     000132.2     000119.0
20041029    170000      02453308.229  002022.642      000140.0     000138.0     000124.0
20041029    200000      02453308.354  002022.647      000142.0     000140.0     000126.0
"""

NOAA_JSON = """[
 {"time-tag": "1749-01", "ssn": 96.7,  "smoothed_ssn": -1.0, "f10.7": -1.0},
 {"time-tag": "1810-06", "ssn": 0.0,   "smoothed_ssn": 0.1,  "f10.7": -1.0},
 {"time-tag": "2004-10", "ssn": 48.5,  "smoothed_ssn": 45.0, "f10.7": 105.6},
 {"time-tag": "2004-11", "ssn": 43.6,  "smoothed_ssn": 44.1, "f10.7": 112.4}
]"""

ACE_TEXT = """fp_year fp_doy ACEepoch proton_density proton_temp proton_speed
1998.06027397  23.000000  65059200.000  -9999.900 -9.9999e+03  -9999.90
1998.06301370  24.000000  65145600.000      7.350  8.0000e+04    410.00
1998.06575342  25.000000  65232000.000      5.120  7.5000e+04    395.00
"""


@pytest.fixture
def drao(tmp_path):
    p = tmp_path / "fluxtable.txt"
    p.write_text(DRAO_TEXT)
    return p


@pytest.fixture
def noaa(tmp_path):
    p = tmp_path / "observed-solar-cycle-indices.json"
    p.write_text(NOAA_JSON)
    return p


@pytest.fixture
def ace(tmp_path):
    p = tmp_path / "ACE_SWEPAM_Data.txt"
    p.write_text(ACE_TEXT)
    return p


def test_mjd_conventions():
    """MJD 0 is 1858-11-17 and the ACE epoch is 1996-01-01."""
    assert si.mjd_from_ymd("18581117") == 0
    assert si.mjd_from_ymd("20041028") == 53306
    assert si.ACE_EPOCH_MJD == 50083


def test_monthly_epoch_is_the_middle_of_the_month():
    """A monthly mean sits at the mid-point of its month, not on day 15.

    It is the mean of the daily values over the calendar month, so its epoch is
    mjd0 + (N - 1) / 2 for an N-day month, with daily epochs at 0h.
    """
    # October has 31 days, so the mid-point is the 16th at 0h
    assert si.mjd_from_year_month("2004-10") == si.mjd_from_ymd("20041016")
    # April has 30, giving the 15th at 12h
    assert si.mjd_from_year_month("2004-04") == si.mjd_from_ymd("20040415") + 0.5
    # February: 29 days in a leap year, 28 otherwise
    assert si.mjd_from_year_month("2004-02") == si.mjd_from_ymd("20040215")
    assert si.mjd_from_year_month("2003-02") == si.mjd_from_ymd("20030214") + 0.5
    assert si.mjd_from_year_month("2004-10") - si.mjd_from_ymd("20041015") == 1.0


def test_monthly_cadence_survives_the_exact_epochs():
    """Exact mid-month epochs are unevenly spaced; the median cadence must hold.

    Consecutive mid-points are half the sum of the two month lengths apart, so the
    spacing is not constant.
    """
    mjd = np.array(
        [
            si.mjd_from_year_month(f"{year}-{month:02d}")
            for year in range(2000, 2005)
            for month in range(1, 13)
        ]
    )
    spacing = np.diff(mjd)
    assert 30.0 <= np.median(spacing) <= 31.0
    assert spacing.min() >= 29.5 and spacing.max() <= 31.0


def test_drao_averages_to_one_value_per_day(drao):
    """Three measurements a day collapse to one daily mean."""
    series = si.parse_drao_f107(drao)
    assert len(series) == 2
    assert_allclose(series.mjd, [53306, 53307])
    assert_allclose(series.values.value[0], np.mean([130.9, 134.0, 132.2]))
    assert_allclose(series.values.value[1], np.mean([138.0, 140.0]))
    assert series.unit == si.sfu


def test_drao_defaults_to_the_1au_adjusted_column(drao):
    """The default is fluxadjflux, not fluxobsflux.

    fluxobsflux carries the Sun-Earth distance modulation, an annual term at the
    frequency a solar wind signal occupies.
    """
    adj = si.parse_drao_f107(drao).values.value
    obs = si.parse_drao_f107(drao, column="fluxobsflux").values.value
    assert not np.allclose(adj, obs)
    assert_allclose(adj[0], np.mean([130.9, 134.0, 132.2]))
    assert_allclose(obs[0], np.mean([132.7, 135.8, 134.0]))


def test_drao_rejects_an_unknown_column(drao):
    with pytest.raises(ValueError, match="not among DRAO columns"):
        si.parse_drao_f107(drao, column="nope")


def test_drao_rejects_a_file_without_the_header(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="not a DRAO flux table"):
        si.parse_drao_f107(p)


def test_noaa_ssn_keeps_genuine_zeros(noaa):
    """A zero sunspot number is real data, not a sentinel.

    The filter is per quantity: -1.0 is missing, 0.0 is a measurement.
    """
    series = si.parse_noaa_ssn(noaa)
    assert len(series) == 4  # every row survives, including the 0.0
    assert 0.0 in series.values.value
    assert series.unit == u.dimensionless_unscaled
    assert len(si.parse_noaa(noaa, "smoothed_ssn")) == 3


def test_no_noaa_f107_source():
    """NOAA F10.7 is not offered.

    It is the monthly mean of the same Penticton series, in the observed convention
    rather than adjusted to 1 AU.
    """
    assert "f107_noaa" not in si.SOURCES
    assert not hasattr(si, "parse_noaa_f107")
    assert [k for k in si.SOURCES if "f107" in k] == ["f107"]


def test_noaa_series_are_sorted(noaa):
    assert np.all(np.diff(si.parse_noaa_ssn(noaa).mjd) > 0)


def test_ace_drops_the_missing_sentinel(ace):
    series = si.parse_ace_density(ace)
    assert len(series) == 2
    assert_allclose(series.values.value, [7.35, 5.12])
    assert series.unit == u.cm**-3
    assert_allclose(series.mjd[0], si.ACE_EPOCH_MJD + 65145600.0 / 86400.0)


def test_ace_rejects_a_file_without_density(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("fp_year ACEepoch proton_speed\n1998.0 0.0 400.0\n")
    with pytest.raises(ValueError, match="not an ACE export"):
        si.parse_ace_density(p)


def test_get_series_with_an_explicit_path(drao):
    series = si.get_series("f107", path=drao)
    assert len(series) == 2
    assert series.name == "f107"
    assert series.filename == str(drao)


def test_get_series_honours_the_env_override(drao, monkeypatch):
    """$PINT_SOLAR_INDICES wins over the network, which is what pins an analysis."""
    monkeypatch.setenv(si.pint_solar_env_var, str(drao.parent))
    series = si.get_series("f107", download_policy="never")
    assert_allclose(series.mjd, [53306, 53307])


def test_get_series_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="unknown series"):
        si.get_series("sunshine")


def test_ace_has_no_automatic_download():
    """ACE is manual, and the error has to say so rather than be cryptic.

    The ACE Science Center publishes no plain-text bulk product carrying proton
    density, only HDF4, so there is no URL for this source.
    """
    assert si.SOURCES["ace_density"].url_base is None
    with pytest.raises(FileNotFoundError, match="no automatic download"):
        si.get_series("ace_density", download_policy="never")


def test_series_carries_its_unit_rather_than_a_convention(drao):
    """Units travel with the data rather than being described in a docstring."""
    series = si.parse_drao_f107(drao)
    assert isinstance(series.values, u.Quantity)
    assert series.unit == si.sfu
    # 1 sfu = 1e-22 W m^-2 Hz^-1 = 1e4 Jy
    assert_allclose((1 * si.sfu).to_value(u.Jy), 1e4)


def test_series_validates_on_construction():
    """Length and ordering are checked up front, as ClockFile does."""
    with pytest.raises(ValueError, match="entries but value has"):
        si.SolarIndexSeries(np.arange(5.0), np.arange(4.0))
    with pytest.raises(ValueError, match="out of order"):
        si.SolarIndexSeries([2.0, 1.0], [1.0, 2.0])


def test_series_evaluate_holds_the_endpoints_or_refuses():
    """Out of range is a modelling choice, so the caller gets to pick."""
    series = si.SolarIndexSeries([10.0, 20.0], [1.0, 2.0] * si.sfu, name="test")
    assert_allclose(series.evaluate([15.0]).value, [1.5])
    assert series.evaluate([15.0]).unit == si.sfu

    with pytest.warns(UserWarning, match="does not cover every epoch"):
        held = series.evaluate([5.0, 25.0])
    assert_allclose(held.value, [1.0, 2.0])  # held flat, not extrapolated

    with pytest.raises(ValueError, match="does not cover every epoch"):
        series.evaluate([5.0], limits="error")


def test_series_builds_its_time_lazily():
    """``.time`` is deferred until something asks for it.

    The sunspot record starts in 1749, where a UTC conversion draws an ErfaWarning.
    Interpolation works from the MJD floats, so the common path builds no Time.
    """
    series = si.SolarIndexSeries([55000.0, 55010.0], [1.0, 2.0])
    assert series._time is None
    series.evaluate([55005.0])
    assert series._time is None
    assert series.time.isot[0].startswith("2009-06-18")  # MJD 55000


def test_series_smoothing_is_derived_and_cadence_aware():
    """A width in days means the same window whether sampled daily or monthly."""
    daily = si.SolarIndexSeries(np.arange(0.0, 400.0), np.arange(400.0) * si.sfu)
    monthly = si.SolarIndexSeries(np.arange(0.0, 400.0, 30.0), np.arange(14.0) * si.sfu)
    assert daily.cadence_days == 1.0
    assert monthly.cadence_days == 30.0

    assert daily.smoothed(81).smooth_days == 81.0
    assert daily.smooth_days == 0.0
    assert daily.smoothed(81).unit == si.sfu
    assert monthly.smoothed(7) is monthly
    assert daily.smoothed(0) is daily


def test_series_smoothing_kernel_is_named_and_selectable():
    """The kernel is a named entry in a registry, not a hidden choice."""
    rng = np.random.default_rng(0)
    noisy = 100.0 + rng.normal(0.0, 10.0, 401)
    series = si.SolarIndexSeries(np.arange(401.0), noisy * si.sfu, name="synthetic")

    assert set(si.SMOOTHING_KERNELS) == {
        "box",
        "gaussian",
        "trapezoid",
        "triangle",
        "hann",
    }
    box = series.smoothed(81)
    gaussian = series.smoothed(81, kernel="gaussian")

    assert box.smooth_kernel == "box"
    assert gaussian.smooth_kernel == "gaussian"
    assert_allclose(series.smoothed(81, kernel="box").values.value, box.values.value)
    assert "smoothed 81 d (box)" in repr(box)
    assert "smoothed 81 d (gaussian)" in repr(gaussian)

    assert_allclose(series.smoothed(81, kernel="BOX").values.value, box.values.value)
    assert series.smoothed(81, kernel="GAUSSIAN").smooth_kernel == "gaussian"

    for smoothed in (box, gaussian):
        assert smoothed.values.std() < series.values.std() / 3
        assert_allclose(
            smoothed.values.mean().value, series.values.mean().value, rtol=0.02
        )
    assert not np.allclose(box.values.value, gaussian.values.value)

    with pytest.raises(ValueError, match="Unknown smoothing kernel"):
        series.smoothed(81, kernel="lanczos")


def test_series_smoothing_window_is_centred():
    """An asymmetric window would put the smoothed series half a sample late."""
    ramp = si.SolarIndexSeries(np.arange(801.0), np.arange(801.0) * si.sfu)
    # a straight line survives only a kernel symmetric about the sample it is
    # evaluated at, which Box1DKernel is at even widths too
    for width in (10, 11, 50, 81):
        for kernel in sorted(si.SMOOTHING_KERNELS):
            # support is wider than the FWHM for every kernel but the box
            margin = len(si.SMOOTHING_KERNELS[kernel](width).array) // 2 + 1
            smoothed = ramp.smoothed(width, kernel)
            interior = slice(margin, -margin)
            assert_allclose(
                smoothed.values.value[interior],
                ramp.values.value[interior],
                atol=1e-9,
            )


def test_series_smoothing_is_the_filter_each_name_claims():
    """BOX is an unweighted mean over the window, and the width of GAUSSIAN is a FWHM."""
    rng = np.random.default_rng(7)
    raw = 100.0 + rng.normal(0.0, 10.0, 801)
    series = si.SolarIndexSeries(np.arange(801.0), raw * si.sfu)

    smoothed = series.smoothed(81).values.value
    for i in (100, 400, 700):
        assert_allclose(smoothed[i], raw[i - 40 : i + 41].mean(), atol=1e-9)

    # Gaussian1DKernel takes a standard deviation, so the FWHM is our conversion
    kernel = si.SMOOTHING_KERNELS["gaussian"](80).array
    centre = len(kernel) // 2
    assert_allclose(kernel[centre + 40] / kernel[centre], 0.5, atol=1e-12)
    assert_allclose(kernel[centre - 40] / kernel[centre], 0.5, atol=1e-12)


def test_series_smoothing_width_is_a_fwhm_for_every_kernel():
    """One SWPRSMOOTH means the same window whichever kernel is named."""
    for name, build in sorted(si.SMOOTHING_KERNELS.items()):
        for width in (27, 81, 365):
            kernel = np.asarray(build(width).array, dtype=float)
            assert len(kernel) % 2 == 1, name
            centre = (len(kernel) - 1) / 2
            half = kernel.max() / 2
            right = np.arange(int(np.ceil(centre)), len(kernel))
            edge = np.interp(half, kernel[right][::-1], right[::-1])
            # a discrete flat top reads one sample narrow at the half maximum
            assert abs(2 * (edge - centre) - width) <= 1.0, (name, width)
            assert_allclose(kernel, kernel[::-1], atol=1e-12)


def test_series_smoothing_takes_a_kernel_instance_but_says_it_cannot_be_written():
    """Anything astropy can build is available, at the cost of serialisability."""
    from astropy.convolution import Trapezoid1DKernel

    rng = np.random.default_rng(3)
    raw = 100.0 + rng.normal(0.0, 10.0, 401)
    series = si.SolarIndexSeries(np.arange(401.0), raw * si.sfu, name="synthetic")

    with pytest.warns(UserWarning, match="cannot be written to a par file"):
        tapered = series.smoothed(81, Trapezoid1DKernel(81))

    assert tapered.smooth_kernel == "Trapezoid1DKernel"
    assert tapered.get_metadata()["smooth_kernel"] == "Trapezoid1DKernel"
    assert tapered.values.std() < series.values.std() / 3
    assert not np.allclose(tapered.values.value, series.smoothed(81).values.value)


def test_series_smoothing_below_one_sample_is_a_no_op():
    """astropy rejects a kernel narrower than a sample rather than passing it through."""
    monthly = si.SolarIndexSeries(np.arange(0.0, 400.0, 30.0), np.arange(14.0) * si.sfu)
    assert monthly.cadence_days == 30.0
    for width in (0.0, 1.0, 7.0, 15.0):
        assert monthly.smoothed(width) is monthly
    assert monthly.smoothed(44) is monthly
    assert monthly.smoothed(46).smooth_days == 46.0


def test_series_metadata_records_what_was_read(drao):
    """A fixed URL serves changing content, so a result has to name its input."""
    series = si.get_series("f107", path=drao)
    prov = series.get_metadata()
    assert prov["name"] == "f107"
    assert prov["unit"] == "sfu"
    assert prov["n_samples"] == 2
    assert prov["mjd_start"] == 53306.0
    assert prov["sha256"] == compute_hash(drao).hex()
    assert prov["retrieved"] is not None
    assert (prov["smooth_days"], prov["smooth_kernel"]) == (0.0, "box")
    smoothed = series.smoothed(81, kernel="gaussian").get_metadata()
    assert (smoothed["smooth_days"], smoothed["smooth_kernel"]) == (81.0, "gaussian")


def test_get_series_warns_when_the_unit_disagrees_with_its_source(drao, monkeypatch):
    """A series in unexpected units silently rescales a fitted coefficient.

    ``SWPRBETA1`` is per proxy unit, so a series in different units changes what the
    coefficient means with nothing in the par file to show it.
    """
    monkeypatch.setitem(si.SOURCES, "f107", si.SOURCES["f107"]._replace(unit=u.cm**-3))
    with pytest.warns(UserWarning, match="SOURCES declares"):
        si.get_series("f107", path=drao)


def test_check_coverage_silent_when_contained():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        n = si.check_coverage(np.arange(100.0), np.arange(10.0, 90.0))
    assert n == (0, 0)


def test_check_coverage_reports_both_ends():
    with pytest.warns(UserWarning, match="does not cover every epoch"):
        n_before, n_after = si.check_coverage(
            np.arange(100.0, 200.0), np.array([50.0, 150.0, 250.0, 260.0])
        )
    assert (n_before, n_after) == (1, 2)


def test_check_coverage_reports_the_gap_size():
    with pytest.warns(UserWarning, match="90 d before"):
        si.check_coverage(np.arange(100.0, 200.0), np.array([10.0, 150.0]))


def test_check_coverage_handles_empty_input():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert si.check_coverage([], [1.0, 2.0]) == (0, 0)
        assert si.check_coverage([1.0, 2.0], []) == (0, 0)


@pytest.mark.skipif(
    os.environ.get("PINT_TEST_NETWORK") != "1",
    reason="set PINT_TEST_NETWORK=1 to exercise the real downloads",
)
@pytest.mark.parametrize("name", ["f107", "ssn"])
def test_live_download(name):
    series = si.get_series(name)
    assert len(series) > 100
    assert np.all(np.diff(series.mjd) > 0)
    assert np.all(np.isfinite(series.values.value))
    assert series.unit == si.SOURCES[name].unit
    assert series.sha256 and series.retrieved is not None
    # A second call must be served from the cache with no network.
    cached = si.get_series(name, download_policy="never")
    assert_allclose(series.mjd, cached.mjd)
