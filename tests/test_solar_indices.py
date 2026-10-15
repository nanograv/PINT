"""Tests for pint.solar_indices.

The parser tests run against small fixtures written here rather than against the
network, so they are fast and deterministic. The one test that does touch the
network is marked and skipped by default.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pint import solar_indices as si


# ---------------------------------------------------------------------------
# fixtures: miniature versions of each upstream format
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# epoch conventions
# ---------------------------------------------------------------------------


def test_mjd_conventions():
    """MJD 0 is 1858-11-17 and the ACE epoch is 1996-01-01."""
    assert si.mjd_from_ymd("18581117") == 0
    assert si.mjd_from_ymd("20041028") == 53306
    assert si.ACE_EPOCH_MJD == 50083
    assert si.mjd_from_year_month("2004-10") == si.mjd_from_ymd("20041015")


# ---------------------------------------------------------------------------
# DRAO
# ---------------------------------------------------------------------------


def test_drao_averages_to_one_value_per_day(drao):
    """Three measurements a day collapse to one daily mean."""
    mjd, flux = si.parse_drao_f107(drao)
    assert len(mjd) == 2
    assert_allclose(mjd, [53306, 53307])
    assert_allclose(flux[0], np.mean([130.9, 134.0, 132.2]))
    assert_allclose(flux[1], np.mean([138.0, 140.0]))


def test_drao_defaults_to_the_1au_adjusted_column(drao):
    """The default is fluxadjflux, not fluxobsflux.

    fluxobsflux carries the Sun-Earth distance modulation, a 6.8% peak-to-peak
    annual term measured on the real table. That lands on exactly the frequency a
    solar wind signal occupies, so picking it up by accident would be invisible in
    the fit and wrong. Asserting the two columns give different numbers is what
    stops a silent swap.
    """
    _, adj = si.parse_drao_f107(drao)
    _, obs = si.parse_drao_f107(drao, column="fluxobsflux")
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


# ---------------------------------------------------------------------------
# NOAA
# ---------------------------------------------------------------------------


def test_noaa_ssn_keeps_genuine_zeros(noaa):
    """A zero sunspot number is real data, not a sentinel.

    The sunspot number is legitimately 0.0 at deep solar minimum, and the real NOAA
    file carries 67 such rows. Filtering on positivity, which would be correct for a
    flux, silently discards them. The filter has to be per quantity: -1.0 is
    missing, 0.0 is a measurement.
    """
    mjd, ssn = si.parse_noaa_ssn(noaa)
    assert len(ssn) == 4  # every row survives, including the 0.0
    assert 0.0 in ssn
    # the -1.0 sentinel on a different key in the same rows is what gets dropped
    assert len(si.parse_noaa(noaa, "smoothed_ssn")[1]) == 3


def test_no_noaa_f107_source():
    """NOAA F10.7 is deliberately not offered.

    It is not an independent measurement: it is the monthly mean of the same
    Penticton series, in the observed convention rather than adjusted to 1 AU, which
    is the convention parse_drao_f107 exists to avoid.
    """
    assert "f107_noaa" not in si.SOURCES
    assert not hasattr(si, "parse_noaa_f107")
    assert [k for k in si.SOURCES if "f107" in k] == ["f107"]


def test_noaa_series_are_sorted(noaa):
    mjd, _ = si.parse_noaa_ssn(noaa)
    assert np.all(np.diff(mjd) > 0)


# ---------------------------------------------------------------------------
# ACE
# ---------------------------------------------------------------------------


def test_ace_drops_the_missing_sentinel(ace):
    mjd, density = si.parse_ace_density(ace)
    assert len(density) == 2
    assert_allclose(density, [7.35, 5.12])
    assert_allclose(mjd[0], si.ACE_EPOCH_MJD + 65145600.0 / 86400.0)


def test_ace_rejects_a_file_without_density(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("fp_year ACEepoch proton_speed\n1998.0 0.0 400.0\n")
    with pytest.raises(ValueError, match="not an ACE export"):
        si.parse_ace_density(p)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_get_series_with_an_explicit_path(drao):
    mjd, flux = si.get_series("f107", path=drao)
    assert len(mjd) == 2


def test_get_series_honours_the_env_override(drao, monkeypatch):
    """$PINT_SOLAR_INDICES wins over the network, which is what pins an analysis."""
    monkeypatch.setenv(si.pint_solar_env_var, str(drao.parent))
    mjd, flux = si.get_series("f107", download_policy="never")
    assert_allclose(mjd, [53306, 53307])


def test_get_series_rejects_an_unknown_name():
    with pytest.raises(KeyError, match="unknown series"):
        si.get_series("sunshine")


def test_ace_has_no_automatic_download():
    """ACE is manual by design; the error has to say so rather than be cryptic.

    The ACE Science Center publishes no plain-text bulk product carrying proton
    density, only HDF4, so there is deliberately no URL for this source.
    """
    assert si.SOURCES["ace_density"].url_base is None
    with pytest.raises(FileNotFoundError, match="no automatic download"):
        si.get_series("ace_density", download_policy="never")


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PINT_TEST_NETWORK") != "1",
    reason="set PINT_TEST_NETWORK=1 to exercise the real downloads",
)
@pytest.mark.parametrize("name", ["f107", "ssn"])
def test_live_download(name):
    mjd, value = si.get_series(name)
    assert len(mjd) > 100
    assert np.all(np.diff(mjd) > 0)
    assert np.all(np.isfinite(value))
    # A second call must be served from the cache with no network.
    mjd2, value2 = si.get_series(name, download_policy="never")
    assert_allclose(mjd, mjd2)
