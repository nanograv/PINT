# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- `SolarWindProxyRegression`: a solar wind model in which the electron density at 1 AU is a linear regression on an external solar activity proxy, `n_E(t) = SWPRBETA0 + SWPRBETA1 * P(t - SWPRLAG)`, with a boxcar width `SWPRSMOOTH` (default 81 days, the conventional F10.7A window) and the series named by `SWPRSOURCE`. The proxy enters in its own units, so the coefficients mean the same thing for every pulsar whatever its span. The series is resolved lazily on first evaluation, so a bare `get_model(par)` builds a model that evaluates; `set_proxy()` loads a series from arrays instead and warns that such a model cannot be written to a par file or rebuilt from one. Shares `category = "solar_wind"` with `SolarWindDispersion`, so the two are mutually exclusive alternatives, `SWPRBETA0` and `NE_SW` being exactly degenerate.
- `SolarWindProxyRegression.get_mean_ne()`, returning the derived mean electron density over the TOAs. This is the quantity to quote and the one comparable to a constant-density fit, since `SWPRBETA0` and `SWPRBETA1` are strongly anticorrelated.
- `pint.solar_indices`: downloading, caching and parsing of solar activity time series, with `f107` (daily 10.7 cm flux adjusted to 1 AU, DRAO Penticton), `ssn` (monthly international sunspot number, NOAA SWPC) and `ace_density` (ACE SWEPAM proton density, from a manual export). Downloading, caching, expiry and failure handling are delegated to `pint.observatory.global_clock_corrections.get_file`, so `download_policy="never"` is offline and deterministic and a network error falls back to the cached copy. Series resolve in three tiers, as clock files do: an explicit `path=` or `$PINT_SOLAR_INDICES`, then the cached download, then a bundled fallback.
- `pint.solar_indices.check_coverage`, which warns with the epoch count and the gap in days when epochs fall outside a series rather than silently holding its endpoint value. Called by `SolarWindProxyRegression.get_proxy_at_toas`.
- Documentation section in `docs/examples/solar_wind.py` on the proxy regression model: where the series comes from, fitting the coefficients on simulated multi-frequency TOAs, interpreting the anticorrelation between them, and the par file round trip.
### Fixed
### Removed
