# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Type hints throughout `pint.models.solar_wind_dispersion`, covering `SolarWindDispersion` and `SolarWindDispersionX` components as well as the newly implemented proxy model class and functions in `pint.solar_indices`.
### Added
- Can output model comparison as JSON
- ``CM_FREF``: reference frequency (MHz) of the chromatic measure, default 1 MHz.
- ``CHROMNOISE_FREF``: reference frequency (MHz) of the ``PLChromNoise`` amplitude, default 1400 MHz.
- ``DMNOISE_FREF``: reference frequency (MHz) of the ``PLDMNoise`` amplitude, default 1400 MHz.
- `SolarWindProxyRegression`: a solar wind model in which the electron density at 1 AU is a linear regression on an external solar activity proxy, `n_E(t) = SWPRBETA0 + SWPRBETA1 * P(t - SWPRLAG)`.
- `pint.solar_indices`: downloading, caching and parsing of solar activity time series, with `f107` (daily 10.7 cm flux adjusted to 1 AU, DRAO Penticton), `ssn` (monthly international sunspot number, NOAA SWPC) and `ace_density` (ACE SWEPAM proton density, from a manual export). Downloading, caching, expiry and failure handling with the `pint.observatory.global_clock_corrections.get_file`.
- Documentation section in `docs/examples/solar_wind.py` on the proxy regression model: where the series comes from, fitting the coefficients on simulated multi-frequency TOAs, interpreting the anticorrelation between them.
- `pint.solar_indices.SMOOTHING_KERNELS`: the smoothing applied to a proxy series is a named `astropy.convolution` kernel, one of `BOX`, `TRAPEZOID`, `TRIANGLE`, `HANN` or `GAUSSIAN`, selected from a par file through the new `SWPRSMOOTHKERNEL` parameter and from Python through `SolarIndexSeries.smoothed(width, kernel=...)`, which also takes any `Kernel1D` instance.
- Chromatic delays are now referenced to the new ``CM_FREF`` parameter instead of an implicit 1 MHz: the ``ChromaticCM``, ``ChromaticCMX`` and ``CMWaveX`` delay is ``CM * DMconst * (freq / CM_FREF)**(-TNCHROMIDX)`` (unchanged for the default ``CM_FREF``).
- ``PLChromNoise`` scales its Fourier basis by ``(CHROMNOISE_FREF/f)**TNCHROMIDX`` instead of a hard-coded 1400 MHz reference (unchanged for the default ``CHROMNOISE_FREF``).
- ``PLDMNoise`` scales its Fourier basis by ``(DMNOISE_FREF/f)**2`` instead of a hard-coded 1400 MHz reference (unchanged for the default ``DMNOISE_FREF``).
### Fixed
### Removed
