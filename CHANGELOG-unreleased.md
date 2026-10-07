# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- `d_delay_d_param` now applies the chain rule for delay components that respond to delays accumulated from earlier components (e.g. a binary delay's dependence on its evaluation epoch). Design-matrix entries for binary pulsars change at the ~1e-5 relative level. The parameter-independent ingredients are computed once per design matrix via the new `TimingModel.delay_deriv_chain`; the unused `acc_delay` argument of `d_delay_d_param` was replaced by the optional `chain` argument.
### Added
- Support for hierarchical triple systems: a second (outer) binary component can be added via a `BINARY2` line with `_2`-suffixed orbital parameters (e.g. `PB_2`, `A1_2`). Outer orbit delay is computed before, and propagated into, the inner binary. Outer wrappers `BinaryDD2`, `BinaryBT2`, and `BinaryELL12` are provided. Delay derivatives account for the outer→inner coupling (chain rule through the previous delay). The projected semi-major axis includes a second time derivative `A1DOT2` (alias `X2DOT`).
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
- `TOAs.get_tdb_seconds()`, returning the TDB times of the TOAs in seconds with a selectable dtype
### Fixed
- Propagate one-way astrometric marginal uncertainties in ``as_ECL`` / ``as_ICRS`` by diagonal covariance rotation instead of a signed "fake proper motion" vector. The old ``as_ECL`` path could assign a negative ELONG/ELAT uncertainty (breaking model construction) after an ecliptic↔ICRS round trip, and both directions returned the wrong marginal σ after a non-trivial frame rotation. Correlations induced by conversion are not retained because timing-model parameters store only marginal uncertainties.
- Remove spurious ``/ Tsun`` factor from analytic DDH ``∂delay/∂STIGMA`` (design matrix / GLS for free ``STIGMA`` was wrong by ``1/Tsun`` since the Maple rewrite in PINT ≥ 1.0).
- Align ``d_delayS3p_H3_STIGMA_exact_d_STIGMA`` with Eq. (28): ``cos(2*Phi)``.
- Prefer DD over BT when guessing the binary model for Tempo2 `T2` par files (`allow_T2`), matching Tempo2's `allTerms=1` behavior
- `WidebandTOAFitter` raises a warning if the model has correlated errors (It used to give wrong results before).
- Fixed bug where "include_bipm" flag was being ignored when loading Fermi TOAs with weights, now defaults to using EPHEM, CLOCK and PLANET_SHAPIRO from the timing model
- When flags are created based off jumps uses strings instead of None
- When writing tempo format parfiles, use 0 instead of inf for TZRFRQ
- Write VLBI frame rotation parameters correctly to par file. 
- Make `get_prefix_timeranges` work for SWX.
- Some of the `gridutils` functions had improper logging behavior
- Fixed bug in changing epoch for ELL1k model
- Fixed `gridutils` behavior for 1 CPU
- Fixed bug in `GaussianRV_gen`, where the probability distribution function was not normalized correctly. Changed to use `scipy.stats.truncnorm` instead of the custom `GaussianRV_gen`.
- Fixed `convert_binary()` for ELL1H models to run `setup()` and not use H4 when not desired
- Fixed bug in `model.compare()` where it failed for `PosixPath` objects
- Fixed bug in printing of parameter correlation/covariance matrices
- `make_fake_toas_fromMJDs` now does not assume `PLANET_SHAPIRO` is in the model - it checks.
- Make VLBI frame rotation work correctly when proper motion is present.
- Changed some API to pass Mac CI
- Log-separated frequency computation for red noise components.
- Place ``solar_windx`` before the binary in ``DEFAULT_ORDER`` so SolarWindDispersionX delays and derivatives chain-rule through the binary the same way as ``solar_wind``.
### Removed
