# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved altitude calculation to TOAs object, to make it only happen once
- `WidebandDownhillFitter` now handles correlated noise correctly.
- `pintk` Diff/Unc calculation now uses post-fit uncertainties.
- Updated GMRT coordinates.
### Added
- Plot whitened DM residuals in pintk.
- `ssb_to_psb_xyz_ECL` and `ssb_to_psb_xyz_ICRS` are now cached
### Fixed
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
- Fixed bug in printing of parameter correlation/covariance matrices
### Removed
