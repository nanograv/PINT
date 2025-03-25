# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- In `Residuals`, store correlated noise amplitudes instead of noise residuals. `Residuals.noise_resids` is now a `@property`.
- Reorder `TimingModel.scaled_toa_uncertainty()` and `TimingModel.scaled_dm_uncertainty()` to improve performance.
- Refactor `pint.fitter` to reduce code duplication
### Added
- Simulate correlated DM noise for wideband TOAs
- Type hints in `pint.models.timing_model`
- `full_designmatrix()` and `full_basis_weights()` methods in `TimingModel`
- Added checkbox for optional subtraction of mean in `pintk`
### Fixed
- Made `TimingModel.is_binary()` more robust.
- Correct value of (1/year) in `powerlaw()` function
- Fixed `TestPintk`
- Fixed the noise realization indexing in `Fitter`s
### Removed
- Definition of `@cached_property` to support Python<=3.7
- The broken `data.nanograv.org` URL from the list of solar system ephemeris mirrors
- Broken fitter class `CompositeMCMCFitter` (this fitter was added seemingly to deal with combined radio and high-energy datasets, but has since been broken for a while.)