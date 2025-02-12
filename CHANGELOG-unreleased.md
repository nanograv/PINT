# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- In `Residuals`, store correlated noise amplitudes instead of noise residuals. `Residuals.noise_resids` is now a `@property`.
- Refactor `pint.fitter` to reduce code duplication
### Added
- Simulate correlated DM noise for wideband TOAs
- Type hints in `pint.models.timing_model`
- `full_designmatrix()` and `full_basis_weights()` methods in `TimingModel`
### Fixed
- Made `TimingModel.is_binary()` more robust.
- Fixed `TestPintk`
- Fixed the noise realization indexing in `Fitter`s
### Removed
- Definition of `@cached_property` to support Python<=3.7
