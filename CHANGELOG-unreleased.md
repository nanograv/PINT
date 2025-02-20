# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- In `Residuals`, store correlated noise amplitudes instead of noise residuals. `Residuals.noise_resids` is now a `@property`.
### Added
- Simulate correlated DM noise for wideband TOAs
- Type hints in `pint.models.timing_model`
### Fixed
- Made `TimingModel.is_binary()` more robust
- Added missing newline in `tempo_polyco_table_writer()`
- Proper warning in `read_polyco_file()`
### Removed
- Definition of `@cached_property` to support Python<=3.7
- The broken `data.nanograv.org` URL from the list of solar system ephemeris mirrors
