# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- Simulate correlated DM noise for wideband TOAs
- Type hints in `pint.models.timing_model`
### Fixed
- Made `TimingModel.is_binary()` more robust. 
### Removed
- Unnecessary definition of `cached_property` from `pint.fitter` (Python 3.8 no longer needs to be supported).
