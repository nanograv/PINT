# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- Can output model comparison as JSON
### Fixed
- Place ``solar_windx`` before the binary in ``DEFAULT_ORDER`` so SolarWindDispersionX delays and derivatives chain-rule through the binary the same way as ``solar_wind``.
- `make_fake_toas` no longer crashes with a `TypeError`/`AttributeError` when `add_correlated_noise=True` is given for a model with no correlated-noise component; it now logs a warning and skips the correlated noise term.
### Removed
