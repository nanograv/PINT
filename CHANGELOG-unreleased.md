# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved altitude calculation to TOAs object, to make it only happen once
### Added
### Fixed
- Fixed bug where "include_bipm" flag was being ignored when loading Fermi TOAs with weights, now defaults to using EPHEM, CLOCK and PLANET_SHAPIRO from the timing model
- When flags are created based off jumps uses strings instead of None
- Make `get_prefix_timeranges` work for SWX.
### Removed
