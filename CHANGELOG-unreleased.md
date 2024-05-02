# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Avoided unnecessary creation of `SkyCoord` objects in `AstrometryEquatorial` and `AstrometryEcliptic`.
- Avoided unnecessary `TOAs` table slices in `SolarSystemShapiro`
- Allow "CLK UNCORR" in par files (indicates no GPS or BIPM corrections). 
- Better documentation for `akaike_information_criterion()`
- `funcParameter`s are no longer listed in the `pintk` interface.
### Added
- `bayesian_information_criterion()` function 
- `pintk` now reads and automatically converts TCB par files and par files with `BINARY T2`.
### Fixed
### Removed
