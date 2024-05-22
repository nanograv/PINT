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
- Type hinting for most of the `pint.utils` module
- `funcParameter`s are no longer listed in the `pintk` interface.
- Updated location of CCERA
### Added
- `bayesian_information_criterion()` function 
- `dmx_setup` function
- `funcParameter`s are no longer listed in the `pintk` interface.
- `pintk` now reads and automatically converts TCB par files and par files with `BINARY T2`.
- Test for `pint.utils.split_swx()`
- Custom type definitions for type hints
- Added `citation.cff`
- `pint.models.chromatic_model.Chromatic` as the base class for variable-index chromatic delays.
- `pint.models.chromatic_model.ChromaticCM` for a Taylor series representation of the variable-index chromatic delay.
### Fixed
- `pint.utils.split_swx()` to use updated `SolarWindDispersionX()` parameter naming convention 
- Fix #1759 by changing order of comparison
- Fixed bug in residual calculation when adding or removing phase wraps
### Removed
