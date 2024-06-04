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
- Removed `include_bipm`, and `bipm_version` from the Observatory class. Now they are passed as arguments to `site.clock_correction()`
- Renamed `include_gps` to `apply_gps2utc` in the Observatory class
- Removed `apply_gps2utc` from `TOAs.clock_corr_info` since it can be different for different observatories. It is not a global setting.
- The following observatories no longer have a default of `include_bipm=False`: magic, lst, virgo, lho, llo, geo600, kagra, hess, hawc
- New algorithm for TCB <-> TDB conversion
### Added
- `bayesian_information_criterion()` function 
- `dmx_setup` function
- `funcParameter`s are no longer listed in the `pintk` interface.
- `pintk` now reads and automatically converts TCB par files and par files with `BINARY T2`.
- Test for `pint.utils.split_swx()`
- Custom type definitions for type hints
- Added `citation.cff`
- `convert_tcb2tdb`, `tcb2tdb_scale_factor`, and `effective_dimensionality` attributes for `floatParameter`s, `MJDParameter`s, `AngleParameter`s, `maskParameter`s, and `prefixParameter`s.
- Added `pint.observatory.find_latest_bipm()` which returns latest BIPM year available
- Documentation: HOWTO about determining tcb<->tdb scaling factors
- Type hints in `pint.toa` and `get_model()` & `get_model_and_toas()` functions
- `pint.models.chromatic_model.Chromatic` as the base class for variable-index chromatic delays.
- `pint.models.chromatic_model.ChromaticCM` for a Taylor series representation of the variable-index chromatic delay.
### Fixed
- `pint.utils.split_swx()` to use updated `SolarWindDispersionX()` parameter naming convention 
- Fix #1759 by changing order of comparison
- Moved the test in `test_pmtransform_units.py` into a function.
- Fixed bug in residual calculation when adding or removing phase wraps
- Fix #1766 by correcting logic and more clearly naming argument (clkcorr->undo_clkcorr)
### Removed
