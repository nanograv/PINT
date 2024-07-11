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
- Reordered plotting axes in `pintk`
- Changed `scipy.integrate.simps` to `scipy.integrate.simpson` to work with scipy 1.14
- Moved the events -> TOAs and photon weights code into the function `load_events_weights` within `event_optimize`.
- Updated the `maxMJD` argument in `event_optimize` to default to the current mjd
### Added
### Fixed
### Removed
- Removed the argument `--usepickle` in `event_optimize` as the `load_events_weights` function checks the events file type to see if the 
file is a pickle file.
- Removed obsolete code, such as manually tracking the progress of the MCMC run within `event_optimize`
