# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Command line scripts now automatically do `allow_tcb` and `allow_T2` while reading par files.
- Updated the `plot_chains` function in `event_optimize` so that the subplots are a fixed size to prevent the subplots from being condensed in the case of many fit parameters.
### Added
- Time derivatives of NE_SW in `SolarWindDispersion`
- New prefix pattern for `split_prefixed_name` to handle derivatives of NE_SW
- Added an option `nbin` to `photonphase` to decide how many phase bins to use for the phaseogram
- Added an option `linearize_model` to speed up the photon phases calculation within `event_optimize` through the designmatrix.
- Added AIC and BIC calculation to be written in the post fit parfile from `event_optimize`
- When TCB->TDB conversion info is missing, will print parameter name
- Piecewise-constant model for chromatic variations (CMX)
- `add_param` returns the name of the parameter (useful for numbered parameters)
- `introduces_dm_errors` class attribute in `NoiseComponent`s to distinguish DM noise
- Simulate correlated DM noise for wideband TOAs
### Fixed
- Changed WAVE_OM units from 1/d to rad/d.
- When EQUAD is created from TNEQ, has proper TCB->TDB conversion info
- TOA selection masks will work when only TOA is the first one
### Removed
