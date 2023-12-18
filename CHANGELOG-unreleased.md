# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved `get_derived_params` to `timing_model`
### Added
- Added numdifftools to setup.cfg to match requirements.txt
- Documentation: Added `convert_parfile` to list of command-line tools in RTD
- Plot model DM in pintk
- More tests for pintk
- `pint.models.chromatic_model.Chromatic` as the base class for variable-index chromatic delays.
- `pint.models.chromatic_model.ChromaticCM` for a Taylor series representation of the variable-index chromatic delay.
### Fixed
- `MCMC_walkthrough` notebook now runs
- Fixed runtime data README 
- Fixed `derived_params` when OMDOT has 0 uncertainty
- Fixed `make_fake_toas_fromtim`
### Removed
