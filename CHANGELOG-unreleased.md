# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- Third-order Roemer delay terms to ELL1 model
- Added WaveX model as DelayComponent with wave amplitudes as fitted parameters
- Options to add a TZR TOA (`AbsPhase`) during the creation of a `TimingModel` using `ModelBuilder.__call__`, `get_model`, and `get_model_and_toas`
- `pint.print_info()` function for bug reporting
### Fixed
- Fixed RTD by specifying theme explicitly.
### Removed
