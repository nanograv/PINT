# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- WAVE parameters can be added to a Wave model with `add_wave_component()` in wave.py 
### Added
- Added WaveX model as DelayComponent with wave amplitudes as fitted parameters
### Fixed
- Wave model `validate()` can correctly use PEPOCH to assign WAVEEPOCH parameter
- Fixed RTD by specifying theme explicitly.
- `.value()` now works for pairParameters
- Setting `model.PARAM1 = model.PARAM2` no longer overrides the name of `PARAM1`
### Removed
