# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Applied `sourcery` refactors to the entire codebase
### Added
- `SpindownBase` as the abstract base class for `Spindown` and `PeriodSpindown` in the `How_to_build_a_timing_model_component.py` example.
- `SolarWindDispersionBase` as the abstract base class for solar wind dispersion components.
- `validate_component_types` method for more rigorous validation of timing model components.
- roundtrip test to make sure clock corrections are not written to tim files
- `calc_phase_mean` and `calc_time_mean` methods in `Residuals` class to compute the residual mean.
### Fixed
- fixed docstring for `add_param_from_top`
### Removed