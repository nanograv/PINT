# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Applied `sourcery` refactors to the entire codebase
- Changed threshold for `test_model_derivatives` test to avoid CI failures
- Unreleased CHANGELOG entries should now be entered in `CHANGELOG-unreleased.md` instead of `CHANGELOG.md`. Updated documentation accordingly.
### Added
- `SpindownBase` as the abstract base class for `Spindown` and `PeriodSpindown` in the `How_to_build_a_timing_model_component.py` example.
- `SolarWindDispersionBase` as the abstract base class for solar wind dispersion components.
- `validate_component_types` method for more rigorous validation of timing model components.
- roundtrip test to make sure clock corrections are not written to tim files
- `calc_phase_mean` and `calc_time_mean` methods in `Residuals` class to compute the residual mean.
- - `PhaseOffset` component (overall phase offset between physical and TZR toas)
- `tzr` attribute in `TOAs` class to identify TZR TOAs
- Documentation: Explanation for offsets
- Example: `phase_offset_example.py`
- method `AllComponents.param_to_unit` to get units for any parameter, and then made function `utils.get_unit`
- can override/add parameter values when reading models
- docs now include list of observatories along with google maps links and clock files
### Fixed
- fixed docstring for `add_param_from_top`
- Gridded calculations now respect logger settings
- Event TOAs now have default error that is non-zero, and can set as desired
- Deleting JUMP1 from flag tables will not prevent fitting
### Removed