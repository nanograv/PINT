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