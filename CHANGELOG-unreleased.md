# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Third-order Roemer delay terms to ELL1 model
- Made the addition of a TZR TOA (`AbsPhase`) in the `TimingModel` explicit in `Residuals` class.
- Updated `CONTRIBUTING.rst` with the latest information.
- Made `TimingModel.params` and `TimingModel.ordered_params` identical. Deprecated `TimingModel.ordered_params`.
### Added
- Third-order Roemer delay terms to ELL1 model
- Options to add a TZR TOA (`AbsPhase`) during the creation of a `TimingModel` using `ModelBuilder.__call__`, `get_model`, and `get_model_and_toas`
- `pint.print_info()` function for bug reporting
- Added an autocorrelation function to check for chain convergence in `event_optimize`
- Minor doc updates to explain default NHARMS and missing derivative functions
### Fixed
- Deleting JUMP1 from flag tables will not prevent fitting
- Simulating TOAs from tim file when PLANET_SHAPIRO is true now works
- Docstrings for `get_toas()` and `get_model_and_toas()`
- Set `DelayComponent_list` and `NoiseComponent_list` to empty list if such components are absent
- Fix invalid access of `PLANET_SHAPIRO` in models without `Astrometry`
- Fix RTD build by making the theme explicit even when on RTD system
### Removed
