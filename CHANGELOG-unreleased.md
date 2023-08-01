# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Made the addition of a TZR TOA (`AbsPhase`) in the `TimingModel` explicit in `Residuals` class.
- Made the parameter order in `TimingModel.designmatrix` consistent with `TimingModel.free_params`
### Added
- Third-order Roemer delay terms to ELL1 model
- Options to add a TZR TOA (`AbsPhase`) during the creation of a `TimingModel` using `ModelBuilder.__call__`, `get_model`, and `get_model_and_toas`
### Fixed
- Deleting JUMP1 from flag tables will not prevent fitting
- Docstrings for `get_toas()` and `get_model_and_toas()`
- Set `DelayComponent_list` and `NoiseComponent_list` to empty list if such components are absent
- Fix invalid access of `PLANET_SHAPIRO` in models without `Astrometry`
### Removed
