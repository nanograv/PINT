# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- Simulate correlated DM noise for wideband TOAs
- Properly incorporate correlated DM noise in fitting
    - Replaced the removed `introduces_correlated_errors` attribute with `WhiteNoiseComponent` and `CorrelatedNoiseComponent` abstract base classes
    - `get_wideband_errors` method in `TOAs`
    - `scaled_wideband_uncertainty`, `noise_model_wideband_designmatrix`, `dm_designmatrix`, `wideband_designmatrix`, `full_designmatrix`, and `full_basis_weight` methods in `TimingModel`
    - `calc_combined_resids` method in `WidebandTOAResiduals`
### Fixed
### Removed
