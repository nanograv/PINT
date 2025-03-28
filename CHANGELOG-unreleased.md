# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- In `Residuals`, store correlated noise amplitudes instead of noise residuals. `Residuals.noise_resids` is now a `@property`.
- Refactor `pint.fitter` to reduce code duplication
- Changed the "number of harmonics" parameters in red noise components into `intParameter`s
### Added
- Simulate correlated DM noise for wideband TOAs
    - `NoiseComponent.introduces_dm_errors` attribute
- Type hints in `pint.models.timing_model`
- `full_designmatrix()` and `full_basis_weights()` methods in `TimingModel`
- Proper fitting of wideband TOAs in the presence of correlated DM noise
    - Abstract base classes for noise components -- `WhiteNoiseComponent` and `CorrelatedNoiseComponent`
    - Changed `NoiseComponent.introduces_correlated_errors` into a `@property`
    - `get_dm_noise_basis()` and `get_wideband_noise_basis()` methods in `CorrelatedNoiseComponent` 
    - New methods in `TimingModel` -- `wideband_covariance_matrix()`, `scaled_wideband_uncertainty()`, `noise_model_wideband_designmatrix()`, `full_designmatrix()`, `full_wideband_designmatrix()`, `full_basis_weight()`, `dm_designmatrix()`, `wideband_designmatrix()`
    - New method in `WidebandTOAResiduals` -- `calc_wideband_resids()`
- Added checkbox for optional subtraction of mean in `pintk`
- Added Log-Linear Powerlaw noise parameters to `PLRedNoise`, `PLDMNoise`, `PLChromNoise`
- `TimingModel.ntmpar` property
### Fixed
- Shape of `Fitter.resids.noise_ampls` (it was wrong before due to bad indexing)
- Made `TimingModel.is_binary()` more robust.
- Correct value of (1/year) in `powerlaw()` function
- Fixed `TestPintk`
- Fixed the noise realization indexing in `Fitter`s
- Added missing newline in `tempo_polyco_table_writer()`
- Proper warning in `read_polyco_file()`
### Removed
- Definition of `@cached_property` to support Python<=3.7
- The broken `data.nanograv.org` URL from the list of solar system ephemeris mirrors
- Broken fitter class `CompositeMCMCFitter` (this fitter was added seemingly to deal with combined radio and high-energy datasets, but has since been broken for a while.)
