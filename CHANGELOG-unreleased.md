# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Change `StepProblem` and `MaxIterReached` into warnings
- Removed numpy < 2.4 restriction
- Cache `TimingModel.noise_model_designmatrix()` (`U`) based on TOAs identity and noise-parameter state to avoid redundant basis recomputation in repeated calls
- Improve basis-space covariance handling to support non-diagonal `Phi` blocks throughout noise-model paths, and use Cholesky-based inversion/solves where applicable for stability and performance
### Added
- Anderson-Darling test for normal data with fixed mean/variance
- KS test to check if the whitened residuals are unit-normal distributed
- Warning about setting of TZRMJD from TOAs
- Method to zero out mean residual based on TZRMJD
- Easy method to add new parameters
- Use VLBI astrometric measurements along with coordinate offset in the timing model
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Regression tests for noise design-matrix caching, including multi-basis coverage (red, DMGP, SWGP, and chromatic GP)
### Fixed
- Fix docstring of `make_fake_toas_uniform`
### Removed
