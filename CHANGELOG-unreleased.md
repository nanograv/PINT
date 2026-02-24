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
### Added
- Anderson-Darling test for normal data with fixed mean/variance
- KS test to check if the whitened residuals are unit-normal distributed
- Warning about setting of TZRMJD from TOAs
- Method to zero out mean residual based on TZRMJD
- Use VLBI astrometric measurements along with coordinate offset in the timing model
### Fixed
### Removed
