# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Simulation functions no longer subtract the residual mean by default.
### Added
- Explicitly specify the Tspan for power-law GP noise parameters (TN*TSPAN)
- TCB <-> TDB conversion for power-law GP noise parameters.
### Fixed
- TN*C parameter are now `intParameters`
- Type hints in `pint.fitter`
- Made `TimingModel.is_binary()` more robust. 
- Bug in `Fitter.plot()`
### Removed
