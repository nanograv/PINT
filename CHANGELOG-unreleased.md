# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved design matrix normalization code from `pint.fitter` to the new `pint.utils.normalize_designmatrix()` function.
- Made `Residuals` independent of `GLSFitter` (GLS chi2 is now computed using the new function `Residuals._calc_gls_chi2()`).
- `TOAs.is_wideband` returns True only if *ALL* TOAs have the -pp_dm flag. Emits a warning if only some TOAs have this flag.
### Added
### Fixed
- Fixed RTD by specifying theme explicitly.
- `.value()` now works for pairParameters
- Setting `model.PARAM1 = model.PARAM2` no longer overrides the name of `PARAM1`
- Fixed an incorrect docstring in pbprime() functions. 
- Fix ICRS -> ECL conversion when parameter uncertainties are not set.
### Removed
