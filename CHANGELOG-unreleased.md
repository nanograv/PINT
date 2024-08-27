# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved the events -> TOAs and photon weights code into the function `load_events_weights` within `event_optimize`.
- Updated the `maxMJD` argument in `event_optimize` to default to the current mjd
- `maskParameter.__repr__()` output now includes the frozen attribute.
- Changed default value of `FDJUMPLOG` to `Y`
- Bumped `black` version to 24.x
### Added
- arXiv link of PINT noise paper in README
- Type hints in `pint.derived_quantities`, `pint.modelutils`, `pint.binaryconvert`, `pint.config`, 
`pint.erfautils`, `pint.fits_utils`, `pint.logging` and `pint.residuals`
- Doing `model.par = something` will try to assign to `par.quantity` or `par.value` but will give warning
- `plrednoise_from_wavex()` and `pldmnoise_from_dmwavex()` functions now compute `TNRedFLow` and `TNDMFLow`
- `powerlaw_corner` function
- `TNREDFLOW` and `TNREDCORNER` parameters in `PLRedNoise`
- `TNDMFLOW` and `TNDMCORNER` parameters in `PLDMNoise`
- `PLChromNoise` component to model chromatic red noise with a power law spectrum
- Fourier series representation of chromatic noise (`CMWaveX`)
- `pint.utils.cmwavex_setup` and `pint.utils.plchromnoise_from_cmwavex` functions
- More validation for correlated noise components in `TimingModel.validate_component_types()`
### Fixed
- Bug in `DMWaveX.get_indices()` function
- Explicit type conversion in `woodbury_dot()` function
- Documentation: Fixed empty descriptions in the timing model components table
- BIC implementation
- `event_optimize`: Fixed a bug that was causing the results.txt file to be written without the median values. 
- SWX model now has SWXP_0001 frozen by default, and new segments should also have SWXP frozen
- Can now properly use local files for ephemeris
### Removed
- Removed the argument `--usepickle` in `event_optimize` as the `load_events_weights` function checks the events file type to see if the 
file is a pickle file.
- Removed obsolete code, such as manually tracking the progress of the MCMC run within `event_optimize`
- Unnecessary default arguments from the `powerlaw()` function.
- `download_data.sh` script and `de432s.bsp` ephemeris file
