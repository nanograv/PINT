# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- `WAVE` parameters can be added to a `Wave` model with `add_wave_component()` in `wave.py` 
- Moved design matrix normalization code from `pint.fitter` to the new `pint.utils.normalize_designmatrix()` function.
- Made `Residuals` independent of `GLSFitter` (GLS chi2 is now computed using the new function `Residuals._calc_gls_chi2()`).
- `ssb_to_psb_ICRS` implementation is now a lot faster (uses erfa functions directly)
- `ssb_to_psb_ECL` implementation within `AstrometryEcliptic` model is now a lot faster (uses erfa functions directly)
- Upgraded versioneer for compatibility with Python 3.12
- Creation of `Fitter` objects will fail if there are free unfittable parameters in the timing model.
- Only fittable parameters will be listed as check boxes in the `plk` interface.
- Update CI tests for Python 3.12
- Made `test_grid` routines faster
- `pintk` uses downhill fitters by default
### Added
- CHI2, CHI2R, TRES, DMRES now in postfit par files
- Added `WaveX` model as a `DelayComponent` with Fourier amplitudes as fitted parameters
- `Parameter.as_latex` method for latex representation of a parameter.
- `pint.output.publish` module and `pintpublish` script for generating publication (LaTeX) output.
- Added radial velocity methods for binary models
- Support for wideband data in `pint.bayesian` (no correlated noise).
- Added `DMWaveX` model (Fourier representation of DM noise)
- Piecewise orbital model (`BinaryBTPiecewise`)
- `TimingModel.fittable_params` property
- Simulate correlated noise using `pint.simulation` (also available via the `zima` script)
- `pintk` will recognize when timfile and parfile inputs are switched and swap them
- `pintk` can plot against solar elongation
- Optionally return the the log normalization factor of the likelihood function from the `Residuals.calc_chi2()` method.
- `DownhilWLSFitter` can now estimate white noise parameters and their uncertainties.
- `Residuals.lnlikelihood()` method
- `pint.utils.akaike_information_criterion()` function
- `TimingModel.d_toasigma_d_param` method to compute derivatives of scaled TOA uncertainties w.r.t. white noise parameters.
- `TimingModel.toasigma_derivs` property to get all derivatives functions of scaled TOA uncertainties.
- `ScaleToaError.register_toasigma_deriv_funcs` method to populate derivatives of scaled TOA uncertainties.
- `ScaleToaError.d_toasigma_d_EFAC` and `ScaleToaError.d_toasigma_d_EQUAD` methods.
- Separate `.fullname` for all observatories
- `pint.models.chromatic_model.Chromatic` as the base class for variable-index chromatic delays.
- `pint.models.chromatic_model.ChromaticCM` for a Taylor series representation of the variable-index chromatic delay.
### Fixed
- Wave model `validate()` can correctly use PEPOCH to assign WAVEEPOCH parameter
- Fixed RTD by specifying theme explicitly.
- `.value()` now works for pairParameters
- Setting `model.PARAM1 = model.PARAM2` no longer overrides the name of `PARAM1`
- Fixed an incorrect docstring in `pbprime()` functions. 
- Fix ICRS -> ECL conversion when parameter uncertainties are not set.
- `get_TOAs` raises an exception upon finding mixed narrowband and wideband TOAs in a tim file. `TOAs.is_wideband` returns True only if *ALL* TOAs have the -pp_dm flag.
- `TimingModel.designmatrix()` method will fail with an informative error message if there are free unfittable parameters in the timing model.
- `make_fake_toas_uniform` and `make_fake_toas_fromMJDs` respects units of errors
- Robust access of EPHEM and PLANET_SHAPIRO in `make_fake_toas_fromtim`
- `pintk` will not allow choices of axes that are not in timing model/data
- `pintk` correctly displays initial log level
- Fixed sign of y coordinate for Pico Veleta observatory (also being fixed in tempo2)
### Removed
