# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the released changes to the codebase. See CHANGELOG-unreleased.md for
the unreleased changes. This file should only be changed while tagging a new version.

## [1.0] 2024-04-26
### Changed
- Moved `get_derived_params` to `timing_model`
- `check_ephemeris_connection` CI test no longer requires access to static NANOGrav site
- `TimingModel.compare()` now calls `change_binary_epoch()`.
- When clock files contain out-of-order entries, the exception now records the first MJDs that are out of order
- `np.compat.long` -> `int` (former is deprecated)
- Turned ErfaWarning into an exception during testing; cleaned up test suite.
- macos-latest runner changed to macos-12 runner for CI tests to avoid M1 architecture issues
### Added
- Added numdifftools to setup.cfg to match requirements.txt
- Documentation: Added `convert_parfile` to list of command-line tools in RTD
- DDH binary model
- function `pint.utils.xxxselections` to do DMX-style selections for any parameter name
- Plot model DM in pintk
- More tests for pintk
- Maximum likelihood fitting for ECORR
    - `is_time_correlated` class attribute in correlated `NoiseComponent`s
    - `has_time_correlated_errors` property in `TimingModel`
    - `Residuals._calc_ecorr_chi2()` method for fast chi2 computation using Sherman-Morrison identity
    - `pint.utils.sherman_morrison_dot` and `pint.utils.woodbury_dot`
    - Refactored repeated code out of `Residuals.calc_phase_mean` and `Residuals.calc_time_mean`
    - Simplified `Residuals._calc_gls_chi2()` so that it uses Woodbury identity directly
    - Refactored WLS chi2 code out of `Residuals.calc_chi2()` into a new function `Residuals._calc_wls_chi2()`
    - `Residuals.d_lnlikelihood_d_whitenoise_param` will throw a `NotImplementedError` when correlated noise is present.
    - `DownhillFitter._fit_noise()` doesn't use derivatives when correlated noise is present.
    - Documentation: Noise fitting example notebook.
- `freeze_params` option in `wavex_setup` and `dmwavex_setup`
- `plrednoise_from_wavex`, `pldmnoise_from_dmwavex`, and `find_optimal_nharms` functions
- fake TOAs can be created with `subtract_mean=False`, to maintain phase coherence between different data sets
- Binary models can be guessed by the `ModelBuilder`. Options and script are added to allow reading/conversion of the T2 binary model
- Better explanation of ELL1H behavior when H3/H4/STIGMA supplied and when NHARMS is used
- FDJumpDM component for System-dependent DM offsets
- Documentation: Explanation for FDJUMP and FDJUMPDM
### Fixed
- `MCMC_walkthrough` notebook now runs
- Fixed runtime data README 
- Fixed `derived_params` when OMDOT has 0 uncertainty
- `model.find_empty_masks` will now also look at DMX and SWX parameters
- Fixed `make_fake_toas_fromtim`
- Use `Hessian` instead of `Hessdiag` in `DownhillFitter._fit_noise`; compute noise parameter uncertainties only once in `DownhillFitter.fit_toas`.
- Consistent naming in `TimingModel.get_params_mapping()`
- Better exceptions for unsupported/unimplemented binary models (BTX, MSS, etc.)
- Emit warnings when `WaveX`/`DMWaveX` is used together with other representations of red/DM noise
- `get_observatory()` no longer overwrites `include_gps` and `include_bipm` of `Observatory` objects unless explicitly stated (BIPM and GPS clock corrections no longer incorrectly applied to BAT TOAs).
- Added back `spacecraft` as an alias for `stl_geo`
- Fix bug 1729 (missing f-string)
- Fixed common failure of test_observatory
- pintk now shows scaled error bars
- Docstring of `DispersionJump`


## [0.9.8] 2023-12-04
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
- `Residuals.calc_whitened_resids()` method
- Plot wideband DM measurements, wideband DM residuals, and wideband DM errors in `pintk`. (Disabled for narrowband data.)
- Optionally generate multi-frequency TOAs in an epoch using `make_fake_toas_uniform` and `make_fake_toas_fromMJDs`
- Documentation: Example notebook for simulations and flag usage
- Proper motion conversion/calculations can now uniformly use float, Quantity, or Time input
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
- Minor bug fixes in example notebooks
- Set `UNITS` to TDB if it's not given in the par file.
### Removed


## [0.9.7] 2023-08-24
### Changed
- Made the addition of a TZR TOA (`AbsPhase`) in the `TimingModel` explicit in `Residuals` class.
- Updated `CONTRIBUTING.rst` with the latest information.
- Made `TimingModel.params` and `TimingModel.ordered_params` identical. Deprecated `TimingModel.ordered_params`.
### Added
- Third-order Roemer delay terms to ELL1 model
- Options to add a TZR TOA (`AbsPhase`) during the creation of a `TimingModel` using `ModelBuilder.__call__`, `get_model`, and `get_model_and_toas`
- `pint.print_info()` function for bug reporting
- Added an autocorrelation function to check for chain convergence in `event_optimize`
- A hacky implementation of system-dependent FD parameters (FDJUMP)
- Minor doc updates to explain default NHARMS and missing derivative functions
### Fixed
- Deleting JUMP1 from flag tables will not prevent fitting
- Simulating TOAs from tim file when PLANET_SHAPIRO is true now works
- Docstrings for `get_toas()` and `get_model_and_toas()`
- Set `DelayComponent_list` and `NoiseComponent_list` to empty list if such components are absent
- Fix invalid access of `PLANET_SHAPIRO` in models without `Astrometry`
### Removed


## [0.9.6] 2023-06-22
### Changed
- Applied `sourcery` refactors to the entire codebase
- Changed threshold for `test_model_derivatives` test to avoid CI failures
- Unreleased CHANGELOG entries should now be entered in `CHANGELOG-unreleased.md` instead of `CHANGELOG.md`. Updated documentation accordingly.
- Changed tests to remove `unittest` and use pure pytest format
- Changed deprecated `sampler.chain` usage
- Download data automatically in the profiling script `high_level_benchmark.py` instead of silently giving wrong results.
### Added
- `SpindownBase` as the abstract base class for `Spindown` and `PeriodSpindown` in the `How_to_build_a_timing_model_component.py` example.
- `SolarWindDispersionBase` as the abstract base class for solar wind dispersion components.
- `validate_component_types` method for more rigorous validation of timing model components.
- roundtrip test to make sure clock corrections are not written to tim files
- `calc_phase_mean` and `calc_time_mean` methods in `Residuals` class to compute the residual mean.
- `PhaseOffset` component (overall phase offset between physical and TZR toas)
- `tzr` attribute in `TOAs` class to identify TZR TOAs
- Documentation: Explanation for offsets
- Example: `phase_offset_example.py`
- method `AllComponents.param_to_unit` to get units for any parameter, and then made function `utils.get_unit`
- can override/add parameter values when reading models
- docs now include list of observatories along with google maps links and clock files
### Fixed
- fixed docstring for `add_param_from_top`
- Gridded calculations now respect logger settings
- Event TOAs now have default error that is non-zero, and can set as desired
- Model conversion ICRS <-> ECL works if PM uncertainties are not set
- Fix `merge_TOAs()` to allow lists of length 1
### Removed

## [0.9.5] 2023-05-01
### Changed
- Changed minimum supported version of `scipy` to 1.4.1
- Moved `DMconst` from `pint.models.dispersion_model` to `pint` to avoid circular imports
- Removed references to `astropy._erfa` (removed since `astropy` 4.2)
- Refactor `Dre` method, fix expressions for Einstein delay and post-Keplerian parameters in DD model
- Updated contributor list (AUTHORS.rst)
- Emit an informative warning for "MODE" statement in TOA file; Ignore "MODE 1" silently
- Version of `sphinx-rtd-theme` updated in `requirements_dev.txt`
- Updated `black` version to 23.x
- Older event loading functions now use newer functions to create TOAs and then convert to list of TOA objects
- Limited hypothesis to <= 6.72.0 to avoid numpy problems in oldestdeps
### Added
- Documentation: Explanation for DM
- Methods to compute dispersion slope and to convert DM using the CODATA value of DMconst
- `TimingModel.total_dispersion_slope` method
- Explicit discussion of DT92 convention to DDK model
- HAWC, HESS and ORT telescopes to the list of known observatories
- Documentation: making TOAs from array of times added to HowTo
- Method to make TOAs from an array of times
- Clock correction for LEAP
- Wideband TOA simulation feature in `pint.simulation` and `zima`
- ELL1k timing model
- Test for `MCMCFitter`
- Added multiprocessing capability to `event_optimize`
- Can merge TOAs using '+' operator; in-place merge using '+=' operator
- `funcParameters` defined as functions operating on other parameters
- Option to save `emcee` backend chains in `event_optimize`
- Documentation on how to extract a covariance matrix
- DDS and DDGR models
- Second-order corrections included in ELL1
- Module for converting between binary models also included in `convert_parfile`
- Method to get a parameter as a `uncertainties.ufloat` for doing math
- Method to get current binary period and uncertainty at a given time regardless of binary model
- TCB to TDB conversion on read, and conversion script (`tcb2tdb`)
- Functions to get TOAs objects from satellite data (Fermi and otherwise)
- Methods to convert a TOAs object into a list of TOA objects
### Fixed
- Syntax error in README.rst
- Broken notebooks CI test
- BIPM correction for simulated TOAs
- Added try/except to `test_pldmnoise.py`/`test_PLRedNoise_recovery` to avoid exceptions during CI
- Import for `longdouble2str` in `get_tempo_result`
- Plotting orbital phase in `pintk` when FB0 is used instead of PB
- Selection of BIPM for random models
- Added 1 sigma errors to update the postfit parfile errors in `event_optimize`
- Fixed DDS CI testing failures
- Add SolarSystemShapiro to the timing model only if an Astrometry component is present.
### Removed

## [0.9.3] 2022-12-16
### Added
- Method to identify mask parameters with no TOAs and optionally freeze them
### Fixed
- Creating fake TOAs properly handles site clock corrections
- Corrected a precision issue with reading ASCII representations of pulse profiles
- Fixed matplotlib 3.6 import issue in pintk
### Removed
- termios import for solar_wind_dispersion

## [0.9.2] 2022-11-30
### Changed
- Minimum supported versions updated to numpy 1.18.5, matplotlib 3.2.0
- `introduces_correlated_errors` is now a class attribute of `NoiseComponent`s
### Added
- Can ignore pulse_number column on TOA read or write (to help merging)
- Can add in missing columns when merging unless told not to
- Can initialize observatories with lat/lon/altitude
- Can output observatories as JSON
- Can extract single TOAs as length=1 table
- SWM=1 models can be used
- SWX models to fit the solar wind over various intervals
- Added a pintk helper function to delete jumped TOAs/remove existing jumps. Fixed indexing issue for single clicks.
- Added PLDMNoise component which allows modeling of stochastic DM variations as red noise with a power law spectrum
- Added Bayesian interface (Timing model and white noise only)
- Can add multiple DMX values at once
- Can add overlapping DMX ranges
- New tests to improve test coverage
- Documentation: Instructions to checkout development branch
- Clock file for effix
- Added energy dependent templates to the lctemplates utilities and added tests
### Fixed
- global clock files now emit a warning instead of an exception if expired and the download fails
- dmxparse outputs to dmxparse.out if save=True
- Excluded noise parameters from the design matrix.
- Split the computation of correlated noise basis matrix and weights into two functions.
- Fixed bug in combining design matrices
- Fixed bug in dmxparse
- Fixed bug in photonphase with polycos
- Made clock file loading log entries a little friendlier
- Typo fixes in documentation
- Fixed failing HealthCheck in tests/test_precision.py
### Removed
- Removed obsolete `ltinterface` module
- Removed old and WIP functions from `gridutils` module


## [0.9.1] 2022-08-12
### Changed
- No tests now change based on $TEMPO or $TEMPO2
- Ensure Fitters work with ELL1 even on Astropy 4 (bug #1316)
- index.txt is only checked at most once a day
- Moved observatories to JSON file.  Changed way observatories are loaded/overloaded
- Split Jodrell Bank observatory based on backend to get correct clock files
- Clock files can be marked as being valid past the end of the data they contain
- Polycos can be written/read from Path or Stream objects
- Polyco format registration now done once as a class method
- Polyco reading/generation from timing model done as class methods
### Added
- delta_pulse_number column is now saved to -padd flag on TOA write
- command-line utility to compare parfiles
- FD_delay_frequency function to easily access the FD model's excess delay
- scripts now have explicit setting of verbosity and `-q`/`-v` options
### Fixed
- TOA flags are properly deepcopy'd when desired (to deal with [astropy bug](https://github.com/astropy/astropy/issues/13435))

## [0.9.0] 2022-06-24
### Changed
- `model.phase()` now defaults to `abs_phase=True` when TZR* params are in the model
- TOAs no longer need to be grouped by observatory
- removed explicit download of IERS and leap second data (handled now by astropy)
- The default version of TT(BIPM) uses BIPM2021
- ClockFile no longer uses metaclass magic or many subclasses, and have friendly names for use in messages
- `model.setup()` now gets called automatically after removing a parameter as part of `remove_param`
- Cleaned up handling of telescopes with no clock files so they don't emit ERROR messages
### Added
- logging now needs to be setup explicitly
- Color-by-jump mode for pintk
- `pytest-xdist` now allows `pytest -n auto` to use all cores on the machine to run tests in parallel; `make test` now does this.
- Added the ability to write clock files in TEMPO or TEMPO2 format
- Added examples of how to write a par file to tutorials
- Added `TimingModel.write_parfile()`
- Added generator for iterating over like items in an array
- Added iterator to iterate over observatory groups
- Clock files are now searched for in the directory PINT_CLOCK_OVERRIDE
- Clock files are now searched for in the online global repository
- You can export the clock files you are using with `export_all_clock_corrections()`
- You can request that all your clock files be updated and loaded into the cache with `update_clock_files()` 
- The `temp_cache` fixture that runs tests with an empty, scratch Astropy cache
### Fixed
- Selecting of TOAs in `pintk` was broken if some TOAs were deleted (bug #1290)
- INCLUDE lines in tim files are now relative to the location of the tim file (bug #1269)
- jump_flags_to_params now works if some JUMPs are present, never modifies the TOAs, and is idempotent
- jump_params_to_flags is now idempotent and unconditionally sets the -jump flag to a correct state
### Changed
- Required version of python updated to 3.8

## [0.8.8] 2022-05-26
### Added
- Warning when A1DOT parameter used with DDK model
- Added the limits="warn" or limits="error" to get_TOAs to select handling of uncorrected TOAs
- Added functions in pint.observatory to request the status of PINT's available clock corrections
- Added the ability to query clock correction files or observatories for their last corrected MJD
- Added an example showing how to check the status of your clock corrections
### Fixed
- WLS fitters no longer ignore EFAC/EQUAD (bug #1226)
### Changed
- Clock correction files that are entirely missing are handled the same way as TOAs past the end of a clock correction file
- Observatory objects can now note that their clock correction files include a bogus "99999" (or similar) entry at the end
- Clock correction files are now checked for being in order (necessary for PINT's interpolation to function)
- Observatories using TEMPO-format clock files now refer to the clock file (for example time_ao.dat) rather than via time.dat
- Observatories that don't use clock corrections (for example CHIME uses GPS time directly) just use an empty list of clock correction files rather than files containing only zeros
- Updated Jodrell clock corrections to include post-1997
- DDK model will now use ICRS or ECL coordinates depending on what the input model is

## [0.8.6 == 0.8.7] 2022-05-10
### Added
- Added computation of other Kopeikin solutions (`solutions = model.alternative_solutions()`)
- Added computation of extra parameters in gridding
- Added gridding based on tuples of parameters (not just regular mesh)
- Added passing of extra parameters to gridding fitter
- Added option to photonphase to compute phases using polycos
- New `colorize()` function in `pint.utils` which can be used for string and unicode output
- Split out get_derived_params() from get_summary() in fitter.py.  Can be used other places.
- `pintk` can now automatically select the right fitter, and can otherwise specify fitter from the command-line or a dropdown
- automatic fitter selection available with standard API as well
- added tempo(2) par/tim output to `pintk` 
- added icon to `pintk`
### Fixed
- Huge number of bugs and improvements to `pintk`, and some to `pintempo`
- Multiple bug fixes in get_summary()/get_derived_params(), especially for binary calculations
- DMDATA now an integer for tempo/tempo2 parfile output
### Changed
- Changed logging to use `loguru`
- Changed to floating point format and added color for correlation matrix output
- prefix parameters no longer inherit frozen status
- Updated clock files for GBT (`time_gbt.dat`) and GPS to UTC conversion (`gps2utc.clk`) with new entries

## [0.8.5] 2022-02-24
### Added
- Added support for Chandra and Swift event data in photonphase.py
### Fixed
- Attempt to fix documentation build by removing very slow notebooks
- Improved compatibility with TEMPO/Tempo2 parfiles
- Fixed handling of Swift and Chandra FITS files (PR #1157)
- Cleaned up some deprecated usages
### Changed
- Gridding code now allows options to be supplied and other parameters to be returned (PR #1173)

## [0.8.4] 2021-10-06
### Fixed
- 0.8.3 was tagged without an updated CHANGELOG. This fixes that.
- Now ensures T2CMETHOD is IAU2000B if it is set at all; likewise DILATEFREQ and TIMEEPH (PR #970)
- Merging TOAs objects now ensures that their index columns don't overlap (PR #1029)
- change_dmepoch now works even if DMEPOCH is not set (PR #1025)
- Fixed factor of 2 error in d_phase_d_toa() (PR #1129)
- Fixed change_binary_epoch (PR #1120)
### Added
- DownhillWLSFitter, DownhillGLSFitter, WidebandDownhillFitter are new Fitters that are more careful about convergence than the existing ones (PR #975)
- Fitters have a .is_wideband boolean attribute (PR #975)
- TOAs now have a .renumber() method to simplify their index column (PR #1029)
- TOAs objects have a .alias_translation attribute to allow them to output TEMPO-compatible observatory names (PR #1017)
- TimingModel objects now remember which alias their parameters were called when read in and write those out by default; this can be overridden with the .use_aliases() method to ensure PINT-default or TEMPO-compatible names. (PR #1017)
- New function utils.info_string() to return information about the PINT run (user, version, OS, optional comments).  This is run during TOA output to tim file or model output to par file by default, but can be suppressed by setting include_info=False (PR #1069)
- New functions for calculations of post-Keplerian parameters (PR #1088)
- New tutorial for simulating data and making a mass-mass plot (PR #1096)
- Added better axis scaling to pintk (PR #1116)
### Changed
- Changed observatory coordinates for CHIME, AO, EFF, and JB (PRs #1143, #1145)
- get_groups() is renamed to get_clusters() and is no longer automatically called during TOA creation.  Can still be run manually, with the gap specified.  Addition of a clusters column to the TOA.table object is optional (PR #1070)
- Some functions from utils.py are now in derived_quantities.py (PR #1102)
- Data for tutorials etc. and clock files are now installed properly, with locations retrievable at runtime (PR #1103)
- Changed from using astropy logging to python logging (PR #1093)
- Code coverage reports are now informational and don't cause CI to fail (PRs #1085, #1087)
- API for TOA flags significantly changes, now only hold strings and allow fancy indexing (PR #1074)

## [0.8.2] - 2021-01-27
### Fixed
- Now preserves the name column in tempo2 files (PR #926)
- Make_fake_toas now uses ephemeris and other settings from the model (PR #926)
- Fix dof bug when updating TOAs (PR #955)
### Added
- get_TOAs can read and cache multiple .tim files (PR #926)
- pickling can be done manually with load_pickle and save_pickle (PR #926)
- TOAs can be checked against the files they were loaded from with check_hashes() (PR #926)
- TOAs can now be checked for equality with == (PR #926)
- Add bounds checking for spacecraft obs and other changes (PR #961)
- Added script/notebook to reproduce profiling tables from PINT paper (PR #934)
### Changed
- Improvements to pulse numbering and track mode
- Removed all __future__ stuff that supported Python 2 (PR #946)

## [0.8.1] - 2021-01-07
### Fixed
- Right click to delete TOAs in pintk now works
- Added exception if orbit extrapolates for satellite observatories
- Fixed Actions to compute and upload coverage
- Doc building fixes
- Fixed badges in README
- Added GitHub Actions for CI testing
- Fix setup.cfg to disable Py2.7 support
- Fixed bug in
### Removed
- Removed two unused files
- Removed use of Travis-CI
### Changed
- Sped up some tests
### Added
- Added Python 3.9 support
- Added DMX support functions add_DMX_range() and remove_DMX_range()
- Improvements to make_fake_toas() to support wideband TOAs

## [0.8] - 2020-12-21
### Fixed
- Fixed an indentation bug in Wideband TOA fitting.
- The CombinedResidual class has API change on the get_data_error(), child residual class in save as dictionary.
### Removed
- Removed Python 2.7 support from travis and tox testing suites and from requirements files
- Removed "landscape" code checker since that package is no longer supported by its author
- Removed scale_by_F0 and scaled_by_F0 as they did not do what they were intended to (PR #861)
### Fixed
- Fixed bug in processing of PHASE commands in .tim file. They are now applied even if pulse numbers are not being used
- Substantial speed increase in Residuals calculation due to removal of redundant phase calculation
- Fixed bug that prevented reading Parkes-format TOAs
- Fixed bug in solar wind model that prevented fitting
- Fix pintempo script so it will respect JUMPs in the TOA file.
- Uncertainties are no longer set to zero if some TOAs lack EFACs. (PR #890)
- Fixed solar wind calculation (PR #894)
- RMS functions in pintk code are now correct (PR #876)
- Fixed orbital phase calculations on ELL1 (PR #795)
### Added
- Added merge_TOAs() function in pint.toa to merge compatible TOAs instances (PR #908)
- Added a get_model_and_toas() function in model_builder to read both, including model-based commands affecting the TOAs (PR #889)
- Added ability to load TOAs including relevant commands (e.g. EPHEM, CLOCK, PLANET_SHAPIRO) from a timing model in get_TOAs() (PR #889)
- Added metadata to observatory definition, to keep track of the data origin
- Added other bipm???? files from TEMPO2
- Added ability to find observatories in [astropy](https://github.com/astropy/astropy-data/blob/gh-pages/coordinates/sites.json) if not present in PINT
- Added is_binary property, and orbital_phase() and conjunction() methods to the timing model
- Allow fitting for either or (not both) of a glitch epoch or its phase
- Added support for -padd flag on TOAs to add phase turns to individual TOAs (matching TEMPO and Tempo2)
- Added caching of TZRMJD TOA to speed up and prevent repeated INFO prints about applying clock corrections
- Added check to ensure clock files are ordered by MJD since interpolation assumes that
- Added ability to disable subtracting mean from residuals
- Added track_mode to Residuals to select pulse number tracking without needing the model to have TRACK -2
- Added support for wideband-TOA fitting (Pennucci 2019).
- Added START and FINISH parameters as MJDParameters to timing_model. They are now modified after a fit and are displayed with a model's .par file output.
- Added solar_angle calculation (PR #892)
- Added parameters to TimingModel to support TEMPO/TEMPO2 compatible par files. (PR #900)
- Added position vectors to Neptune (PR #901)
- Added checking for TOAs in DMX bins and other similar parameters, if free (PR #874)
- Added PiecewiseSpindown model, for spindown correction lasting for a given MJD range
### Changed
- New observatories will no longer overwrite existing ones silently.  Will either raise ValueError or require overwrite=True
- Large speed increase when using Ecliptic coordinates
- Changed Residuals so that use_weighted_mean and subtract_mean are attributes set on initialization
- Refactored code for orbiting observatories, most now share a single class, see observatory/satellite_obs.py
- Fixed 2mus amplitude bug in TT to TDB conversion for orbiting observatories.
- Changed requirements to astropy>=4.0
- get_model can now read from file-like, including StringIO, objects (handy for testing) (PR #871)
- WidebandDMResiduals now support access to their parts through .toa and .dm attributes (PR #861)
- Fitters now update the fitted model to record things like EPHEM used for the TOAs (PR #900)
- EFACs and EQUADs can be set independently from each other now. (PR #890)
- WLSFitter and GLSFitter and WidebandTOAFitter can now report degenerate parameter combinations (PR #874)
- Raise an exception if the DDK model is provided with SINI (PR #864)
- Free parameters on models can now be set by assigning to model.free_params (PR #871)
- Deprecated chi2_reduced in favor of reduced_ch2 (which we already had) (PR #859)
- TimingModel objects now act like dictionaries of their parameters (PR #855)
- Notebooks are stored in a more convenient format and are now downloadable from the PINT documentation (PR #849)
- TOAs objects now support fancy indexing to select subsets (PR #844)
- Fitters can now respect pulse numbers (PR #814)
- Updated clock files (PR #835)
- Pickles are now (partially) recomputed if the ephemeris or other settings change (PR #838)
- Default BIPM version is BIPM2019 (was BIPM2015)
## [0.7.0] - 2020-05-27
### Changed
- Changed units of Phase to be u.dimensionless_unscaled instead of u.cycle, which was confusing
- Added checkbox to enable/disable random model plotting in GUI
- Changed algorithm for basic dmx_ranges() function.
- Renamed old dmx_ranges() to dmx_ranges_old() and fix bug under Python 2.7
### Added
- Added safety check so for pickled TOAs to ensure they were created with same PINT version
- Added unit tests for Phase()
- Added __mul__ and __rmul__ to Phase() class
- Added observatory locations for LST and MAGIC gamma-ray observatories
### Fixed
- Fixed missing clock correction info when unpickling TOAs object
- Fixed some bugs in GUI plotting
- Fixed units usage in test that used TimeDelta

## [0.6.3] - 2020-05-04
### Added
- Added pmtot() convenience function
- Added dmxstats() utility function
- Added chisq gridding utilities
### Fixed
- Cleaned up some unnecessary warnings in tests
### Changed
- Defer updating IERS B from when pint is imported to when erfautils is imported (for conda testing)
- Fixed installation instructions in README

## [0.6.2] - 2020-05-04
### Changed
- Removed deprecated pytest-runner from setup.cfg

## [0.6.1] - 2020-04-30
### Added
- Added dmx_ranges to compute DMX bins and build a Component
- Add function to compute epoch averaged residuals based on ECORR
- Added model comparison pretty printer
- Added functions to change PEPOCH, DMEPOCH, and binary epoch
- Added dmxparse function
- Added code to ensure that IERS B table is up to date
- Added fitter.print_summary()
### Changed
- Changed API for adding and removing model components
- Increased minimum version required for numpy
- Change calculation of positions and velocities to use astropy
- Changed the way scaled parameters like PBDOT are handled (no longer scales units)
- Reworked tutorial notebooks in docs
- Changed random_models to return list of models
- Adjusted logging to have fewer unnecessary INFO messages
### Removed
- Remove Python 3.5 support
### Fixed
- Fixed incorrect calculation of degrees of freedom in fit
- Fixed incorrect uncertainties on RAJ, DECJ
- Fix some bugs when adding JUMPs
- Fixed bug in zima plotting
- Fixed bug in Powell fitter (actually accommodate upstream issue with scipy)

## [0.5.7] - 2020-03-16
### Added
- First release using PyPI
- Initial entry in CHANGELOG 
