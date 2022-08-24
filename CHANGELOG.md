# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Changed
- global clock files now emit a warning instead of an exception if expired and the download fails

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
- removed explicit download of IERS and leapsecond data (handled now by astropy)
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
- The CombinedResidual class has API change on the get_data_error(), child residueal class in save as dictionary.
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
- Aded dmxparse function
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
