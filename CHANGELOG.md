# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2021-01-07
## Fixed
- Right click to delete TOAs in pintk now works
- Added exception if orbit extrapolates for satellite observatories
- Fixed Actions to compute and upload coverage
- Doc building fixes
- Fixed badges in README
- Added GitHub Actions for CI testing
- Fix setup.cfg to disable Py2.7 support
- Fixed bug in 
## Removed
- Removed two unused files
- Removed use of Travis-CI
## Changed
- Sped up some tests
## Added
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
