# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Removed
- Removed Python 2.7 support from travis and tox testing suites and from requirements files
- Removed "landscape" code checker since that package is no longer supported by its author
### Fixed
- Fixed bug in processing of PHASE commands in .tim file. They are now applied even if pulse numbers are not being used
- Substantial speed increase in Residuals calculation due to removal of redundant phase calculation
- Fixed bug that prevented reading Parkes-format TOAs
- Fixed bug in solar wind model that prevented fitting
- Fix pintempo script so it will respect JUMPs in the TOA file.
### Added
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
- Added START and FINISH parameters as MJDParameters to timing_model. They are now 
modified after a fit and are displayed with a model's .par file output.
### Changed
- New observatories will no longer overwrite existing ones silently.  Will either raise ValueError or require overwrite=True
- Large speed increase when using Ecliptic coordinates
- Changed Residuals so that use_weighted_mean and subtract_mean are attributes set on initialization

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
