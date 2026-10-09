# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
### Fixed
- A par-file line without a fit flag (`NAME value` or `NAME value uncertainty`) now always gives a frozen parameter (TEMPO2 also defaults missing fit flags to zero). Previously the parameter kept its default, which is unfrozen for the first parameter of some families (e.g. `DMX_0001`, `WXSIN_0001`, `CMX_0001`)
- `dmxparse` with frozen DMX bins no longer fails looking up their covariance: frozen bins are excluded from the mean and its uncertainty and get NaN variance errors, and a model with no fitted DMX bin raises a clear `ValueError`
### Removed
