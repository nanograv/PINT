# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- Warning when TOAs read from a `.tim` file repeat a flag with different values (e.g. `-j A -j B`): PINT stores only one value per flag, so masks (JUMPs, noise parameters) selecting the other values silently miss these TOAs, whereas TEMPO2 matches every value
### Fixed
### Removed
