# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
### Added
- When TCB->TDB conversion info is missing, will print parameter name
- `add_param` returns the name of the parameter (useful for numbered parameters)
### Fixed
- When EQUAD is created from TNEQ, has proper TCB->TDB conversion info
- TOA selection masks will work when only TOA is the first one
### Removed
