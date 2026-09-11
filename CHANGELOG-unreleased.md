# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Every active FBX model now uses FB coefficients as its complete orbital-phase representation. An ordinary `PBDOT` supplied with `FB0` is converted to `FB1` instead of being silently ignored. `PB` and applicable `PBDOT` values are exposed as derived views, so `as_parfile()` gains commented `# PB`/`# PBDOT` lines for existing pure-FBX models and no longer exposes an unset writable `PBDOT` that the FBX orbit would ignore.
- DDGR+FBX is now rejected explicitly before normalization. The new general PB/FB bridge does not broaden DDGR into a partially supported configuration or rely on incidental missing-FB0/PB-conflict failures. Correct support requires dynamic synchronization of DDGR's PB-dependent post-Keplerian quantities and their analytic `PB(FB0)` chain-rule derivatives.
- `BinaryDDH` now accepts a negative fitted `H3` with a warning when `STIGMA` is finite and positive. The DDH delay is smooth in the signed amplitude `H3`, but the resulting negative derived `M2` is not physically interpretable. Validation of free `M2`, generic `SINI`, and non-DDH binary models is unchanged.
### Added
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
- `TOAs.get_tdb_seconds()`, returning the TDB times of the TOAs in seconds with a selectable dtype
- Binary orbital-phase normalization now accepts `PB` (and optionally ordinary `PBDOT`) together with `FBn` while `FB0` is omitted. PINT converts the nominal model to a canonical FBX Taylor series, inserts frozen zero coefficients for sparse indices, and exposes `PB`/`PBDOT` as read-only derived views. Every valued FB coefficient is now guaranteed to contribute to orbital phase or produce a clear error.
### Fixed
- Sparse FBX series no longer silently ignore coefficients after the first missing index.
### Removed
