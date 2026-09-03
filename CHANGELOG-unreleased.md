# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Chromatic delays are now referenced to the new ``CM_FREF`` parameter (default 1400 MHz) instead of an implicit 1 MHz: the ``ChromaticCM``, ``ChromaticCMX`` and ``CMWaveX`` delay is ``CM * DMconst * (freq / CM_FREF)**(-TNCHROMIDX)``. CM (and CMX/CMWX amplitude) values in existing par files describe the same delay only after dividing by ``CM_FREF**TNCHROMIDX``.
- ``PLChromNoise`` scales its Fourier basis by ``(CM_FREF/f)**TNCHROMIDX`` instead of a hard-coded 1400 MHz reference (unchanged for the default ``CM_FREF``).
- ``PLDMNoise`` scales its Fourier basis by ``(DM_FREF/f)**2`` instead of a hard-coded 1400 MHz reference (unchanged for the default ``DM_FREF``).
### Added
- ``CM_FREF``: reference frequency (MHz) of the chromatic measure, default 1400 MHz, frozen by default.
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
- `TOAs.get_tdb_seconds()`, returning the TDB times of the TOAs in seconds with a selectable dtype
### Fixed
### Removed
