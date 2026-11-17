# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Chromatic delays are now referenced to the new ``CM_FREF`` parameter instead of an implicit 1 MHz: the ``ChromaticCM``, ``ChromaticCMX`` and ``CMWaveX`` delay is ``CM * DMconst * (freq / CM_FREF)**(-TNCHROMIDX)`` (unchanged for the default ``CM_FREF``).
- ``PLChromNoise`` scales its Fourier basis by ``(CHROMNOISE_FREF/f)**TNCHROMIDX`` instead of a hard-coded 1400 MHz reference (unchanged for the default ``CHROMNOISE_FREF``).
- ``PLDMNoise`` scales its Fourier basis by ``(DMNOISE_FREF/f)**2`` instead of a hard-coded 1400 MHz reference (unchanged for the default ``DMNOISE_FREF``).
### Added
- Can output model comparison as JSON
- ``CM_FREF``: reference frequency (MHz) of the chromatic measure, default 1 MHz.
- ``CHROMNOISE_FREF``: reference frequency (MHz) of the ``PLChromNoise`` amplitude, default 1400 MHz.
- ``DMNOISE_FREF``: reference frequency (MHz) of the ``PLDMNoise`` amplitude, default 1400 MHz.
### Fixed
### Removed
