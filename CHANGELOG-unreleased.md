# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- TCB/TDB conversion now matches PINT's IAU 2006/Astropy TDB forward model,
  keeps radio frequency undilated, converts FD/FDJUMP and order-aware DM
  coefficients, leaves PX and UTC/data-span selectors explicitly invariant,
  and reports unsupported active deterministic terms. Converted epochs and DM
  values therefore differ from the previous legacy-IFTE conversion. The
  compatibility alias `IFTE_K` now denotes the IAU/ERFA rate; the obsolete
  `IFTE_MJD0` and `IFTE_KM1` module constants have been removed. The no-refit
  contract is now stated explicitly: with TOAs, clocks and ephemeris held
  fixed, an accepted conversion reproduces the source model's residuals to
  better than 1 ns with nothing refitted, up to the overall phase offset that
  each timing package fixes by its own convention.
- TCB/TDB conversion of the DM family now follows the `DILATEFREQ` declared by
  the input par instead of always assuming `N`. TEMPO2 defaults to
  `DILATEFREQ Y` and writes it into the TCB par files it produces; converting
  such a par with the undilated exponent was wrong by `K^2-1 = 3.1e-8` of the
  dispersion delay (~3 ns at 1.4 GHz, ~40 ns at 400 MHz for DM=50) and, being
  chromatic, was not absorbed by the phase offset. The flag as read is recorded
  in `TimingModel.meta["tcb_source_dilatefreq"]`, parameters declare their
  frequency power through the new `tcb2tdb_freq_power` metadata, and a dilated
  source leaves the frequency-dependent components unaudited, since PINT
  evaluates undilated frequencies and cannot certify that branch by closure.
- Documented that PINT evaluates IAU 2006 Resolution B3 TDB (Astropy/ERFA), while TEMPO2 currently
  labels a different realization `UNITS TDB`: Irwin & Fukushima's *Teph*,
  which differs from IAU 2006 TDB by a constant 64.5 ns plus a rate of
  2.8e-18 (1.8 ns over a 20-year span). The constant falls inside the phase
  offset and the rate inside the `F0` column, so neither moves a fitted
  solution, but a par file labelled `TDB` is not bit-for-bit the same object
  in the two packages.
- `d_delay_d_param` now applies the chain rule for delay components that respond to delays accumulated from earlier components (e.g. a binary delay's dependence on its evaluation epoch). Design-matrix entries for binary pulsars change at the ~1e-5 relative level. The parameter-independent ingredients are computed once per design matrix via the new `TimingModel.delay_deriv_chain`; the unused `acc_delay` argument of `d_delay_d_param` was replaced by the optional `chain` argument.
### Added
- Support for hierarchical triple systems: a second (outer) binary component can be added via a `BINARY2` line with `_2`-suffixed orbital parameters (e.g. `PB_2`, `A1_2`). Outer orbit delay is computed before, and propagated into, the inner binary. Outer wrappers `BinaryDD2`, `BinaryBT2`, and `BinaryELL12` are provided. Delay derivatives account for the outer→inner coupling (chain rule through the previous delay). The projected semi-major axis includes a second time derivative `A1DOT2` (alias `X2DOT`).
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
- `TOAs.get_tdb_seconds()`, returning the TDB times of the TOAs in seconds with a selectable dtype
### Fixed
- TCB/TDB parameter scaling uses ``x + x (K^n-1)`` with ``K^n-1`` formed
  from ``L_B``, so a float64 rounding of ``K`` (a factor near 1) cannot
  put a ~10 ns error into ``F0`` over 1250 d.
- Place ``solar_windx`` before the binary in ``DEFAULT_ORDER`` so SolarWindDispersionX delays and derivatives chain-rule through the binary the same way as ``solar_wind``.
### Removed
