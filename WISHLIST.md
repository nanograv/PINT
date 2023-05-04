# PINT Development Wishlist

Items with an active/work in progress pull request are marked with (*).

## New features

- Implement `fftfit` in Python instead of relying on `PRESTO` (*)
- Pulsar binary radial velocity computation
- Easy way to pre-download clock files, ephemerides etc.
- Support INPOP ephemerides
- Implement `DRACULA` using `PINT`
- Full support for Bayesian Single Pulsar Noise and Timing Analysis 
- Support for derivative-aware MCMC methods - Requires derivatives w.r.t. noise parameters
- Easy way to extract whitened residuals
- Correlated noise simulation
- Support for platforms without extended precision (`float80` or `float128`)
- `general2`-like command line tool to be used with unix pipes
- Visualize DM model and DM residuals for wideband TOAs using `pintk`
- Implement robust fitting methods (e.g. Huber regression)
### Timing model components
- Explicit fittable absolute phase (*)
- Support for fitting `WAVE` parameters.
- Support `P0` and `P1` in place of `F0` and `F1`
- Implement Keith et al 2013 DM noise model (`DMMODEL`)
- Piecewise orbital model (`BTX`)
- Main sequence binary model (`MSS`)
- Multiple planet system models (`BT1P` and `BT2P`)
- Hyperbolic encounter events (Jennings et al 2020)
- Chromatic events
- Time jumps applied to Site Arrival Times (`SATJUMP`) and Barycentric Arrival Times (`BATJUMP`?)
- System-dependent FD parameters (`FDJUMP`) (*)
- Variable-index chromatic red noise model
- Red noise models (`PLRedNoise` and `PLDMNoise`) should support corner frequency (`TNREDCORNER`) and fundamental frequency of the basis (`TNREDFLOW`) as parameters

## API and functionality improvements
- Support using pulse numbers for a subset of the TOAs
- Use CamelCase uniformly in timing model parameter mapping
- More rigorous validation of timing model components (*)
- Improve/simplify parameter API
- Setting `MJDParameter` from a time object
- Improve random models module (?)
- Improve naming conventions for `maskParameter`s
- Simplify working with TOA flags
- Implement epoch averaging independent of `ECORR`
- `compare_models` should `change_binary_epoch`
- Check TOA flags for IPTA compliance
- Use a faster algorithm to solve Kepler equation
- Easy way to extract correlation matrices

## Performance
- Make pintk more responsive
- Improve performance of orbit interpolator (used for satellite observatories)
- Speed up tests
- Speed up TOA file reading

## Code quality and tests
- Improve unit test coverage
- Audit and clean up skipped and xpassed tests
- Review test tolerances
- Find and remove unused modules and dead code
- Test that the noise models in PINT are fully compatible with `ENTERPRISE`
- Make sure that `PINT` updates don't break downstream codes (`pint_pal` and `ENTERPRISE`)


## Documentation and Examples
- A table of known observatories
- Fix broken example notebooks

## Serious bugs
- Higher order terms in ELL1 model may be wrong
- `CompositeMCMCFitter` is broken