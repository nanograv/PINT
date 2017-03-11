# PINT Command-line tools

PINT comes with several command line tools that perform several useful
tasks, without needing to write a Python script.  The scripts are installed
automatically by setup.py into the bin directory of your Python distro, so they
should be found in your PATH.

Examples of the scripts are below.  It is assumed that you are running in the 
"examples" subdirectory of the PINT distro.

All of the tools accept `-h` or `--help` to provide a description of the options.

## pintbary

`pintbary` does quick barycentering calculations, convering an MJD(UTC) 
on the command line to TDB with barycentric delays applied.
The position used for barycentering can be read from a par file or
from the command line

```
pintbary 56000.0 --parfile J0613-sim.par 
```

```
pintbary 56001.0 --ra 12h13m14.2s --dec 14d11m10.0s --ephem DE421
```

## pintempo

`pintempo` is a command line tool for PINT that is similar to `tempo` or `tempo2`.
It takes two required arguments, a parfile and a tim file. 

```
pintempo --plot NGC6440E.par NGC6440E.tim
```

## zima

`zima` is a command line tool that uses PINT to create simulated TOAs

```
zima NGC6440E.par fake.tim
```

## nicerphase

This tool reads FITS event files from the NICER experiment (also compatible with
RXTE) and computes phases for each photon, according to a timing model. The phases
can be plotted or output as a FITS file column (to be implemented).

```
nicerphase B1509_RXTE_short.fits FPorbit_Day6223 J1513-5908_PKS_alldata_white.par
```

## fermiphase

This tool uses PINT to read Fermi LAT event (FT1) files and compute phases
for each photon.  Currently just makes a plot of the phaseogram, but should
be extended to write PULSE_PHASE column back to FITS file.

Currently works only with geocentered events (as produced by the Fermi Science Tool `gtbary tcorrect=geo`), or barycentered events.  Should be fixed by adding a Fermi observatory
that can look up spacecraft positions in FT2 file.

```
fermiphase J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits PSRJ0030+0451_psrcat.par CALC
```

## event_optimize

This code uses PINT and emcee to do an MCMC likelihood fitting of a timing model
to a set of Fermi LAT photons.  Currently requires the Fermi FT1 file to
contain *geocentered* events (usually from `gtbary tcorrect=geo`).

The code reads in Fermi photon events, along with 
a par file and a pulse profile template, and optimizes the timing model
using an MCMC sampling process.  The parameters to fit and their
priors are determined by reading the par file.  It can use photon weights,
if available, or compute them based on a simple heuristic computation, if 
desired.  There are many options to control the behavior.

An example run is shown below, using sample files that are included in
the examples subdirectory of the PINT distro.

```
event_optimize J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits PSRJ0030+0451_psrcat.par templateJ0030.3gauss --weightcol=PSRJ0030+0451 --minWeight=0.9 --nwalkers=100 --nsteps=500
```

## htest_optimize

This is an unsupported script that uses the emcee framework to optimize a timing 
model based on the H-test, rather than a full likelihood like event_optimize uses.
This code is not installed by setup.py and is not tests. Use or adapt at your
own risk...


