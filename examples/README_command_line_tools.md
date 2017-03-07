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


