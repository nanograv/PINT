PINT
====

## PINT is not TEMPO3

PINT is a project to develop a new pulsar timing solution based on
python and modern libraries.  It is in the _extremely_ early stages of
development.

The primary reasons we are developing PINT are:
  - To have a robust system to check high-precision timing results that is completely independent of TEMPO and TEMPO2
  - To make a system that is easy to extend and modify due to a good design and the use of a modern programming language, techniques, and libraries

## Some design philosophy
 - Emphasis on modularity
 - Well defined interface
 - Mostly written in python
 - Heavy use of well-debugged libraries
   - JPL SPICE
   - astropy
   - ERFA (i.e. "freed" version of SOFA)
   - optimization libraries (scipy, pyMinuit, emcee)
 - GUI will not be required
 - Strong isolation of parts needing extended precision
 - Full use of docstrings
 - Self-describing objects
 - Effective use of the Python standard library
    (i.e. text reading, parsing, simple DB, binary search, etc)
 - Backwards compatible with TEMPO
 - Code independence from TEMPO and TEMPO2
 - Use of defined constants
 - Unit tests

## Team members or interested parties include (alphabetical order):
 - Anne Archibald
 - Paul Demorest
 - Luo Jing
 - Rick Jenet
 - Scott Ransom
 - Chris Sheehy
 - Michele Vallisneri
 - Rutger van Haasteren

## Required packages/libraries
 - numpy
 - mpmath (the hope is to phase out this requirement)
 - astropy (development version): http://github.com/astropy/astropy
 - PySPICE: http://github.com/rca/PySPICE
 - Latest ERFA: http://github.com/liberfa/erfa-fetch  See below to compile.
 - ERFA python: http://github.com/nirinA/erfa_python

## Additional packages/libraries for developers
 - pylint
 - nosetests
 - sphinx
 - coverage
 - tempo
 - tempo2
 - tempo_utils

## Data files
PINT requires detailed ephemerides for the Solar System motion and for 
the Earth's rotation. Files containing this information can be 
automatically downloaded using datafiles/download_data.sh; the total volume 
is a few hundred megabytes. PINT also requires observatory clock correction 
data; these are currently read from the clock correction database of an 
installed copy of TEMPO (which it locates using the TEMPO environment 
variable; see below).

## Environment variables
In order to locate its data files, PINT allows an environment variable
named PINT be set to the location of the PINT distribution; in particular, it 
will look for its data files in $PINT/datafiles. If this is not set it will 
look in a directory near the directory containing the source code. PINT also 
requires access to  clock correction files, which it looks for in $TEMPO/clock.

## Development
As it stands, this package is chiefly useful to people who want to develop 
the code itself (though we hope this will change shortly!). At this 
early stage it is very important that the code remain clean, consistent, 
and reusable. We therefore have the following requests for new 
(and existing!) code:
 - All code should be documented with docstrings, and with 
    additional explanation where appropriate; please follow the AstroPy
    documentation [guidelines] for "finished" functions and at least
    include a one-sentence summary plus an explanation even if the
    function is still under development. Check Sphinx output to see if
    it makes sensse.
 - All code should be tested by unit tests
    - Brief doctests in the docstring should provide usage examples
    - More complex unit tests should go in the tests/ directory
    - Use the nose testing framework so tests are as independent as possible
    - When fixing a bug, please write a unit test that triggers the bug, 
       and keep it around to make sure the bug is not reintroduced.
 - All code should have as consistent a style as possible

In order to encourage us to stick to these rules, the script run_tests.sh 
is available in the root directory of the distribution (thanks to Michiel 
Brentjens). This generates the documentation, runs the test suite, and 
runs pylint. The goal is to have pylint return a clean bill of health, 
all the tests pass, and all the documentation clear and intelligible. It 
is probably a good idea to run this script (and clean up the results!) 
before every commit to git. While it is possible to disable particular warnings 
from pylint, please be very reluctant to do so, especially if they 
are stylistic warnings: if it thinks your method takes too many arguments, 
for example, but you can't figure out how to reduce their number, leave 
the warning as a reminder until you or somebody else figure out how to 
break up the method into simpler pieces.

[guidelines]: http://docs.astropy.org/en/stable/development/docguide.html

## NOTE:  Compiling ERFA from scratch
This can be a kind of a PITA, so until ERFA is part of astropy (which
we hope will happen).  You can use the following "script" to compile
and install it.

--------------------------
    export INSTDIR=/home/sransom
    git clone https://github.com/liberfa/erfa-fetch.git
    cd erfa-fetch
    python sofa_deriver.py
    cd erfa
    gcc -O2 -I. -c -fPIC *.c
    gcc -shared -o liberfa.so *.o -lm
    # On Mac, do the following instead
    # gcc -shared -Wl,-compatibility_version,2.0.0,-current_version,2.0.0 -o liberfa.so *.o -lm
    cp erfa*.h $INSTDIR/include/
    cp liberfa.so $INSTDIR/lib/liberfa.so.1.0.0
    ln -sf $INSTDIR/lib/liberfa.so.1.0.0 $INSTDIR/lib/liberfa.so
    ln -sf $INSTDIR/lib/liberfa.so.1.0.0 $INSTDIR/lib/liberfa.so.1
--------------------------
