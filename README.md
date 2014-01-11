PINT
====

## PINT is not TEMPO3

PINT is a project to develop a pulsar timing solution which is
completely independent of TEMPO and TEMPO2.  It is in _extremely_
early stages of development.

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
 - mpmath
 - astropy (development version): http://github.com/astropy/astropy
 - PySPICE: http://github.com/rca/PySPICE
 - ERFA python: http://github.com/nirinA/erfa_python

## Additional packages/libraries for developers

 - pylint
 - nosetests
 - sphinx
 - coverage

## Development

As it stands, this package is chiefly useful to people who want to develop 
the code itself (though we hope this will change shortly!). At this 
early stage it is very important that the code remain clean, consistent, 
and reusable. We therefore have the following requests for new 
(and existing!) code:

 - All code should be documented with docstrings, and with 
    additional explanation where appropriate
 - All code should be tested by unit tests
    - Brief doctests in the docstring should provide usage examples
    - More complex unit tests should go in the tests/ directory
    - When fixing a bug, please write a unit test that triggers the bug, 
       and keep it around to make sure the bug is not reintroduced.
 - All code should have as consistent a style as possible

In order to encourage us to stick to these rules, the script run_tests.sh 
is available in the root directory of the distribution (thanks to Michiel 
Brentjens). This generates the documentation, runs the test suite, and 
runs pylint. The goal is to have pylint return a clean bill of health, 
all the tests pass, and all the documentation clear and intelligible. It 
is probably a good idea to run this script (and clean up the results!) 
on every test run. While it is possible to disable particular warnings 
from pylint, please be very reluctant to do so, especially if they 
are stylistic warnings: if it thinks your method takes too many arguments, 
for example, but you can't figure out how to reduce their number, leave 
the warning as a reminder until you or somebody else figure out how to 
break up the method into simpler pieces.

## Some design philosophy

 - Emphasis on modularity
 - Well defined interface
 - Mostly written in python
 - Heavy use of well-debugged libraries
   - JPL SPICE
   - astropy
   - optimization libraries
 - GUI will not be required
 - Strong isolation of parts needing extended precision
 - Full use of docstrings
 - Self-describing objects
 - Effective use of the Python standard library
    (i.e. text reading, parsing, simple DB, binary search, etc)
 - Backwards compatible with TEMPO
 - Code independence from TEMPO and TEMPO2
 - Use of defined constants
 - Unittests

