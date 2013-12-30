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
 - Use of defined constants
 - Unittests (hopefully.  Volunteers for this would be appreciated.)
