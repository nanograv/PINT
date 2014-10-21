#! /usr/bin/env python
import sys
import pint.models as tm

try:
    parfile
except NameError:
    parfile = None

if parfile==None:
    try:
        parfile = sys.argv[1]
    except IndexError:
        print "usage: test_parfile.py (parfile)"
        sys.exit(0)

TestModel = tm.generate_timing_model("TestModel",(tm.Astrometry,tm.Spindown))
m = TestModel()

print "model.param_help():"
m.param_help()
print

print "calling model.read_parfile():"
m.read_parfile(parfile)
print

print "print model:"
print m
print

print "model.as_parfile():"
print m.as_parfile()
print
