# pinttestdata.py

# import this to get the location of the datafiles for tests.  This file
# must be kept in the appropriate location relative to the test data
# dir for this to work.

import os

# Location of this file and the test data scripts
testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "datafile")
