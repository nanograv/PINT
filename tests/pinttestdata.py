# Import this to get the location of the datafiles for tests.  This file
# must be kept in the appropriate location relative to the test data dir
# for this to work.

from pathlib import Path

# Location of this file and the test data scripts
# testdir = os.path.dirname(os.path.abspath(__file__))
testdir = Path(__file__).resolve().parent
datadir = testdir / "datafile"
