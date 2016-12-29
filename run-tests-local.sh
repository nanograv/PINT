#!/bin/bash

MODULE=pint

# Before running this, make sure the module has been build in place with this command:
# python setup.py build_ext --inplace

# This ensures that the local directory is in PYTHONPATH so that the module being tested is the local one, not the installed version
# This is useful for testing before installing, or for testing code changes without needing to install after each edit.
PYTHONPATH="`pwd`:$PYTHONPATH"

NOSETESTS=`which nosetests 2> /dev/null`

if [[ ! -f "$NOSETESTS" ]] ; then
    NOSETESTS=`which nosetests2`
fi

echo ''
echo "  *** Testing module $MODULE at" `date` "***"
echo ''

if [[ ! -f "$NOSETESTS" ]] ; then
    echo 'Cannot find nosetests or nosetests2';
else
   echo "Using $NOSETESTS"

   $NOSETESTS \
              --with-coverage \
              --cover-package="$MODULE" \
              --cover-tests \
              --cover-html \
              --cover-html-dir=coverage \
              --cover-erase 

# Eventually, we should re-enable --with-doctest, once the doctests have been written correctly (issue #198)

fi

echo ''
