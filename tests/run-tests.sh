#!/bin/bash

# This tests the installed version, so make sure PINT has been installed with a command like:
# python setup.py install --user
# Then, cd tests and run this script (./run-tests.sh)

MODULE=pint



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

   $NOSETESTS 
# Eventually, we should re-enable --with-doctest, once the doctests have been written correctly (issue #198)

fi

echo ''
