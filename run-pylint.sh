#!/bin/bash

MODULE=pint

# Should we add the local directory to our PYTHONPATH for pylint?  I don't think so.
# PYTHONPATH="`pwd`:$PYTHONPATH"

PYLINT=`which pylint 2> /dev/null`
if [[ ! -f "$PYLINT" ]] ; then
    PYLINT=`which pylint2`
fi

if [[ ! -f "$PYLINT" ]] ; then
    echo 'Cannot find pylint';
else
    $PYLINT --output-format=colorized \
	           --rcfile=pylint.rc \
                   --reports=n $MODULE;
fi

echo ''
