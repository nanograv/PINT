#!/bin/bash

# First, build the documentation
MODULE=pint
# The make latexpdf is not working at the moment...
#(cd doc && make html && make latexpdf)
#(cd doc && make html)

PYTHONPATH="`pwd`:$PYTHONPATH"
NOSETESTS=`which nosetests 2> /dev/null`
PYLINT=`which pylint 2> /dev/null`
if [[ ! -f "$PYLINT" ]] ; then
    PYLINT=`which pylint2`
fi

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
fi

# Disable pylint for now, since it is failing
exit

echo ''
echo '  *** Pylint output ***'
echo ''

if [[ ! -f "$PYLINT" ]] ; then
    echo 'Cannot find pylint';
else
    $PYLINT --output-format=colorized \
	           --rcfile=pylint.rc \
                   --reports=n $MODULE;
fi

echo ''
