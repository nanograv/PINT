.. highlight:: shell

============
Installation
============

Prerequisites
-------------

* Python 2.7 or 3.5 or later (Note that Python 3 support is currently not
  complete, but is a design goal)

* The current list of required python packages is in

  - requirements.txt_
  - requirements_dev.txt_

.. _requirements.txt: https://github.com/nanograv/PINT/blob/master/requirements.txt
.. _requirements_dev.txt: https://github.com/nanograv/PINT/blob/master/requirements_dev.txt

* The simplest way to install the prerequisites, if you are in virtualenv or
  want to install them in the system python is to use pip [1]_::

    pip install -r requirements.txt
    pip install -r requirements_dev.txt

  If you want to install them in your local user site-packages, rather than the
  system python (perhaps because you don't have sudo privileges),
  append ``--user`` to those command lines.

  Some of those packages may have been already installed, for example by MacPorts.
  For MacPorts users, this command will get many of the requirements::

    port install py27-numpy py27-scipy py27-astropy py27-nose py27-cython py27-emcee

  You probably want to avoid having multiple versions of packages installed,
  to reduce confusion.  Working in a virtualenv can be helpful.

* `TEMPO`_ is not required, but if you have it installed PINT can find clock
  correction files in ``$TEMPO/clock``

.. _TEMPO: http://tempo.sourceforge.net

* `Tempo2`_ is not required, but if you have it installed PINT can find clock
  correction files in ``$TEMPO2/clock``

.. _Tempo2: https://bitbucket.org/psrsoft/tempo2

.. [1] If you don't have `pip`_ installed, this `Python installation guide`_ can guide
   you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Installing from Source
----------------------

The sources for pint can be downloaded from the `Github repo`_.

First, clone the public repository:

.. code-block:: console

    $ git clone https://github.com/nanograv/PINT.git

Normally, you want to use the ``master`` branch, since PINT is in active
development and this makes it easy to get the latest fixes with a simple
``git pull``.

Once you have a copy of the source, if you are using a virtualenv, or want
to install PINT in your system site-packages (may require sudo),
you can install it with:

.. code-block:: console

  $ python setup.py install

Or, synonymously:

.. code-block:: console

  $ make install

Or, to install it in your local user site-packages

.. code-block:: console

  $ python setup.py install --user

Finally, if you want to be able to edit or pull changes and have them
take effect without having to re-run setup.py, you can install using links
to the source itself, like this (again append ``--user`` if you want
to install in your per-user site-packages location). *This is how most PINT
developers work*:

.. code-block:: console

  $ python setup.py develop

.. _Github repo: https://github.com/nanograv/pint
.. _tarball: https://github.com/nanograv/pint/tarball/master

Running tests
-------------

To verify that your installed pint is functional, you can (and should) run
the test suite.  This can be done using::

  make test

or::

  python setup.py nosetests

Build the documentation
-----------------------

This is not normally needed, since the documentation is available online_
but you can build your own copy for offline use::

  make docs

At completion, a browser will open with the documentaion.

.. _online: http://nanograv-pint.readthedocs.io/en/latest/

Data files
----------

PINT requires detailed ephemerides for the Solar System motion and for the
Earth's rotation. Many of these files are downloaded automatically by
astropy. Others are distributed with PINT in the ``pint/datafiles`` directory
or are automatically downloaded by setup.py; the total volume is a few hundred
megabytes. On installation, the data files are copied into the install
directory, so you end up with two copies (unless you install in develop mode).

PINT also requires observatory clock correction data. The PINT distribution
includes a set in the datafiles directory, but clock corrections can also be
read from TEMPO or Tempo2 clock directories if they are installed.

Other Makefile features
-----------------------

The makefile can do several other useful things including cleaning up cruft,
and building tar distributions.

.. code-block:: console

  $ make help
  clean                remove all build, test, coverage and Python artifacts
  clean-build          remove build artifacts
  clean-pyc            remove Python file artifacts
  clean-test           remove test and coverage artifacts
  lint                 check style with flake8
  test                 run tests quickly with the default Python
  coverage             check code coverage quickly with the default Python
  docs                 generate Sphinx HTML documentation, including API docs
  servedocs            compile the docs watching for changes
  dist                 builds source and wheel package
  install              install the package to the active Python's site-packages
