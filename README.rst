PINT
====

.. image:: https://travis-ci.org/nanograv/PINT.svg?branch=master
    :target: https://travis-ci.org/nanograv/PINT

.. image:: https://readthedocs.org/projects/nanograv-pint/badge/?version=latest
    :target: http://nanograv-pint.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/nanograv/PINT/badge.svg?branch=master
    :target: https://coveralls.io/github/nanograv/PINT?branch=master

.. image:: https://landscape.io/github/nanograv/PINT/master/landscape.svg?style=flat
    :target: https://landscape.io/github/nanograv/PINT/master
    :alt: Code Health

PINT is not TEMPO3
------------------

PINT is a project to develop a new pulsar timing solution based on
python and modern libraries. It is still in active development,
but it can already produce residuals from most "normal"
timing models that agree with Tempo and Tempo2 to within ~10
nanoseconds. It can be used within python scripts or notebooks,
and there are several command line tools that come with it.

The primary reasons we are developing PINT are:

* To have a robust system to check high-precision timing results that is
  completely independent of TEMPO and Tempo2

* To make a system that is easy to extend and modify due to a good design
  and the use of a modern programming language, techniques, and libraries.

Read the PINT documentation here_.

.. _here:   http://nanograv-pint.readthedocs.io/en/latest/

Installing
----------

Currently PINT is not available via PyPI or Conda. To install it you must
obtain the source from GitHub (for example by cloning it), then ensure that
all PINT's dependencies are available:

    $ pip install -r requirements.txt

Then you can just install PINT:

    $ python setup.py install

When you run setup.py for the first time, or when it is run by pip, PINT will
attempt to download a number of data files from a server at JPL. These will be
stored in pint/datafiles in the source tree and installed alongside PINT. It
will not be necessary to re-download them into this source tree unless PINT
requests versions with different contents.

If you want to install PINT on a machine without internet access, the file
datafile_urls.txt contains URLs you can download the data files from on
another machine; you can then copy the files into pint/datafiles. Or, since
you presumably had to download the source code on a machine with Internet
access, on that machine you can run python setup.py with no arguments; this
will download the files but make no attempt to install anything else. Be aware
that AstroPy also requires data files from the Internet, for example the
International Earth Rotation Service bulletins, which are regularly updated.
So AstroPy may not work without access to the Internet, and PINT requires
AstroPy.

Using
-----

See the online documentation_.

.. _here:   http://nanograv-pint.readthedocs.io/en/latest/

