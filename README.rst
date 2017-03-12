PINT
====

.. image:: https://travis-ci.org/nanograv/PINT.svg?branch=master
    :target: https://travis-ci.org/nanograv/PINT

.. image:: https://readthedocs.org/projects/nanograv-pint/badge/?version=latest
    :target: http://nanograv-pint.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/nanograv/PINT/badge.svg?branch=master
    :target: https://coveralls.io/github/nanograv/PINT?branch=master

.. image:: https://landscape.io/github/jellis18/PINT/master/landscape.svg?style=flat
    :target: https://landscape.io/github/jellis18/PINT/master
    :alt: Code Health

PINT is not TEMPO3
------------------

PINT is a project to develop a new pulsar timing solution based on
python and modern libraries. It is in the very early stages of
development, but it can already produce residuals from most "normal"
timing models that agree with Tempo and Tempo2 to within ~10
nanoseconds.

The primary reasons we are developing PINT are: - To have a robust
system to check high-precision timing results that is completely
independent of TEMPO and TEMPO2 - To make a system that is easy to
extend and modify due to a good design and the use of a modern
programming language, techniques, and libraries

Most users will want to install PINT as follows:

::

    python setup.py install --user

To build PINT so that it can be run from the local directory (without
installing), such as for developers or to run the tests, use:

::

    python setup.py build_ext --inplace

For more information on installing and developing PINT, see the
`Wiki <https://github.com/nanograv/PINT/wiki>`__ and the documentation
in the doc subdirectory in the distribution. Also, follow along with the
examples in the examples subdirectory, particularly the IPython notebook
``examples/PINT_walkthrough.ipynb``.
