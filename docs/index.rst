PINT Is Not TEMPO3
==================

.. image:: https://api.travis-ci.org/nanograv/PINT.svg?branch=master
    :target: https://travis-ci.org/nanograv/PINT
    :alt: Build Status

.. image:: https://readthedocs.org/projects/nanograv-pint/badge/?version=latest
    :target: https://nanograv-pint.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Build Status

.. image:: https://coveralls.io/repos/github/nanograv/PINT/badge.svg?branch=master
    :target: https://coveralls.io/github/nanograv/PINT?branch=master
    :alt: Code Coverage

.. image:: https://landscape.io/github/nanograv/PINT/master/landscape.svg?style=flat
    :target: https://landscape.io/github/nanograv/PINT/master
    :alt: Code Health

.. image:: https://badges.gitter.im/nanograv-PINT/community.svg
    :target: https://gitter.im/nanograv-PINT/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Gitter chat

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

.. toctree::
   :maxdepth: 3

   installation
   usage
   command-line
   structure
   api/pint
   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
