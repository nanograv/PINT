

PINT
====

.. image:: https://github.com/nanograv/pint/workflows/CI%20Tests/badge.svg
   :target: https://github.com/nanograv/pint/actions
   :alt: Actions Status

.. image:: https://codecov.io/gh/nanograv/PINT/branch/master/graph/badge.svg?token=xIOFqcKKrP
   :target: https://codecov.io/gh/nanograv/PINT
   :alt: Coverage
   
.. image:: https://readthedocs.org/projects/nanograv-pint/badge/?version=latest
    :target: https://nanograv-pint.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/pint-pulsar
    :target: https://github.com/nanograv/PINT/blob/master/LICENSE.md
    :alt: License:BSD

.. image:: https://img.shields.io/badge/code_of_conduct-Contributor_Covenant-blue.svg
    :target: https://github.com/nanograv/PINT/blob/master/CODE_OF_CONDUCT.md
    :alt: Code of Conduct

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

IMPORTANT Note!
---------------

PINT has a naming conflict with the `pint <https://pypi.org/project/Pint/>`_ units package available from PyPI (i.e. using pip) and conda.  
Do **NOT** ``pip install pint`` or ``conda install pint``!  See below!

Installing
----------

.. image:: https://anaconda.org/conda-forge/pint-pulsar/badges/version.svg
   :target: https://anaconda.org/conda-forge/pint-pulsar
   :alt: Conda Version

.. image:: https://img.shields.io/pypi/v/pint-pulsar.svg
   :target: https://pypi.python.org/pypi/pint-pulsar
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/pint-pulsar.svg
   :target: https://pypi.python.org/pypi/pint-pulsar
   :alt: PyVersions

PINT is now available via PyPI as the package `pint-pulsar <https://pypi.org/project/pint-pulsar>`_, so it is now simple to install via pip.
For most users, who don't want to develop the PINT code, installation should just be a matter of::

   $ pip install pint-pulsar

By default this will install in your system site-packages.  Depending on your system and preferences, you may want to append ``--user`` 
to install it for just yourself (e.g. if you don't have permission to write in the system site-packages), or you may want to create a 
virtualenv to work on PINT (using a virtualenv is highly recommended by the PINT developers).

PINT is also available for Anaconda python under the conda-forge channel:

    $ conda install -c conda-forge pint-pulsar

The above two options install the latest *released* version. If you want access to the latest development version, 
the source code, example notebooks, and tests, you can install from source, by 
cloning the source repository from GitHub, then install
it, ensuring that all dependencies needed to run PINT are available::

    $ git checkout https://github.com/nanograv/PINT.git
    $ cd PINT
    $ pip install .

Complete installation instructions are availble here_.

.. _here: https://nanograv-pint.readthedocs.io/en/latest/installation.html


Using
-----

See the online documentation_.

.. _documentation:   http://nanograv-pint.readthedocs.io/en/latest/
