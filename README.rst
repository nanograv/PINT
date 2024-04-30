.. image:: https://github.com/nanograv/PINT/blob/master/docs/logo/PINT_LOGO_128trans.png
   :alt: PINT Logo
   :align: right

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

.. image:: https://img.shields.io/badge/arXiv-2012.00074-red
   :target: https://arxiv.org/abs/2012.00074
   :alt: PINT Paper on arXiv

.. image:: https://img.shields.io/badge/ascl-1902.007-blue.svg?colorB=262255
   :target: https://www.ascl.net/1902.007
   :alt: PINT on ASCL

.. image:: https://img.shields.io/pypi/l/pint-pulsar
    :target: https://github.com/nanograv/PINT/blob/master/LICENSE.md
    :alt: License:BSD

.. image:: https://img.shields.io/badge/code_of_conduct-Contributor_Covenant-blue.svg
    :target: https://github.com/nanograv/PINT/blob/master/CODE_OF_CONDUCT.md
    :alt: Code of Conduct

PINT is not TEMPO3
------------------

PINT is a project to develop a pulsar timing solution based on
python and modern libraries. It is still in active development,
but it is in production use by the NANOGrav collaboration and
it has been demonstrated produce residuals from most "normal"
timing models that agree with Tempo and Tempo2 to within ~10
nanoseconds. It can be used within python scripts or notebooks,
and there are several command line tools that come with it.

The primary reasons PINT was developed are:

* To have a robust system to produce high-precision timing results that is
  completely independent of TEMPO and Tempo2

* To make a system that is easy to extend and modify due to a good design
  and the use of a modern programming language, techniques, and libraries.

IMPORTANT Notes!
----------------

PINT has a naming conflict with the `pint <https://pypi.org/project/Pint/>`_ units package available from PyPI (i.e. using pip) and conda.  
Do **NOT** ``pip install pint`` or ``conda install pint``!  See below!

PINT requires ``longdouble`` (80- or 128-bit floating point) arithmetic within ``numpy``, which is currently not supported natively on M1/M2/M3 Macs. 
However, you can use an x86 version of ``conda`` even on an M1/M2/M3 Mac (which will run under Rosetta emulation): 
see `instructions for using Apple Intel packages on Apple 
silicon <https://conda-forge.org/docs/user/tipsandtricks.html#installing-apple-intel-packages-on-apple-silicon>`_. 
It's possible to have `parallel versions of conda for x86 and 
ARM <https://towardsdatascience.com/python-conda-environments-for-both-arm64-and-x86-64-on-m1-apple-silicon-147b943ffa55>`_.


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

PINT is also available for Anaconda python under the conda-forge channel::

    $ conda install -c conda-forge pint-pulsar

The above two options install the latest *released* version. If you want access to the latest development version, 
the source code, example notebooks, and tests, you can install from source, by 
cloning the source repository from GitHub, then install
it, ensuring that all dependencies needed to run PINT are available::

    $ git clone https://github.com/nanograv/PINT.git
    $ cd PINT
    $ pip install .

Complete installation instructions are available on `readthedocs <https://nanograv-pint.readthedocs.io/en/latest/installation.html>`_.


Using
-----

See the online documentation_. Specifically:

* `Tutorials <https://nanograv-pint.readthedocs.io/en/latest/tutorials.html>`_
* `API reference <https://nanograv-pint.readthedocs.io/en/latest/reference.html>`_
* `How-Tos for common tasks <https://github.com/nanograv/PINT/wiki/How-To>`_

Are you a NANOGrav member?  Then join the #pint channel in the NANOGrav slack.
  
If you have tasks that aren't covered in the material above, you can
email pint@nanograv.org or one of the people below:

* Scott Ransom (sransom@nrao.edu)
* Paul Ray (paul.s.ray3.civ@us.navy.mil)
* David Kaplan (kaplan@uwm.edu)  

Want to do something new? Submit a github `issue <https://github.com/nanograv/PINT/issues>`_.
  
.. _documentation:   http://nanograv-pint.readthedocs.io/en/latest/

And for more details, please read and cite(!) the PINT paper_.

.. _paper:   https://ui.adsabs.harvard.edu/abs/2021ApJ...911...45L/abstract

Articles that cite the PINT paper can be found in an ADS `Library <https://ui.adsabs.harvard.edu/search/q=citations(bibcode%3A2021ApJ...911...45L)&sort=date%20desc%2C%20bibcode%20desc&p_=0>`_.
A list of software packages that use PINT can be found `here <https://nanograv-pint.readthedocs.io/en/latest/dependent-packages.html>`_.
