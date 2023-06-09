.. highlight:: shell
.. _`Installation`:

===================
How to Install PINT
===================

There are two kinds of PINT installation you might be interested in. The first
is a simple PINT installation for someone who just wants to use PINT. The
second is an installation for someone who wants to be able to run the tests and
develop PINT code. The latter naturally requires more other python packages and
is more complicated (but not too much).

Prerequisites
-------------

PINT requires Python 3.8+ [1]_

Your Python must have the package installation tool pip_ installed.  Also make sure your ``setuptools`` are up to date (e.g. ``pip install -U setuptools``).

We highly recommend using a Conda_/`Anaconda <https://www.anaconda.com/products/individual>`_ environment or the package isolation tool virtualenv_.


IMPORTANT Notes!
---------------

Naming conflict
'''''''''''''''

PINT has a naming conflict with the `pint <https://pypi.org/project/Pint/>`_ units package available from PyPI (i.e. using pip) and conda.  
Do **NOT** ``pip install pint`` or ``conda install pint``!  See :ref:`Basic Install via pip` or :ref:`Install with Anaconda`.

Apple M1/M2 processors
''''''''''''''''''''''

PINT requires ``longdouble`` (80- or 128-bit floating point) arithmetic within ``numpy``, which is currently not supported natively on M1/M2 Macs. 
However, you can use an x86 version of ``conda`` even on an M1/M2 Mac: see `instructions for using Apple Intel packages on Apple silicon <https://conda-forge.org/docs/user/tipsandtricks.html#installing-apple-intel-packages-on-apple-silicon>`_. 
It's possible to have `parallel versions of conda for x86 and ARM <https://towardsdatascience.com/python-conda-environments-for-both-arm64-and-x86-64-on-m1-apple-silicon-147b943ffa55>`_.


Basic Install via pip
---------------------

PINT is now available via PyPI as the package `pint-pulsar <https://pypi.org/project/pint-pulsar>`_, so it is now simple to install via pip.
This will get you the latest *released* version of PINT.

For most users, who don't want to develop the PINT code, installation should just be a matter of::

   $ pip install pint-pulsar

By default this will install in your system site-packages.  Depending on your system and preferences, you may want to append ``--user`` 
to install it for just yourself (e.g. if you don't have permission to write in the system site-packages), or you may want to create a 
virtualenv to work on PINT (using a virtualenv is highly recommended by the PINT developers).  In that case, you just activate your 
virtualenv before running the ``pip`` command above.


.. _anaconda:

Install with Anaconda
---------------------

If you use `Anaconda <https://www.anaconda.com/products/individual>`_ environments to manage your python packages, 
PINT is also available for Anaconda python under the `conda-forge <https://conda-forge.org>`_ channel::

    $ conda install -c conda-forge pint-pulsar

Install from Source
-------------------

If you want access to the latest development version of PINT, or want to be able to make any edits to the code, you can install
from source by cloning the git repository.

If your python setup is "nice", you should be able to install as easily as::

   $ git clone https://github.com/nanograv/PINT.git
   $ cd PINT
   $ mkvirtualenv -p `which python3` pint
   (pint) $ pip install -e .
   (pint) $ python
   >>> import pint

Note that you can use your own method to activate your virtualenv if you don't have virtualenvwrapper_ installed.
This *should* install PINT along with any python packages it needs to run. (If
you want to run the test suite or work on PINT code, see below.)
Note that the ``-e`` installs PINT in "editable" or "develop" mode.  This means that the source code is what is actually being run,
rather than making a copy in a site-packages directory. Thus, if you edit any .py file, or do a ``git pull`` to update the code
this will take effect **immediately** rather than having to run ``pip install`` again.  This is a choice, but is the way 
most developers work.

Unfortunately there are a number of reasons the install can go wrong. Most have to do
with not having a "nice" python environment. See the next section for some tips.

Potential Install Issues
------------------------

Old setuptools (``egg-info`` error message)
'''''''''''''''''''''''''''''''''''''''''''

PINT's ``setup.cfg`` is written in a declarative style that does not work with
older versions of ``setuptools``. The lack of a sufficiently recent version of
``setuptools`` is often signalled by the otherwise impenetrable error message
``error: 'egg_base' must be a directory name (got src)``. You can upgrade with
``pip``::

   $ pip install -U pip setuptools

If this does not help, check your versions of installed things::

   $ pip list

You should be able to upgrade to ``setuptools`` version at least ``0.41``. If
running ``pip`` does not change the version that appears on this list, or if
your version changes but the problem persists, you may have a problem with your
python setup; read on.

Bad ``PYTHONPATH``
''''''''''''''''''

The virtualenv mechanism uses environment variables to create an isolated
python environment into which you can install and upgrade packages without
affecting or being affected by anything in any other environment. Unfortunately
it is possible to defeat this by setting the ``PYTHONPATH`` environment
variable. Double unfortunately, setting the ``PYTHONPATH`` environment used to
be the Right Way to use python things that weren't part of your operating
system. So many of us have ``PYTHONPATH`` set in our shells. You can check this::

   $ printenv PYTHONPATH

If you see any output, chances are that's causing problems with your
virtualenvs. You probably need to go look in your ``.bashrc`` and/or
``.bash_profile`` to see where that variable is being set and remove it. Yes,
it is very annoying that you have to do this.

Previous use of ``pip install --user``
''''''''''''''''''''''''''''''''''''''

Similarly, it used to be recommended to install packages locally as your user
by running ``pip install --user thing``. Unfortunately this causes something of
the same problem as having a ``PYTHONPATH`` set, where packages installed
outside your virtualenv can obscure the ones you have inside, producing bizarre
error messages. Record your current packages with ``pip freeze``, then try,
outside a virtualenv, doing ``pip list`` with various options, and ``pip uninstall``; you shouldn't be able to uninstall anything system-wise (do not
use ``sudo``!) and you shouldn't be able to uninstall anything in an inactive
virtualenv. So once you've blown away all those packages, you should be able to
work in clean virtualenvs. If you saved the output of ``pip freeze`` above, you
should be able to use it to create a virtualenv with all the same packages you
used to have in your user directory.

Bad ``conda`` setup
'''''''''''''''''''

Conda_ is a tool that attempts to create isolated environments, like a
combination of virtualenv and ``pip``. It should make installing scientific
software with lots of dependencies easy and reliable, and you should just be
able to set up an appropriate ``conda`` environment and use the basic install
instructions above. But it may not work.

Specifically, for some reason the python 3 version of ``conda`` does not
provide the ``gdbm`` module, which ``astropy`` needs to work on Linux. Good
luck.

.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _virtualenvwrapper: https://virtualenvwrapper.readthedocs.io/en/latest/
.. _Conda: https://docs.conda.io/en/latest/
.. _Anaconda: https://www.anaconda.com

Installing PINT for Developers
------------------------------

You will need to be able to carry out a basic install of PINT as above.
You very likely want to install in a virtualenv_ and using the develop mode ``pip -e``. 
Then you will need to install the additional development dependencies::

   $ pip install -Ur requirements_dev.txt


PINT development (building the documentation) requires pandoc_, which isn't a
python package and therefore needs to be installed in some way appropriate for
your operating system. On Linux you may be able to just run::

   $ apt install pandoc

On a Mac using MacPorts this would be::

   $ sudo port install pandoc

Otherwise, there are several ways to `install pandoc`_

For further development instructions see :ref:`Developing PINT`

.. _pip: https://pip.pypa.io/en/stable/
.. _pandoc: https://pandoc.org/
.. _`install pandoc`: https://pandoc.org/installing.html

.. rubric:: Footnotes
.. [1] Python 2.7 and 3.5+ are supported for PINT 0.7.x and earlier.
