.. _`basic-installation`:

Basic installation
==================

**IMPORTANT Note: **
PINT has a naming conflict with the `pint <https://pypi.org/project/Pint/>`_ units package available from PyPI (i.e. using pip) and conda.  
Do **NOT** ``pip install pint`` or ``conda install pint``!  See below!

PINT is now available via PyPI as the package `pint-pulsar <https://pypi.org/project/pint-pulsar>`_, so it is now simple to install via pip.

For most users, who don't want to develop the PINT code, installation should just be a matter of::

   $ pip install pint-pulsar

By default this will install in your system site-packages.  Depending on your system and preferences, you may want to append ``--user`` 
to install it for just yourself (e.g. if you don't have permission to write in the system site-packages), or you may want to create a 
virtualenv to work on PINT (using a virtualenv is highly recommended by the PINT developers).

If you want access to the source code, example notebooks, and tests, you can install from source, by 
cloning the source repository from GitHub, then install it, ensuring that all dependencies needed to run PINT are available::

    $ git checkout https://github.com/nanograv/PINT.git
    $ cd PINT
    $ pip install .

If this fails, or for more explicit installation and troubleshooting instructions see :ref:`Installation`.
