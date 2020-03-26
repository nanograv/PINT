.. _`basic-installation`:

Basic installation
==================

PINT is now available via PyPI as the package `pint-pulsar <https://pypi.org/project/pint-pulsar>`_, so it is now simple to install via pip.

For most users, who don't want to develop the PINT code, installation should just be a matter of::

   $ pip install pint-pulsar

By default this will install in your system site-packages.  Depending on your system and preferences, you may want to append ``--user`` 
to install it for just yourself (e.g. if you don't have permission to write in the system site-packages), or you may want to create a 
virtualenv to work on PINT (using a virtualenv is highly recommended by the PINT developers).

If this fails, or for more explicit installation and troubleshooting instructions see :ref:`Installation`.
