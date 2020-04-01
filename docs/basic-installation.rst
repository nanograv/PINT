.. _`basic-installation`:

Basic installation
==================

If your python installation is "nice", installation should just be a matter of::

   $ git clone https://github.com/nanograv/PINT.git
   $ cd PINT
   $ mkvirtualenv -p `which python3` pint
   (pint) $ pip install -e .
   (pint) $ python
   >>> import pint

If this doesn't work, we apologize, but it's probably not PINT's fault. We have
a more explicit installation and troubleshooting guide at :ref:`Installation`.
