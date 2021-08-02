.. _`data-files`:

Data files
==================

The data files (``par`` and ``tim``) associated with the tutorials and
other examples can be located via:

::

   import pint
   filename = pint.datafile(<filename>)


For example, the file ``NGC6440E.par`` from the
:ref:`examples/time_a_pulsar.ipynb` notebook can be found via:

::

   import pint
   filename = pint.datafile("NGC6440E.par")

Also see :func:`pint.datadir` and :func:`pint.datafile`.  
