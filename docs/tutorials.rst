.. _`Tutorials`:

Tutorials
=========

These are step-by-step guides you can follow to show you what PINT can
do. Quick common tasks are explained on the `PINT Wiki
<https://github.com/nanograv/PINT/wiki>`_. If you want some
explanations of why things work this way, see the :ref:`Explanation`
section. If you want details on a particular function that comes up,
see the :ref:`Reference` section. More complicated tasks are discussed
in the :ref:`How-tos`.

Data Files
----------

The data files (``par`` and ``tim``) associated with the tutorials and
other examples can be located via :func:`pint.config.examplefile`
(available via the :mod:`pint.config` module):

::

   import pint.config
   fullfilename = pint.config.examplefile(filename)


For example, the file ``NGC6440E.par`` from the
`Time a Pulsar`_ notebook can be found via:

::

   import pint
   fullfilename = pint.config.examplefile("NGC6440E.par")

Examples
--------

These tutorials examples are
in the form of `Jupyter <https://jupyter.org>`_ notebooks, downloadable from a link at the top
of each page. (Also available in the same place is a plain-python
script version, in case this is more convenient.) You should be able
to download these files and run them from anywhere convenient
(provided ``PINT`` is installed). Finally, there are additional
notebooks you can download from the `PINT Wiki
<https://github.com/nanograv/PINT/wiki>`_ or the 
`GitHub examples <https://github.com/nanograv/PINT/tree/master/docs/examples>`_ directory: these 
are not included in the default build because they take too long, but you can download and run them yourself.

.. The list below is just those that are built.  Other notebooks are excluded from the build
.. Using the exclude_patterns list in conf.py

.. toctree::

   basic-installation
   examples/time_a_pulsar.ipynb
   examples/PINT_walkthrough.ipynb
   examples/fit_NGC6440E.ipynb
   examples/covariance.ipynb
   examples/check_clock_corrections.ipynb
   examples/understanding_timing_models.ipynb
   examples/understanding_parameters.ipynb
   examples/build_model_from_scratch.ipynb
   examples/How_to_build_a_timing_model_component.ipynb
   examples/understanding_fitters.ipynb
   examples/noise-fitting-example.ipynb
   examples/rednoise-fit-example.ipynb
   examples/WorkingWithFlags.ipynb
   examples/Wideband_TOA_walkthrough.ipynb
   examples/Simulate_and_make_MassMass.ipynb
   examples/check_phase_connection.ipynb
   examples/PINT_observatories.ipynb
   examples/solar_wind.ipynb
   examples/MCMC_walkthrough.ipynb
   examples/bayesian-example-NGC6440E.py
   examples/bayesian-wideband-example.py
   examples/simulation_example.py
   examples-rendered/paper_validation_example.ipynb

.. _`Time a Pulsar`: examples/time_a_pulsar.html
