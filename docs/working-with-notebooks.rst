.. highlight:: shell
.. _working-with-notebooks:

How to Work With Example Notebooks
==================================

Converting Notebooks to Plain Python
------------------------------------

PINT's documentation includes a certain number of Jupyter notebooks. When the
online documentation is built these are executed and the results are included
in the documentation. This is a nice way to set up python tutorials, but there
are a few wrinkles in the way these are integrated into version control. In
particular, storing a Jupyter notebook in ``git`` causes headaches. So we store
a sort of "distilled" python version.

If you create a new notebook, tell `jupytext`_ that you want to keep a plain python copy::

   $ jupytext --set-formats ipynb,py:percent docs/examples/my_notebook.ipynb

This will generate a ``.py`` version that also contains the
information from non-Python cells as comments. The format is
understandable to `Spyder`_ as well, which can recognize and execute
code cells.

Where to Put the Data
---------------------

Put any data files in ``src/pint/data/examples``, and include a note about the
data (where you got it from) in ``src/pint/data/examples/README.md``.  This
will ensure that the data get put in the proper place on installing
``pint``.  To refer to the files use :func:`pint.config.examplefile`:

::

   import pint.config
   fullfilename = pint.config.examplefile(filename)


Compiling and Synchronizing the Notebooks
-----------------------------------------
   
If you check something out of ``git``, or switch branches, or want to make sure you have current versions of all the notebooks, run::

   $ make notebooks

or::

   $ tox -e notebooks

This will both synchronize the working Jupyter notebooks with the python
version and also execute the notebooks. So if there is an error in a notebook,
this may stop part-way through. If this happens, try the simpler::

   $ jupytext --sync

This will synchronize the notebook contents without trying to execute them.

Using the Notebooks
-------------------

Whichever of those you ran, now you can use `Jupyter Lab`_ to work
with the notebooks as per normal. You may see a strange message about
rebuilding and jupytext; just hit okay. The jupytext code should ensure that as
you manipulate the notebook, the plain python is kept in sync (it contains only the
inputs, not the outputs).


Checking it Back Into GitHub
----------------------------

When you are ready to check things in to ``git``, just run::

   $ make notebooks

or::

   $ tox -e notebooks

This will synchronize and execute all the notebooks; if an error occurs, you
have a problem in your notebooks and you probably shouldn't check them in to
``git``. If everything is fine, and especially if you have added a new
notebook, ensure that it gets checked in to ``git`` with::

   $ git add docs/examples/my_notebook.py

That is, check the python versions in to ``git``, **not** the ``.ipynb``
versions.

Adding it to the Documentation
------------------------------

Now add the new notebook to the documentation somewhere — after all, that's why
you wrote it, right? Do this by putting it in a "toctree", that is, add its
name (without the ``.py`` extension) to a list of sub-documents like this one
from our :ref:`Tutorials` section::

   .. toctree::

      basic-installation
      examples/PINT_walkthrough
      examples/Example of parameter usage
      examples/TimingModel_composition
      examples/Timing_model_update_example

Finally, since you changed the documentation, rebuild it::

   $ make docs

or::

   $ tox -e docs
   $ firefox docs/_build/index.html

This will rebuild the documentation (and in the case of ``make`` open it in a
browser window); if you have any bad formatting or links that point to nowhere
it will stop with an error.

.. _Spyder: https://www.spyder-ide.org/
.. _`Jupyter Lab`: https://jupyterlab.readthedocs.io/en/stable/
.. _Jupytext: https://jupytext.readthedocs.io/en/latest/install.html
