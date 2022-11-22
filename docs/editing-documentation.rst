.. highlight:: shell

How to Edit PINT's Documentation
================================

PINT's documentation is in the Sphinx_ format, which is based on
reStructuredText_. This is a plain-text-based documentation format, which means
that the documentation is largely editable as plain text with various arcane
bits of punctuation strewn throughout to keep you on your toes. This
documentation can be converted automatically into HTML, LaTeX, and PDF, but in
practice by far the most useful form is that produced by and served up on the
readthedocs_ servers. You can find that rendered version here_.

PINT's documentation is created from three different inputs:

   - reStructuredText files (``.rst``) under ``docs/`` (but not ``docs/api/``;
     see below),
   - docstrings in the code; these follow the ``numpy`` `docstring guidelines`
     and thus are not pure reStructuredText, and
   - Jupyter notebooks under ``docs/examples``; see :ref:`working-with-notebooks`.

It should be fairly clear where to look for any given piece of documentation's
code, but if there is any question, start from ``docs/index.rst``. Everything
is in a tree depending from here.  If you add a function or a class to
a file, make sure that the new function/class is included in the
``__all__`` list, so that it will be included in the documentation.

To build a local copy of the documentation, run::

   $ make docs

or::

   $ tox -e docs
   $ firefox docs/_build/index.html

These both use Sphinx to construct the documentation, check it for formatting
and internal consistency, and open it in a browser window. If you have a
browser window already open, say on a page you are working on, I recommend
using ``tox`` and skipping the new window and just hitting reload. You can
also run a faster build that regenerates only what has changed (though this can
become confused, and may not detect errors as vigorously)::

   $ make -C docs/ html

If something goes wrong, the error messages *should* be fairly clear, but it
may not be obvious what the right way is to do what you are trying to do. In an
ideal world, the documentation tools themselves would have good documentation,
and it would be easy to look up the right way to do things. A few pointers:

In a ``.rst`` file:

   - Web links: ``short_`` and ```long text`_``, then at the bottom of the section
     add lines ``.. _short: http://...`` and ``.. _`long text`: http://...``
   - Section references: ``:ref:`label```; before the section header add
     a line ``.. _`label`:``
   - To refer to a class, module, or function in text, use
     ``:class:`~astropy.module.ClassName```, ``:mod:`~astropy.module```,
     or ``:func:`~astropy.module.function```.
   - To get ``typewriter font`` for text, use double back-ticks. Single back-ticks
     are for Sphinx special objects like links and class names and things.
   - To get *emphasis*, use asterisks.
   - To get lists, start with a blank line, start each list item with an indented
     hyphen, indent successive lines further, and end with a blank line.
   - To get definition lists (lists where each item starts with a highlighted word
     or phrase), start with a blank line, then the word or phrase, then the definition
     indented, then the next word or phrase at indentation level zero, then its
     definition indented... end with a blank line.
   - To get a code block, end the preceding line with a double colon ``::``, leave a
     blank line, indent every line of the code, then leave a blank line at the end.

In a docstring:

   - In most text, you can write things the way you do in normal documentation.
   - When specifying an argument type, just type the fully qualified name (for
     example ``astropy.units.Quantity``) and napoleon_ will fill in the link.
   - Use the standard sections ``Parameters``, ``Returns``, and ``Note`` as
     appropriate.

Of course you can also just look at existing documentation to see how it is
done, but please check that the documentation you are using as a reference
renders correctly!

You may also want to run, from time to time::

   $ make linkcheck

This will try to make sure that all the external links in the documentation
still work, though of course they can't verify that the page that comes up is
still the intended one and not `something else`_.

.. _Sphinx: http://www.sphinx-doc.org/en/master/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _readthedocs: https://readthedocs.org/
.. _here: https://nanograv-pint.readthedocs.io/en/latest/
.. _`docstring guidelines`: https://numpydoc.readthedocs.io/en/latest/format.html
.. _napoleon: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
.. _`something else`: https://placekitten.com/


