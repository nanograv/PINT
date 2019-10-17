Setting up your environment for PINT development
================================================

Working on PINT code requires a few more tools than simply running PINT, and
there are a few settings that can make it much easier. Some of what follows
will depend on your editor, but most of it is possible in any modern
programmer's editor, including Sublime Text, Atom, Visual Studio Code, PyCharm,
vim/neovim, and emacs. (Okay those last two are only arguably modern but they
are extensible enough that they can be made to do most of the things described
here.) Some of these tools may also be available in more basic editing
environments like the Jupyter notebook, the JupyterLab text editor, and Spyder.

What you should expect your editor to do
----------------------------------------

It may take some configuration, but once set up any modern editor should be
able to do the following:

   - Highlight python syntax.
   - Flag syntax or style errors (line length, unused/undefined variables,
     dubious exception handling) visually as you edit.
   - Offer completions for any identifier (keyword, function, variable, et
     cetera) you start typing.
   - Reformat text into the black code style with a keypress.
   - Sort your imports into the standard arrangement with a keypress.
   - Jump to the definition of a function, class, or method with a keypress.

Command-line tools and automation
---------------------------------

PINT is developed with ``git`` and on GitHub_. Some operations are presented
graphically in the web interface, but in many cases you will want to do
something direct on your local machine. Having the right tools available and
configured should make this easy.

.. _GitHub: https://github.com/nanograv/PINT
