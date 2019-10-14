.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/nanograv/pint/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* The output of ``pint.__version__`` and ``pint.__file__``
* Any details about your local setup that might be helpful in troubleshooting,
  such as the command used to install PINT and whether you are using a virtualenv.
* Detailed steps to reproduce the bug, as simply as possible.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

pint could always use more documentation, whether as part of the
official pint docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/nanograv/pint/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `pint` for local development.

1. Fork_ the `pint` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pint.git

3. Install your local copy into a virtualenv. Assuming you have
   virtualenvwrapper installed, this is how you set up your fork for local
   development::

    $ mkvirtualenv pint
    $ cd PINT/
    $ pip install -r requirements_dev.txt
    $ pip install -r requirements.txt
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the
   tests. Also check that any new docs are formatted correctly::

    $ make test
    $ tox
    $ make docs

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

8. Check that our automatic testing "Travis CI" passes your code. If
   problems crop up, fix them, commit the changes, and push a new version,
   which will automatically update the pull request::

   $ git add pint/file-i-just-fixed.py
   $ git commit -m "Fixed bug where..."
   $ git push

9. The maintainers will review and comment on the PR. They may ask why
   you made certain design decisions or ask you to make some stylistic or
   functional changes. If accepted, it will be merged into the master branch.

.. _Fork: https://help.github.com/en/articles/fork-a-repo

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. Try to write clear pythonic_ code, follow our_coding_style_, and think
   about how others might use your new code.
2. The pull request should include tests that cover both the expected
   behaviour and sensible error reporting when given bad input.
3. If the pull request adds or changes functionality, the docs should
   be updated. Put your new functionality into a function with a
   docstring. Check the HTML documentation produced by ``make docs``
   to make sure your new documentation appears and looks reasonable.
4. The pull request should work for Python 2.7 and 3.5+. Check
   https://travis-ci.org/nanograv/pint/pull_requests
   and make sure that the tests pass for all supported Python versions.

Additional Tools
----------------

To track and checkout another user's branch::

    $ git remote add other-user-username https://github.com/other-user-username/pint.git
    $ git fetch other-user-username
    $ git checkout --track -b branch-name other-user-username/branch-name

To run tests on just one file::

   $ pytest tests/test_my_new_thing.py

To run just one test::

   $ pytest tests/test_my_new_thing.py::test_specific

To test everything but start with tests that failed last time, stopping when
something goes wrong (this is great when you're trying to fix that one bug; if
you haven't you'll get new error messages, if you have, it'll continue on to
run all the tests)::

   $ pytest --ff -x

To drop into the python debugger at the point where a test fails so you can
investigate_, for example go up and down the call history and inspect local
variables::

   $ pytest --pdb -x

The python_debugger_ also allows you to step through your code, put in
breakpoints, and many other things. It can save a ton of time compared to
putting print statements in and rerunning your code, especially if the code
takes a while and you don't know exactly what you want to inspect. There is a
friendlier version, pdbpp_, that you can transparently swap in::

   $ pip install pdbpp

To run tests on multiple python versions and build
the documentation in parallel::

   $ tox --parallel=auto

If this finds a problem in just one python environment that doesn't appear in
your development environment, you can run just the problem environment::

   $ tox -e py27

You can also run other things in the environments ``tox`` uses, including
interactive python sessions (though these will include only PINT's installation
requirements, so no IPython)::

   $ tox -e py27 -- pytest --ff --pdb -x
   $ tox -e py27 -- pytest tests/test_my_new_thing.py
   $ tox -e py27 -- python

Under ``examples/`` there are a few Jupyter notebooks. These actually get
incorporated into the online documentation (you may have seen them). To avoid
headaches, we don't store these as notebooks on github but as special markdown
files. If you are using ``jupyter`` or ``jupyter-lab``, it should be smart
enough to synchronize these between the storage format and normal notebooks,
but if there is any confusion, try ``make notebooks``, which synchronizes the
two formats and runs all the notebooks to fill in the outputs. If something
goes wrong, try ``jupytext --sync``, which synchronizes the code between the
notebooks and the storage format but doesn't run the notebooks.

If you make a mistake and get ``git`` into a strange or awkward state. Don't
panic, and try Googling the specific error message. ``git`` is quite thorough
about keeping history around, so you can probably undo whatever has happened,
especially if you have been pushing your changes to GitHub. If it helps, there
is "Dang_it_git_" (there is a ruder version which may feel more appropriate
in the moment).

.. _investigate: https://realpython.com/python-debugging-pdb/
.. _python_debugger: https://docs.python.org/3/library/pdb.html
.. _pdbpp: https://github.com/pdbpp/pdbpp
.. _Dang_it_git: https://dangitgit.com/

.. _our_coding_style:

Coding Style
------------

We would like `PINT` to be easy to use and easy to contribute to. To
this end we'd like to ask that if you're going to contribute code or
documentation that you try to follow the below style advice. We know
that not all of the existing code does this, and it's something we'd
like to change.

Concrete rules:

   - Code should follow PEP8_. This constrains whitespace,
     formatting, and naming.
   - Code should follow the black_ style.
   - Use the standard formats for imports.

      - The only abbreviated imports are ``import numpy as np``,
        ``import astropy.units as u``, and
        ``import astropy.constants as c``; always use these modules
        in this form.
      - If you want to import a deeply nested module, use
        ``from pint.models import parameter``.
      - If you are using a function frequently or the module name
        is long, use ``from pint.utils import interesting_lines``
      - Sort imports with isort_.
      - Remove all unused imports.

   - Raise an exception if the code cannot continue and produce
     correct results.
   - Use :mod:`~astropy.log` to signal conditions the user should
     know about but that do not prevent the code from producing
     correct results.
   - Use keyword argument names when calling a function with more
     than two or three arguments.
   - Every public function or class should have a docstring. It
     should be in the correct format (numpy guidelines_).
   - Use :class:`~astropy.units.Quantity` for things with physical units.
   - PINT should work with python 2.7, 3.5, 3.6, 3.7, and later. There
     is no need to maintain compatibility with older versions, so use
     approprate modern constructs like sets and iterators.
   - Use six_ to manage python 2/3 compatibility problems.

More general rules and explanations:

   - Think about how someone might want to use your code in various ways.
     Is it called something helpful so that they will be able to find it?
     Will they be able to do something different with it than you wrote
     it for? How will it respond if they give it incorrect values?
   - Code should follow PEP8_. Most importantly, if at all possible, class
     names should be in CamelCase, while function names should be in
     snake_case. There is also advice there on line length and whitespace.
     You can check your code with the tool ``flake8``, but I'm afraid
     much of PINT's existing code emits a blizzard of warnings.
   - Files should be formatted according to the much more specific rules
     enforced by the tool black_. This is as simple as ``pip install black``
     and then running ``black`` on a python file. If an existing file does not
     follow this style please don't convert it unless you are modifying almost
     all the file anyway; it will mix in formatting changes with the actual
     substantive changes you are making when it comes time for us to review
     your pull request.
   - Functions, modules, and classes should have docstrings. These should
     start with a short one-line description of what the function (or module
     or class) does. Then, if you want to say more than fits in a line, a
     blank line and a longer description. If you can, if it's something that
     will be used widely, please follow the numpy docstring guidelines_ -
     these result in very helpful usage descriptions in both the interpreter
     and online docs. Check the HTML documentation for the thing you are
     modifying to see if it looks okay.
   - Tests are great! When there is a good test suite, you can
     make changes without fear you're going to break something. *Unit*
     tests are a special kind of test, that isolate the functionality
     of a small piece of code and test it rigorously.

      - When you write a new function, write a few tests for it. You
        will never have a clearer idea of how it's supposed to work
        than right after you wrote it. And anyway you probably used
        some code to see if it works, right? Make that into a test,
        it's not hard. Feed it some bogus data, make sure it raises
        an exception. Make sure it does the right thing on empty lists,
        multidimensional arrays, and NaNs as input - even if that's to
        raise an exception. We use pytest_. You can easily run just your
        new tests.
      - Give tests names that describe what property of what thing they
        are testing.  We don't call test functions ourselves so there
        is no advantage to them having short names. It is perfectly
        reasonable to have a function called
        ``test_download_parallel_fills_cache`` or
        ``test_cache_size_changes_correctly_when_files_are_added_and_removed``.
      - If your function depends on complicated other functions or data,
        consider using something like `unittest.Mock` to replace that
        complexity with mock functions that return specific values. This
        is designed to let you test your function specifically in
        isolation from potential bugs in other parts of the code.
      - When you find a bug, you presumably have some code that triggers
        it. You'll want to narrow that down as much as possible for
        debugging purposes, so please turn that bug test case into a
        test - before you fix the bug! That way you know the bug *stays*
        fixed.
      - If you're trying to track down a tricky bug and you have a test
        case that triggers it, running
        ``pytest tests/test_my_buggy_code.py --pdb`` will drop you into
        the python debugger pdb_ at the moment failure occurs so you
        can inspect local variables and generally poke around.

   - When you're working with a physical quantity or an array of these,
     something that has units, please use :class:`~astropy.units.Quantity` to
     keep track of what these units are. If you need a plain floating-point
     number out of one, use ``.to(u.m).value``, where ``u.m`` should be
     replaced by the units you want the number to be in. This will raise
     an exception (good!) if the units can't be converted (``u.kg`` for
     example) and convert if it's in a compatible unit (``u.cm``, say).
     Adding units to a number when you know what they are is as simple as
     multiplying.
   - When you want to let the user know some information from deep inside
     PINT, remember that they might be running a GUI application where
     they can't see what comes out of ``print``. Please use :mod:`~astropy.logger`.
     Conveniently, this has levels ``debug``, ``info``,
     ``warning``, and ``error``; the end user can
     decide which levels of severity they want to see.
   - When something goes wrong and your code can't continue and still
     produce a sensible result, please raise an exception. Usually
     you will want to raise a ValueError with a description of what
     went wrong, but if you want users to be able to do something with
     the specific thing that went wrong (for example, they might want to
     use an exception to know that they have emptied a container), you
     can quickly create a new exception class (no more than
     ``class PulsarProblem(ValueError): pass``)
     that the user can specifically catch and distinguish from other
     exceptions. Similarly, if you're catching an exception some code might
     raise, use ``except PulsarProblem:`` to catch just the kind you
     can deal with.

There are a number of tools out there that can help with the mechanical
aspects of cleaning up your code and catching some obvious bugs. Most of
these are installed through PINT's ``requirements_dev.txt``.

   - flake8_ reads through code and warns about style issues, things like
     confusing indentation, unused variable names, un-initialized variables
     (usually a typo), and names that don't follow python conventions.
     Unfortunately a lot of existing PINT code has some or all of these
     problems. ``flake8-diff`` checks only the code that you have touched -
     for the most part this pushes you to clean up functions and modules
     you work on as you go.
   - isort_ sorts your module's import section into conventional order.
   - black_ is a draconian code formatter that completely rearranges the
     whitespace in your code to standardize the appearance of your
     formatting. ``blackcellmagic`` allows you to have ``black`` format the
     cells in a Jupyter notebook.
   - pre-commit_ allows ``git`` to automatically run some checks before
     you check in your code. It may require an additional installation
     step.
   - ``make coverage`` can show you if your tests aren't even exercising
     certain parts of your code.
   - editorconfig_ allows PINT to specify how your editor should format
     PINT files in a way that many editors can understand (though some,
     including vim and emacs, require a plugin to notice).

Your editor, whether it is emacs, vim, JupyterLab, Spyder, or some more
graphical tool, can probably be made to understand that you are editing
python and do things like highlight syntax, offer tab completion on
identifiers, automatically indent text, automatically strip trailing
white space, and possibly integrate some of the above tools.

.. _six: https://six.readthedocs.io/
.. _black: https://black.readthedocs.io/en/stable/
.. _isort: https://pypi.org/project/isort/
.. _flake8: http://flake8.pycqa.org/en/latest/
.. _pre-commit: https://pre-commit.com/
.. _editorconfig: https://editorconfig.org/

.. _pythonic:

The Zen of Python
~~~~~~~~~~~~~~~~~
by Tim Peters

| Beautiful is better than ugly.
| Explicit is better than implicit.
| Simple is better than complex.
| Complex is better than complicated.
| Flat is better than nested.
| Sparse is better than dense.
| Readability counts.
| Special cases aren't special enough to break the rules.
| Although practicality beats purity.
| Errors should never pass silently.
| Unless explicitly silenced.
| In the face of ambiguity, refuse the temptation to guess.
| There should be one-- and preferably only one --obvious way to do it.
| Although that way may not be obvious at first unless you're Dutch.
| Now is better than never.
| Although never is often better than *right* now.
| If the implementation is hard to explain, it's a bad idea.
| If the implementation is easy to explain, it may be a good idea.
| Namespaces are one honking great idea -- let's do more of those!


.. _guidelines: https://numpy.org/devdocs/docs/howto_document.html
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _pytest: https://docs.pytest.org/en/latest/
.. _pdb: https://docs.python.org/3/library/pdb.html
