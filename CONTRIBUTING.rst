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

1. Fork the `pint` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pint.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv pint
    $ cd pint/
    $ pip install -r requirements_dev.txt
    $ pip install -r requirements.txt
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests. Also check that any new docs are formatted correctly::

    $ make test
    $ make docs

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

8. The maintainers will review and comment on the PR. If accepted, it will be
merged into the master branch.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.7 and 3.5+. Check
   https://travis-ci.org/nanograv/pint/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To track and checkout another user's branch::

    $ git remote add other-user-username https://github.com/other-user-username/pint.git
    $ git fetch other-user-username
    $ git checkout --track -b branch-name other-user-username/branch-name

Coding Style
------------

We would like `pint` to be easy to use and easy to contribute to. To
this end we'd like to ask that if you're going to contribute code or
documentation that you try to follow the below style advice. We know
that not all of the existing code does this, and it's something we'd
like to change.

   - Functions, modules, and classes should have docstrings. These should
     start with a short one-line description of what the function (or module
     or class) does. Then, if you want to say more than fits in a line, a
     blank line and a longer description. If you can, if it's something that
     will be used widely, please follow the numpy docstring guidelines_ -
     these result in very helpful usage descriptions in both the interpreter
     and online docs.    
   - Code should follow PEP8_. Most importantly, if at all possible, class
     names should be in CamelCase, while function names should be in
     snake_case. There is also advice there on line length and whitespace.
     You can check your code with the tool ``flake8``, but I'm afraid
     much of PINT's existing code emits a blizzard of warnings.
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
     something that has units, please use `astropy.units.Quantity` to
     keep track of what these units are. If you need a plain floating-point
     number, use ``.to(u.m).value``, where ``u.m`` should be replaced by
     the units you want the number to be in. This will raise an exception
     (good!) if the units can't be converted (``u.kg`` for example) and
     convert if it's in a compatible unit (``u.cm``, say). Adding units
     when you know what they are is as simple as multiplying.
   - When you want to let the user know some information from deep inside
     `pint`, remember that they might be running a GUI application where
     they can't see what comes out of `print`. Please use `astropy.log`.
     Conveniently, this has levels `astropy.log.debug`, `astropy.log.info`,
     `astropy.log.warning`, and `astropy.log.error`; the end user can
     decide which levels of severity they want to see.
   - When something goes wrong and your code can't continue and still
     produce a sensible result, please raise an exception. Usually
     you will want to raise a ValueError with a description of what
     went wrong, but if you want users to be able to do something with
     the specific thing that went wrong, you can quickly create a new
     exception class (no more than ``class PulsarProblem(ValueError): pass``)
     that the user can specifically catch and distinguish from other
     exceptions. Similarly, if you're catching an exception some code might
     raise, use ``except PulsarProblem:`` to catch just the kind you
     can deal with.

.. _guidelines: https://numpy.org/devdocs/docs/howto_document.html
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _pytest: https://docs.pytest.org/en/latest/
.. _pdb: https://docs.python.org/3/library/pdb.html
