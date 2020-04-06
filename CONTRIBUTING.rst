.. highlight:: shell
.. _`Contributing`:

=========================
How to Contribute to PINT
=========================

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

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/nanograv/pint/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with `good first issue`_,
`help wanted`_, or bug_ is open to whoever wants to implement it. If you want to fix a bug or
add any other code, please use GitHub and suggest your changes in the form of a
Pull Request (see below); this makes it easy for everyone to examine your changes, discuss
them with you, and update them as needed.

.. _`good first issue`: https://github.com/nanograv/PINT/labels/good%20first%20issue
.. _`help wanted`: https://github.com/nanograv/PINT/labels/help%20wanted
.. _bug: https://github.com/nanograv/PINT/labels/bug

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.  If your idea is
for a new feature or an important change, you may want to open an issue where
the idea can be discussed before you write too much code.

Write Documentation
~~~~~~~~~~~~~~~~~~~

PINT could always use more documentation, whether as part of the
official pint docs, in docstrings, or even on the web in blog posts,
articles, and such.

Writing documentation is a great way to get started: everyone wants there to be
documentation, but no one wants to stop writing code long enough to write it,
so we are all very grateful when you do. And as a result of figuring out enough
to write good documentation, you come to understand the code very well.

Get Started!
------------

Ready to contribute? Here's how to set up PINT for local development.

1. Fork_ the ``pint`` repo on GitHub.
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

1. Try to write clear :ref:`pythonic` code, follow our :ref:`CodingStyle`, and think
   about how others might use your new code.
2. The pull request should include tests that cover both the expected
   behaviour and sensible error reporting when given bad input.
3. If the pull request adds or changes functionality, the docs should
   be updated. Put your new functionality into a function with a
   docstring. Check the HTML documentation produced by ``make docs``
   to make sure your new documentation appears and looks reasonable.
4. The pull request should work for Python 2.7 and 3.6+. Check
   https://travis-ci.org/nanograv/pint/pull_requests
   and make sure that the tests pass for all supported Python versions.

