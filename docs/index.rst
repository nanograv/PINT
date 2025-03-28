PINT Is Not TEMPO3
==================

.. image:: https://github.com/nanograv/pint/workflows/CI%20Tests/badge.svg
   :target: https://github.com/nanograv/pint/actions
   :alt: Actions Status

.. image:: https://readthedocs.org/projects/nanograv-pint/badge/?version=latest
    :target: https://nanograv-pint.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Build Status

.. image:: https://coveralls.io/repos/github/nanograv/PINT/badge.svg?branch=master
    :target: https://coveralls.io/github/nanograv/PINT?branch=master
    :alt: Code Coverage

PINT is a project to develop a new pulsar timing solution based on
python and modern libraries. It is still in active development,
but it can already produce residuals from most "normal"
timing models that agree with Tempo and Tempo2 to within ~10
nanoseconds. It can be used within python scripts or notebooks,
and there are several command line tools that come with it.

The primary reasons we are developing PINT are:

* To have a robust system to check high-precision timing results that is
  completely independent of TEMPO and Tempo2

* To make a system that is easy to extend and modify due to a good design
  and the use of a modern programming language, techniques, and libraries.

How the documentation is organized
----------------------------------

We try to keep the PINT documentation divided into four categories:

:ref:`Tutorials`
   Easy-to-follow walkthroughs to show new users what PINT can do. No
   explanations or alternatives or fallback methods, just getting people to a
   point where they know what questions to ask. These should work for everyone
   without surprises or problems. Jupyter notebooks are a natural format for
   these. Includes :ref:`basic-installation`.
:ref:`Explanation`
   Descriptions of how PINT works, why certain design choices were made, what
   the underlying concepts are and so forth. This is for users who know more
   or less what to do but want to understand what is going on.
:ref:`Reference`
   Specific details of how particular things work. This is for users who are
   trying to use PINT and know what function or object or method they need but
   need the details of how it works.
:ref:`How-tos`
   Detailed guides on how to accomplish specific things, for people who
   already know what questions to ask. Explanations and reference are not
   needed here but even experienced users can benefit from being pointed
   at the right way to do something, with fallbacks and troubleshooting
   advice.

This is based on the Django_ documentation structure, and it is intended both
to help users find the correct information and to help documentation writers
keep clear in our heads who we are writing for and what we are trying to
communicate to them.

.. _Django: https://docs.djangoproject.com/en/2.2/

.. toctree::
   :maxdepth: 3

   tutorials
   explanation
   reference
   howto
   history
   authors
   dependent-packages


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
