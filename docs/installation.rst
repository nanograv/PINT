.. highlight:: shell
.. _`Installation`:

===================
How to Install PINT
===================

There are two kinds of PINT installation you might be interested in. The first
is a simple PINT installation for someone who just wants to use PINT. The
second is an installation for someone who wants to be able to run the tests and
develop PINT code. The latter naturally requires more other python packages and
is more complicated (but not too much).

Basic Install
-------------

If your python setup is "nice", you should be able to install as easily as::

   $ git clone https://github.com/nanograv/PINT.git
   $ cd PINT
   $ mkvirtualenv -p `which python3` pint
   (pint) $ pip install -e .
   (pint) $ python
   >>> import pint

This *should* install PINT along with any python packages it needs to run. (If
you want to run the test suite or work on PINT code, see below.)

Unfortunately there are a number of reasons this can go wrong. Most have to do
with not having a "nice" python environment.

Basic dependencies
''''''''''''''''''

You need a python interpreter (probably came with your operating system), its
support for the package isolation tool virtualenv_, the convenience functions
in virtualenvwrapper_, and the package installation tool pip_.  You probably
need to  have these installed system-wide or in some other way. Sorry. Try some
Googling if you don't.[1]_

No python 3
'''''''''''

Install python 3. It's time.

But if you can't, PINT will work with python 2, so just change the last two
lines::

   $ mkvirtualenv -p `which python2` pint
   (pint) $ pip install -e .

This will leave you with an old version of astropy and many other packages, as
almost everyone has ceased support for python 2. Unfortunately, as you probably
know if you have read this far, some systems and software are still in the
Stone Age and don't work with python 3.

Bad ``PYTHONPATH``
''''''''''''''''''

The virtualenv mechanism uses environment variables to create an isolated
python environment into which you can install and upgrade packages without
affecting or being affected by anything in any other environment. Unfortunately
it is possible to defeat this by setting the ``PYTHONPATH`` environment
vairable. Double unfortunately, setting the ``PYTHONPATH`` environment used to
be the Right Way to use python things that weren't part of your operating
system. So many of us have ``PYTHONPATH`` set in our shells. This is often
signalled by the otherwise impenetrable error message "error: 'egg_base' must
be a directory name (got src)". You can check this::

   $ printenv PYTHONPATH

If you see any output, chances are that's causing problems with your
virtualenvs. You probably need to go look in your ``.bashrc`` and/or
``.bash_profile`` to see where that variable is being set and remove it. Yes,
it is very annoying that you have to do this.

Previous use of ``pip install --user``
''''''''''''''''''''''''''''''''''''''

Similarly, it used to be recommended to install packages locally as your user
by running ``pip install --user thing``. Unfortunately this causes something of
the same problem as having a ``PYTHONPATH`` set, where packages installed
outside your virtualenv can obscure the ones you have inside, producing bizarre
error messages. Record your current packages with ``pip freeze``, then try,
outside a virtualenv, doing ``pip list`` with various options, and ``pip
uninstall``; you shouldn't be able to uninstall anything system-wise (do not
use ``sudo``!) and you shouldn't be able to uninstall anything in an inactive
virtualenv. So once you've blown away all those packages, you should be able to
work in clean virtualenvs. If you saved the output of ``pip freeze`` above, you
should be able to use it to create a virtualenv with all the same packages you
used to have in your user directory.

Bad ``conda`` setup
'''''''''''''''''''

Conda_ is a tool that attempts to create isolated environments, like a
combination of virtualenv and ``pip``. It should make installing scientific
software with lots of dependencies easy and reliable, and you should just be
able to set up an appropriate ``conda`` environment and use the basic install
instructions above. But it may not work.

Specifically, for some reason the python 3 version of ``conda`` does not
provide the ``gdbm`` module, which ``astropy`` needs to work on Linux. Good
luck.

.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _virtualenvwrapper: https://virtualenvwrapper.readthedocs.io/en/latest/
.. _Conda: https://docs.conda.io/en/latest/

Installing PINT for Development
-------------------------------

You will need to be able to carry out a basic install of PINT as above. So try
that first. Then you will need to install the additional development dependencies::

   $ pip install -Ur requirements_dev.txt

Some of those packages may have been already installed, for example by MacPorts.
For MacPorts users, this command will get many of the requirements::

    port install py27-numpy py27-scipy py27-astropy py27-nose py27-cython py27-emcee py27-sphinx py27-sphinx_rtd_theme

`TEMPO`_ is not required, but if you have it installed PINT can find clock
correction files in ``$TEMPO/clock``

`Tempo2`_ is not required, but if you have it installed PINT can find clock
correction files in ``$TEMPO2/clock``

PINT development (building the documentation) requires pandoc_, which isn't a
python package and therefore needs to be installed in some way appropriate for
your operating system. On Linux you may be able to just run::

   $ apt install pandoc

Otherwise, there are several ways to `install pandoc`_

.. _[1]: If you don't have `pip`_ installed, this `Python installation guide`_ can guide
   you through the process.
.. _pip: https://pip.pypa.io/en/stable/
.. _TEMPO: http://tempo.sourceforge.net
.. _Tempo2: https://bitbucket.org/psrsoft/tempo2
.. _Python installation guide: https://docs.python-guide.org/starting/installation/
.. _pandoc: https://pandoc.org/
.. _`install pandoc`: https://pandoc.org/installing.html
