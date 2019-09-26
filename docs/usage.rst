=====
Usage
=====

PINT is used for pulsar timing and related activities. Some of it may
make a lot more sense if you know something about the science it is used
for. You can find an excellent introduction in the Handbook of Pulsar
Astronomy, by Lorimer and Kramer. This document is aimed at using PINT
specifically, and may also be more understandable if you have used
other pulsar timing software, TEMPO_ or TEMPO2_, though we hope that
you will fint PINT sufficient for all your needs!

.. _TEMPO: http://tempo.sourceforge.net/
.. _TEMPO2: https://www.atnf.csiro.au/research/pulsar/tempo2/

Examples
--------

A number of examples are available in the ``examples`` directory_. Those
that are Jupyter notebooks are also included below.  To see how PINT is
used programmatically, look at ``examples/fit_NGC6440E.py`` (on GitHub_).

You can also run PINT from the command line with::

  pintempo --plot parfile.par timfile.tim

Use ``-h`` to get help. This is not fully developed yet, so extensions to
this (like a nice GUI) are most welcome.

.. _directory: https://github.com/nanograv/PINT/blob/master/examples/
.. _GitHub: https://github.com/nanograv/PINT/blob/master/docs/examples/fit_NGC6440E.py

.. toctree::
   :maxdepth: 2

   examples/PINT_walkthrough
   examples/MCMC_walkthrough
   examples/Example of parameter usage
   examples/TimingModel_composition
   examples/Timing_model_update_example


Time
----

With modern instrumentation, we are able to measure time - both time
intervals and an absolute time scale - to stupendous accuracy. Pulsar
timing is a powerful tool in large part because it takes advantage of
that accuracy. Getting time measurements and calculations right to this
level of accuracy does require a certain amount of care, in general and
while using (and writing PINT).

Precision
'''''''''

The first challenge that arises is numerical precision. Computers
necessarily represent real numbers to finite precision. Python, in
particular, uses floating-point numbers that occupy 64 bits, 11 of
which encode the exponent and 53 of which encode the mantissa. This
means that numbers are represented with a little less than 16 decimal
digits of precision::

   >>> import numpy as np
   >>> np.finfo(float).eps
   2.220446049250313e-16
   >>> 1 + np.finfo(float).eps
   1.0000000000000002
   >>> 1 + np.finfo(float).eps/2
   1.0
   >>> 1 + np.finfo(float).eps/2 == 1
   True

Unfortunately, we have observations spanning decades and we would often
like to work with time measurements at the nanosecond level. It turns
out that python's floating-point numbers simply don't have the precision
we need for this::

   >>> import astropy.units as u
   >>> (10*u.year*np.finfo(float).eps).to(u.ns)
   <Quantity 70.07194824 ns>

That is, if I want to represent a ten-year span, the smallest increment
python's floating-point can cope with is about 70 nanoseconds - not enough
for accurate pulsar timing work! There are a number of ways to approach
this problem, all somewhat awkward in python. One approach of interest
is that ``numpy`` provides floating-point types, for example,
``numpy.longdouble``, with more precision::

   >>> np.finfo(np.longdouble).eps
   1.084202172485504434e-19
   >>> (10*u.year*np.finfo(np.longdouble).eps).to(u.ns)
   <Quantity 0.03421482 ns>

These numbers are represented with 80 bits, and most desktop and server
machines have hardware for computing with these numbers, so they are not
much slower than ordinary ("double-precision", 64-bit) floating-point
numbers. Let me warn you about one point of possible confusion: modern
computers have very complicated cache setups that prefer data to be
aligned just so in memory, so ``numpy`` generally pads these numbers out
with zeroes and stores them in larger memory spaces. Thus you will often
see ``np.float96`` and ``np.float128`` types; these contain only
numbers with 80-bit precision. Actual 128-bit precision is not currently
available in ``numpy``, in part because on almost all current machines all
calculations must be carried out in software, which takes 20-50 times as
long.

An alternative approach to dealing with more precision than your machine's
floating-point numbers natively support is to represent numbers as a pair
of double-precision values, with the second providing additional digits
of precision to the first. These are generically called double-double
numbers, and can be faster than "proper" 128-bit floating-point numbers.
Sadly these are not implemented in ``numpy`` either. But because it is
primarily *time* that requires such precision, ``astropy`` provides a type
``astropy.time.Time`` (and ``astropy.time.TimeDelta``) that uses a similar
representation internally: two floating-point numbers, one of which is
the integer number of days (in the Julian-Day_ system) and one of which
is the fractional day. This allows very satisfactory precision::

   >>> (1*u.day*np.finfo(float).eps).to(u.ns)
   <Quantity 0.01918465 ns>
   >>> t = astropy.time.Time("2019-08-19", format="iso")
   >>> t
   <Time object: scale='utc' format='iso' value=2019-08-19 00:00:00.000>
   >>> (t + 0.1*u.ns) - t
   <TimeDelta object: scale='tai' format='jd' value=1.1102230246251565e-15>
   >>> ((t + 0.1*u.ns) - t).to(u.ns)
   <Quantity 0.09592327 ns>

Thus it is important when dealing with a time to ensure that it is stored
in either an ``astropy`` time object or a ``np.longdouble``. Because python's
default is to use less precision, it is easy to lose digits::

   >>> 1+np.finfo(np.longdouble).eps
   1.0000000000000000001
   >>> print("Number: {}".format(1+np.finfo(np.longdouble).eps))
   Number: 1.0
   >>> print("Number: {}".format(1+np.finfo(float).eps))
   Number: 1.0000000000000002
   >>> print("Number: {}".format(str(1+np.finfo(np.longdouble).eps)))
   Number: 1.0000000000000000001

.. _Julian-Day: https://aa.usno.navy.mil/data/docs/JulianDate.php

Time Scales
'''''''''''

A second concern when dealing with times at this level of precision is that
Einstein's theory of relativity becomes relevant: Clocks at the surface
of the Earth advance more slowly than clocks in space nearby because of the
slowing of time by the Earth's gravity. As the Earth moves around the Sun,
its changing velocity affects clock rates, and so does its movement deeper
and shallower in the Sun's gravity well. Of course none of these things
affect a pulsar's rotation, so we need some way to compensate for that.

On a more human scale, observations are recorded in convenient time units,
often UTC; but UTC has leap seconds, so some days have one more second (or
one fewer) than others!

The upshot of all this is that if you care about accuracy, you need to be
quite careful about how you measure your time. Fortunately, there is a
well-defined system of time scales, and ``astropy.time.Time`` automatically
keeps track of which one your time is in and does the appropriate
conversions - as long as you tell it what kind of time you're putting
in, and what kind of time you're asking for::

   >>> t = astropy.time.Time("2019-08-19", format="iso", scale="utc")
   >>> t
   <Time object: scale='utc' format='iso' value=2019-08-19 00:00:00.000>
   >>> t.tdb
   <Time object: scale='tdb' format='iso' value=2019-08-19 00:01:09.183>

The conventional time scale for working with pulsars, and the one PINT
uses, is Barycentric Dynamical Time (TDB). You should be aware that there
is another time scale, not yet supported in PINT, called Baycentric
Coordinate Time (TCB), and that because of different handling of
relativistic corrections, it does not advance at the same rate as TDB
(there is also a many-second offset). TEMPO2 uses TCB by default, so
you may encounter pulsar timing models or even measurements that use
TCB. PINT will attempt to detect this and let you know.

Note that the need for leap seconds is because the Earth's rotation is
somewhat erratic - no, we're not about to be thrown off, but its
unpredictability can get as large as a second after a few years. So
the International_Earth_Rotation_Service_ announces leap seconds about
six months in advance. This means that ``astropy`` and pint need to
keep their lists of leap seconds up-to-date by checking the IERS
website from time to time.

It is also conventional to record pulsar data with reference to an
observatory clock, usually a maser, that may drift with respect to
International Atomic Time (TAI_). Usually GPS is used to track the
deviations of this observatory clock and record them in a file. PINT
also needs up-to-date versions of these observatory clock correction files
to produce accurate results.

Even more detail about how PINT handles time scales is available on the
github wiki_.

.. _International_Earth_Rotation_Service: https://www.iers.org/IERS/EN/Home/home_node.html
.. _TAI: https://www.bipm.org/en/bipm-services/timescales/tai.html
.. _wiki: https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

External Data
-------------

In order to provide sub-microsecond accuracy, PINT needs a certain
number of data files, for example Solar System ephemerides, that
would be cumbersome to include in the package itself. Further, some
of this external data needs to be kept up-to-date - precise measurements
of the Earth's rotation, for example, or observatory clock corrections.

Most of this external data is obtained through ``astropy``'s data downloading
mechanism (see ``astropy.utils.data``). This will result in the data being
downloaded the first time it
is required on your machine but thereafter stored in a "cache" in your home
directory. If you plan to operate offline, you may want to run some commands
before disconnecting to ensure that this data has been downloaded. Data
that must be up-to-date is generally in the form of a time series, and
"up-to-date" generally means that it must cover the times that occur in
your data. This can be an issue for simulation and forecasting; there should
always be a mechanism to allow out-of-date data if you can accept lower
accuracy.

Not all the data that PINT uses is easily accessible for programs to
download. Observatory clock corrections, for example, may need to be
obtained from the observatory through various means (often talking to a
support scientist). We intend that PINT should notify you when this is
necessary, but be aware that you may obtain reduced accuracy if you
have old clock correction files.


