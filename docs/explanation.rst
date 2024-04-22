.. _`Explanation`:

Explanation
===========

This section is to explain pulsar timing, how PINT works, and why it is built the way it is.

PINT is used for pulsar timing and related activities. Some of it may
make a lot more sense if you know something about the science it is used
for. You can find an excellent introduction in the Handbook of Pulsar
Astronomy, by Lorimer and Kramer. This document is aimed at using PINT
specifically, and may also be more understandable if you have used
other pulsar timing software, TEMPO_ or TEMPO2_, though we hope that
you will find PINT sufficient for all your needs!

.. _TEMPO: http://tempo.sourceforge.net/
.. _TEMPO2: https://www.atnf.csiro.au/research/pulsar/tempo2/

Time
----

With modern instrumentation, we are able to measure time - both time
intervals and an absolute time scale - to stupendous accuracy. Pulsar
timing is a powerful tool in large part because it takes advantage of
that accuracy. Getting time measurements and calculations right to this
level of accuracy does require a certain amount of care, in general and
while using (and writing) PINT.

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
is another time scale, not yet fully supported in PINT, called Barycentric
Coordinate Time (TCB). Because of different handling of relativistic 
corrections, the TCB timescale does not advance at the same rate as TDB
(there is also a many-second offset). TEMPO2 uses TCB by default, so
you may encounter pulsar timing models or even measurements that use
TCB. PINT provides a command line tool `tcb2tdb` to approximately convert
TCB timing models to TDB. PINT can also optionally convert TCB timing models
to TDB (approximately) upon read.

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

Even more detail about how PINT handles time scales is available on the github
wiki_.

Specifically, there is a complexity in using MJDs to specify times in the UTC
time scale, which is the customary way observatories work. PINT attempts to
handle this correctly by default, but if you see timing anomalies on days with
leap seconds, this may be the problem. Alternatively, you may not be using
up-to-date leap-second data files, or the process that generated the MJDs may
not (this is a particular concern when working with X-ray or gamma-ray data).

.. _International_Earth_Rotation_Service: https://www.iers.org/IERS/EN/Home/home_node.html
.. _TAI: https://www.bipm.org/en/bipm-services/timescales/tai.html
.. _wiki: https://github.com/nanograv/PINT/wiki/Clock-Corrections-and-Timescales-in-PINT

Dispersion Measure (DM)
-----------------------
Radio waves emitted by the pulsar experience dispersion as they travel through the ionized 
interstellar medium (ISM). The time delay due to the interstellar dispersion is given by
:math:`\frac{K\times DM}{\nu^2}`, where :math:`\nu` is the frequency of the radio signal.
The dominant source of this dispersion is the presence of free electrons in the ISM, and 
to a first approximation, the DM can be interpreted as the electron column density along the 
line of sight to the pulsar. :math:`K` is known as the DM constant, and should be equal to 
:math:`\frac{e^2}{8 \pi ^2 c \epsilon _0 m_e} \approx 1.3445365918(9)\times 10^{-7}\; \text{m}^2/\text{s}` 
based on the latest measurements of the physical constants. However, pulsar astronomers have 
traditionally used a fixed value :math:`1.3447217\times 10^{-7}\; \text{m}^2/\text{s}` for :math:`K` over 
the years. For example, the Handbook of Pulsar Astronomy by Lorimer & Kramer (Chapter 5) provides 
the value :math:`2.41\times 10^{-4}\; \text{MHz}^{-2} \text{pc}\, \text{cm}^{-3} s^{-1}` for the 
reciprocal of :math:`K`. `PINT` follows this convention to be compatible with older pulsar 
ephemerides and with other pulsar timing packages. The value of :math:`K` used by `PINT` can be 
accessed as ``pint.DMconst``.

It should also be noted that there are other effects contributing to the dispersion delay than 
the free electrons, such as ions in the ISM, interstellar magnetic fields, and the ISM temperature.
Hence, it has been argued (see Kulkarni 2020 https://arxiv.org/abs/2007.02886) that the dispersion
slope :math:`K\times DM` should be treated as the primary observable rather than the DM, which 
is usually interpreted as the electron column density. The dispersion slope corresponding to a DM value
can be computed using :func:`pint.derived_quantities.dispersion_slope`. A DM value measured based 
on the conventional value of :math:`K` can be converted to a value based on the latest physical 
constant values using :func:`pint.utils.convert_dispersion_measure`.

The total DM and dispersion slope predicted by a given timing model (:class:`pint.models.timing_model.TimingModel`)
for a given set of TOAs (:class:`pint.toa.TOAs`) can be computed using :func:`pint.models.timing_model.TimingModel.total_dm`
and :func:`pint.models.timing_model.TimingModel.total_dispersion_slope` methods respectively.
 
Offsets in pulsar timing
------------------------
Offsets arise in pulsar timing models for a variety of reasons. The different types of
offsets are listed below:

Overall phase offset (PHOFF)
''''''''''''''''''''''''''''
The pulse phase corresponding to the TOAs are usually computed in reference to an arbitrary 
fiducial TOA known as the TZR TOA (see :class:`pint.models.absolute_phase.AbsPhase`). Since the 
choice of the TZR TOA is arbitrary, there can be an overall phase offset between the TZR TOA and 
the measured TOAs. There are three ways to account for this offset: (1) subtract the weighted mean
from the timing residuals, (2) make the TZR TOA (given by the `TZRMJD` parameter) fittable, or 
(3) introduce a fittable phase offset parameter between measured TOAs and the TZR TOA.
Traditionally, pulsar timing packages have opted to implicitly subtract the residual mean, and this
is the default behavior of `PINT`. Option (2) is hard to implement because the TZR TOA may be 
specified at any observatory, and computing the TZR phase requires the application of the clock 
corrections. The explicit phase offset (option 3) can be invoked by adding the `PHOFF` parameter, 
(implemented in :class:`pint.models.phase_offset.PhaseOffset`). If the explicit offset `PHOFF`
is given, the implicit residual mean subtraction behavior will be disabled.

In the pulsar ephemeris (par) file, an example `PHOFF` parameter looks like this:

    `PHOFF   0.1   1   0.001`

System-dependent delays (`JUMP`s)
'''''''''''''''''''''''''''''''''
It is very common to have TOAs for the same pulsar obtained using different observatories, 
telescope receivers, backend systems, and data processing pipelines, especially in long-running 
campaigns. Delays can arise between the TOAs measured using such different systems due to, among
other reasons, instrumental delays, differences in algorithms used for RFI mitigation, folding, TOA 
measurement etc., and the choice of different template profiles used for TOA measurement. Such 
offsets are usually modeled using phase jumps (the `JUMP` parameter, see :class:`pint.models.jump.PhaseJump`) 
between TOAs generated from different systems.

Here are some examples for `JUMP` parameters in a par file:
    `JUMP   -f 430_PUPPI    0.01  1   1e-5`
    `JUMP   tel ao          0.01  1   1e-5`
    `JUMP   mjd 55000 55100 0.01  1   1e-5`
    `JUMP   freq 1000 1400  0.01  1   1e-5`

System-dependent DM offsets (`DMJUMP`s and `FDJUMPDM`s)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''
Similar to system-dependent delays, offsets can arise between wideband DM values measured using 
different systems due to the choice of template portraits with different fiducial DMs. This is 
usually modeled using DM jumps (the `DMJUMP` parameter, see :class:`pint.models.dispersion_model.DispersionJump`).
This type of offset only applies to the wideband DM values and not to the wideband TOAs. 

Here are some examples for `DMJUMP` parameters in a par file:
    `DMJUMP   -f 430_PUPPI    1e-4  1   1e-5`
    `DMJUMP   tel ao          1e-4  1   1e-5`
    `DMJUMP   mjd 55000 55100 1e-4  1   1e-5`
    `DMJUMP   freq 1000 1400  1e-4  1   1e-5`

Similar offsets also arise in the case of narrowband TOAs. Unlike the wideband case, these offsets 
manifest as system-dependent corrections to the DM delay. They are modeled using the `FDJUMPDM` parameters
(see see :class:`pint.models.dispersion_model.FDJumpDM`)

Here are some examples for `FDJUMPDM` parameters in a par file:
    `FDJUMPDM   -f 430_PUPPI       1e-4  1   1e-5`
    `FDJUMPDM   -f L-wide_PUPPI    1e-4  1   1e-5`

System- and frequency-dependent offsets (`FDJUMP`s)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
In narrowband datasets, the template profiles often do not adequately model the frequency-dependent 
evolution of pulse profiles, resulting in a frequency-dependent artefact in the timing residuals.
This systematic effect is usually modeled phenomenologically as a log-polynomial function of frequency 
whose coefficients are the so-called FD parameters (see :class:`pint.models.frequency_dependent.FD`). 
Sometimes, this effect needs to be modeled separately for different systems since different template 
profiles will be used for each system. This is achieved through system-dependent FD parameters or `FDJUMP`s 
(see :class:`pint.models.fdjump.FDJump`). 

Here are some examples for `FDJUMP` parameters in a par file:
    `FD1JUMP   -f L-wide_PUPPI    1e-4  1   1e-5`
    `FD2JUMP   -f L-wide_PUPPI    1e-4  1   1e-5`
    `FD1JUMP   -f 430_PUPPI       1e-4  1   1e-5`
    `FD2JUMP   -f 430_PUPPI       1e-4  1   1e-5`

Observatories
-------------

PINT comes with a number of defined observatories.  Those on the surface of the Earth 
are :class:`~pint.observatory.topo_obs.TopoObs` instances.  It can also pull in observatories 
from ``astropy``, and you can define your own.  Observatories are generally referenced when 
reading TOA files, but can also be accessed directly::

  import pint.observatory
  gbt = pint.observatory.get_observatory("gbt")  

Observatory definitions
'''''''''''''''''''''''

Observatory definitions are included in ``pint.config.runtimefile("observatories.json")``.  
To see the existing names, :func:`pint.observatory.Observatory.names_and_aliases` will 
return a dictionary giving all of the names (primary keys) and potential aliases (values).  
You can also find the full list at :ref:`Observatory List`.

The observatory data are stored in JSON format.  A simple example is::

    "gbt": {
        "tempo_code": "1",
        "itoa_code": "GB",
        "clock_file": "time_gbt.dat",
        "itrf_xyz": [
            882589.289,
            -4924872.368,
            3943729.418
        ],
        "origin": "The Robert C. Byrd Green Bank Telescope.\nThis data was obtained by Joe Swiggum from Ryan Lynch in 2021 September.\n"
    }

The observatory is defined by its name (``gbt``) and its position.  This can be given as 
geocentric coordinates in the International_Terrestrial_Reference_System_ (ITRF) through 
the ``itrf_xyz`` triple (units as ``m``), or geodetic coordinates (WGS84_ assumed) through 
``lat``, ``lon``, ``alt`` (units are ``deg`` and ``m``).  Conversion is done through 
Astropy_EarthLocation_.

Other attributes are optional.  Here we have also specified the ``tempo_code`` and 
``itoa_code``, and a human-readable ``origin`` string.

A more complex/complete example is::

  "jbroach": {
        "clock_file": [
            {
                "name": "jbroach2jb.clk",
                "valid_beyond_ends": true
            },
            "jb2gps.clk"
        ],
        "clock_fmt": "tempo2",
        "aliases": [
            "jboroach"
        ],
        "bogus_last_correction": true,
        "itrf_xyz": [
            3822625.769,
            -154105.255,
            5086486.256
        ],
        "origin": [
            "The Lovell telescope at Jodrell Bank.",
            "These are the coordinates used for VLBI as of March 2020 (MJD 58919). They are based on",
            "a fiducial position at MJD 50449 plus a (continental) drift velocity of",
            "[-0.0117, 0.0170, 0.0093] m/yr. This data was obtained from Ben Perera in September 2021.",
            "This data is for the Roach instrument - a different clock file is required for this instrument to accommodate recorded instrumental delays."
        ]
    }

Here we have included additional explicit ``aliases``, specified the clock format via 
``clock_fmt``, and specified that the last entry in the clock file is bogus (``bogus_last_correction``).  
There are two clock files included in ``clock_file``:

* ``jbroach2jb.clk`` (where we also specify that it is ``valid_beyond_ends``)
* ``jb2gps.clk``

These are combined to reference this particular telescope/instrument combination.  
For the full set of options, see :class:`~pint.observatory.topo_obs.TopoObs`.


Adding New Observatories
''''''''''''''''''''''''

In addition to modifying ``pint.config.runtimefile("observatories.json")``, there are other 
ways to add new observatories.  
**Make sure you define any new observatory before you load any TOAs.**

1. You can define them pythonically:
::

    import pint.observatory.topo_obs
    import astropy.coordinates
    newobs = pint.observatory.topo_obs.TopoObs("newobs", location=astropy.coordinates.EarthLocation.of_site("keck"), origin="another way to get Keck")

This can be done by specifying the ITRF coordinates, (``lat``, ``lon``, ``alt``), or a 
:class:`~astropy.coordinates.EarthLocation` instance.

2. You can include them just for the duration of your python session:
::

    import io
    from pint.observatory.topo_obs import load_observatories
    # GBT but no clock file
    fakeGBT = r"""{
        "gbt": {
            "tempo_code": "1",
            "itoa_code": "GB",
            "clock_file": "",
            "itrf_xyz": [
                882589.289,
                -4924872.368,
                3943729.418
            ],
            "origin": "The Robert C. Byrd Green Bank Telescope.\nThis data was obtained by Joe Swiggum from Ryan Lynch in 2021 September.\nHowever this has no clock correction"
        }
        }"""
    load_observatories(io.StringIO(fakeGBT), overwrite=True)

Note that since we are overwriting an existing observatory (rather than defining a 
completely new one) we specify ``overwrite=True``.  

3. You can define them in a different file on disk.  If you took the JSON above and 
put it into a file ``/home/user/anothergbt.json``, 
you could then do::

    export $PINT_OBS_OVERRIDE=/home/user/anothergbt.json

(or the equivalent in your shell of choice) before you start any PINT scripts.  
By default this will overwrite any existing definitions.

4. You can rely on ``astropy``.  For instance:
::

    import pint.observatory
    keck = pint.observatory.Observatory.get("keck")

will find Keck.  :func:`astropy.coordinates.EarthLocation.get_site_names` will return a list 
of potential observatories.

.. _International_Terrestrial_Reference_System: https://en.wikipedia.org/wiki/International_Terrestrial_Reference_System_and_Frame
.. _WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
.. _Astropy_EarthLocation: https://docs.astropy.org/en/stable/api/astropy.coordinates.EarthLocation.html

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

Clock corrections
'''''''''''''''''

Not all the data that PINT uses is easily accessible for programs to download.
Observatory clock corrections, for example, may need to be obtained from the
observatory through various means (often talking to a support scientist). PINT
uses a global repository, https://ipta.github.io/pulsar-clock-corrections/ to
retrieve up-to-date clock corrections for all telescopes it knows about.  PINT
should notify you when your clock files are out of date for the data you are
using; be aware that you may obtain reduced accuracy if you have old clock
correction files.

Normally, if you try to do some operation that requires unavailable clock
corrections, PINT will emit a warning but continue. If you want to be stricter,
you can specify ``limit="error"`` to various functions like
:func:`pint.toa.get_TOAs`.

If you need to check how up to date your clock corrections are, you can use
something like ``get_observatory("gbt").last_clock_correction_mjd()``: the
function :func:`pint.observatory.Observatory.last_clock_correction_mjd` checks
when clock corrections are valid for. For most telescopes, this combines the
per-telescope clock correction with PINT's global GPS and BIPM clock
corrections (both of which cannot be reliably extrapolated too far into the
future). PINT provides two convenience functions,
:func:`pint.observatory.list_last_correction_mjds` and
:func:`pint.observatory.check_for_new_clock_files_in_tempo12_repos`, that will
help you check the state of your clock corrections.

If you need clock files that are not in the global repository, perhaps more
recent versions or clock files for telescopes not included in the global
repository or specific versions for reproducibility, you have several options:

#. Set the environment variable ``PINT_CLOCK_OVERRIDE`` to point to a directory
   that contains clock files. Any clock file found there will supersede the
   version found in the global repository. You can also use
   :func:`pint.observatory.export_clock_files` to export the clock files you
   are currently using to a directory for use in this way later.

#. Modify ``src/pint/data/runtime/observatories.json`` so that the observatory you
   are interested in points to the correct clock file. (You may have to redo
   ``pip install`` for PINT to make this take effect.) If you set
   ``clock_dir="TEMPO"`` or ``clock_dir="TEMPO2"`` then PINT will look in the
   clock directory referenced by your environment variables ``$TEMPO`` or
   ``$TEMPO2`` (and nowhere else; it will no longer find clock corrections for
   this observatory that are included with PINT). You can also specify a
   specific directory as ``clock_dir="/home/burnell/clock-files/"``. Editing
   this file also allows you to choose between TEMPO- and TEMPO2-format clock
   corrections with the ``clock_fmt`` argument.

#. Create a new observatory in your own code. This involves creating a new
   :class:`pint.observatory.topo_obs.TopoObs` object like those in
   ``src/pint/data/runtime/observatories.json``. As long as this object is created
   before you read in any TOAs that need it, and as long as its name does not
   overlap with any existing observatory, you should be able to create your
   custom observatory and point the clock correction files to the right place
   as above.

Structure of Pulsar Timing Data Formats
---------------------------------------

Pulsar timing data has traditionally been divided into two parts: a list of
pulse arrival times, with sufficient metadata to work with (a ``.tim`` file),
and a description of the timing model, with parameter values, metadata, and
some fitting instructions (a ``.par`` file). These have been ad-hoc formats,
created to be easy to work with (originally) using 1980s FORTRAN code
(specifically ``TEMPO``). The advent of a second tool that works with these
files (``TEMPO2``) did not, unfortunately, come with a standardization effort,
and so files varied further in structure and were not necessarily interpreted
in the same way by both tools. As PINT is a third tool, we would prefer to
avoid introducing our own, incompatible (obviously or subtly) file formats. We
therefore formalize them here.

We are aware that not every set of timing data or parameters "in the wild" will
follow these rules. We hope to be able to lay out a clear and specific
description of these files and how they are interpreted, then elaborate on how
non-conforming files are handled, as well as how TEMPO and TEMPO2 interpret
these same files. Where possible we have tried to ensure that our description
agrees with both TEMPO and TEMPO2, but as they disagree for some existing
files, it may be necessary to offer PINT some guidance on how to interpret some
files.

Parameter files (``.par``)
''''''''''''''''''''''''''

Parameter files are text files, consisting of a collection of lines whose order
is irrelevant. Lines generally begin with an all-uppercase parameter name, then
a space-separated list of values whose interpretation depends on the parameter.

We separate parsing such a file into two steps: determining the structure of
the timing model, that is, which components make up the timing model and how
many parameters they have, then extracting the values and settings from the par
file into the model. It is the intention that in PINT these two steps can be
carried out separately, for example manually constructing a timing model from a
collection of components then feeding it parameter values from a parameter
file. It is also the intent that, unlike TEMPO and TEMPO2, PINT should be able
to clearly indicate when anomalies have occurred, for example if some parameter
was present in the parameter file but not used by any model.

Selecting timing model components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We describe a simple procedure for selecting the relevant timing model
components.

   - If the ``BINARY`` line is present in the parameter file, its value
     determines which binary model to use; if not, no binary model is used.
   - Each model component has one or more "special parameters" or families of
     parameters identified by a common prefix. If a par file contains a special
     parameter, or a known alias of one, then the timing model uses the
     corresponding component.
   - Components are organized into categories. No more than one component from
     each category may be present; some categories may be required but in
     others no component is necessary:
     - Solar system dispersion
     - Astrometry
     - Interstellar dispersion
     - Binary
     - Spin-down
     - Timing noise
   - Each component may indicate that it supersedes one or more others, that
     is, that its parameters are a superset of the previous model. In this
     case, if both are suggested by the parameter file, the component that is
     superseded is discarded. If applying this rule does not reduce the number
     of components in the category down to one, then the model is ambiguous.

We note that many parameters have "aliases", alternative names used in certain
par files. For these purposes, aliases are treated as equivalent to the special
parameters they are aliases for. Also note that not all parameters need to be
special for any component; the intent is for each component to identify a
parameter that is unique to it (or models that supersede it) and will always be
present.

We intend that PINT have facilities for managing parameter files that are
ambiguous by this definition, whether by applying heuristics or by allowing
users to clarify their intent.

This scheme as it stands has a problem: some parameter files found "in the
wild" specify equatorial coordinates for the pulsar but ecliptic values for the
proper motion. These files should certainly use ecliptic coordinates for
fitting.

Timing files (``.tim``)
'''''''''''''''''''''''

There are several commonly-used timing file formats. These are collections of
lines, but in some cases they can contain structure in the form of blocks that
are meant to be omitted from reading or have their time adjusted. We recommend
use of the most flexible format, that defined by TEMPO2 and now also supported
(to the extent that the engine permits) by TEMPO.

Fitting
-------

A very common operation with PINT is fitting a timing model to timing data.
Fundamentally this operation tries to adjust the model parameters to minimize
the residuals produced when the model is applied to a set of TOAs. The result
of this process is a set of best-fit model parameters, uncertainties on (and
correlations between) these, and residuals from this best-fit model. This is
carried out by constructing a :class:`pint.fitter.Fitter` object from
a :class:`pint.toa.TOAs` object and
a :class:`pint.models.timing_model.TimingModel` object and then running the
:func:`pint.fitter.Fitter.fit_toas` method; there are several example notebooks
that demonstrate this. Nevertheless there are some subtleties to how fitting
works in PINT that we explain here.

Timing noise and correlated errors
''''''''''''''''''''''''''''''''''

Precision pulsar timing requires a quite sophisticated model of the errors that
appear in our measurement. While each TOA has an associated uncertainty
estimate, in reality these can need to be adjusted to reflect unmodelled
sources of error; PINT (and TEMPO and TEMPO2) provide two adjustments, EFAC and
EQUAD. If these are set, and the claimed uncertainty is U, PINT will treat the
uncertainty on a data point as
:math:`\textrm{EFAC}\sqrt{U^2+\textrm{EQUAD}^2}`. We also expect a certain
amount of correlation between measurements that were taken simultaneously but
at different frequencies; this is parametrized by ECORR. More, the way we
choose to handle "red" timing noise in pulsars is to treat it as a noise
component that introduces long-term correlations in the timing measurements,
where the amount of those correlations depends on the time between measurements
and the spectrum of the timing noise. The introduction of correlations between
the errors on TOAs requires a somewhat more complicated procedure for fitting
models to TOAs, and even to simply measuring the goodness of fit of a model to
TOAs.

The most direct way of handling correlated errors between TOAs is by
constructing a covariance matrix describing all the correlations between the
measurements; a square root of this matrix can be computed using the Cholesky
decomposition, and this square root can be used to transform the fitting
problem into a conventional least-squares problem. This procedure is described
in Coles_et_al_2011_ and implemented in PINT (via the ``full_cov=True`` option to
fitters). Unfortunately this method requires a decomposition of a matrix that
is the size of the number of TOAs by the number of TOAs; this can be very
expensive in terms of memory and computation.

Fortunately, Lentati_et_al_2013_ and van_Haasteren_and_Vallisneri_2015_ describe
a method for using a low-rank approximation to the covariance matrix to remove
the need to ever construct these very large matrices; the implementation in
PINT follows the mathematics in the NANOGrav_9-year_ data analysis paper,
Appendix C.

The idea of this reduced-rank approach is to represent the correlations using
basis functions - blocks of 1s for each set of residuals grouped by ECORR, or
sinusoids for a red noise model - whose coefficients are added to the list of
parameters to be fit. The linear least-squares fitting problem is then adjusted
based on the prior estimates of the amplitudes of these basis functions (for
example the ECORR value or the amplitude of sinusoids of that frequency in the
timing model), and this modified least-squares fit is carried out. The best-fit
combinations of these noise basis functions can be subtracted from the
residuals to produce "whitened" residuals, and the goodness of fit can be
described by taking the usual chi-squared of these whitened residuals and
adding a term based on the sizes of the noise basis coefficients.

Specifically the mathematics takes an approximate solution and models the residuals as

.. math::

    \delta t = M\epsilon + Fa + Uj + n

where :math:`M` is the Jacobian matrix of the model (the derivative of each
predicted TOA with respect to each model parameter, :math:`\epsilon` is an
error in the model parameters, :math:`F` is a "Fourier design matrix", a set of
sine and cosine functions at each of a range of frequencies, :math:`a` is the
amplitudes of these basis functions in the red noise contribution, :math:`U` is
a matrix of basis functions representing the ECORR blocks, :math:`j` is their
coefficients, and :math:`n` is a vector of uncorrelated noise of amplitude
coming from the adjusted TOA uncertainties. The NANOGrav_9-year_ paper gives
expressions for the likelihood of such a representation, suitable for use in
Bayesian fitting methods, but for PINT's fitters the goal is to find the
maximum-likelihood values for :math:`\epsilon`, a corresponding set of
residuals :math:`n`, and a goodness-of-fit statistic distributed as a :math:`\chi^2`
distribution for some number of degrees of freedom.

The paper develops this, constructing additional matrices

.. math::

    N_{ij} = E_i^2(\sigma_i^2+Q_i^2)\delta_{ij}

    T = \begin{bmatrix} M & F & U \end{bmatrix}

    b = \begin{bmatrix} \epsilon \\ a \\ j \end{bmatrix}

    B = \begin{bmatrix} \infty & 0 & 0 \\ 0 & \phi & 0 \\ 0 & 0 & J \end{bmatrix}

where :math:`N` is a diagonal matrix of the adjusted TOA uncertainties, and
:math:`B` is a block matrix with diagonal matrices on the blocks; the
:math:`\infty` is a diagonal matrix of infinities (we will be using
:math:`B^{-1]`), while :math:`\phi` and :math:`J` are "weights" corresponding
to the noise basis functions' expected amplitudes.

They then construct the objects :math:`d = T^T N^{-1} \delta t` and
:math:`\Sigma = (B^{-1} + T^T N^{-1} T)`. Then they say that the maximum
likelihood values of :math:`b` and its uncertainties are given by

.. math::

    b = \Sigma^{-1} d

    \textrm{cov}(b) = \Sigma^{-1}

This is what is implemented in PINT's fitters, both the generalized
least-squares fitter for narrowband data, and the fitter used for all wideband
data (whether it has correlated errors or not).

It is perhaps worth noting that if :math:`B^{-1}` were zero or omitted, these
would be the equations for a linear least squares fit for :math:`b` to match
:math:`\delta t` with variances represented in :math:`N`. The addition of
:math:`B^{-1}` in :math:`\Sigma` is where our knowledge about the amplitudes of
the noise basis functions is applied.

The formula is not worked out in the NANOGrav_9-year_ data set paper, but if we
want a goodness-of-fit statistic for a set of model parameters that correctly
reflects both the mis-fit of the data and also the penalization of the noise
components, we need to fix all the model parameters we care about, reducing
:math:`M` to almost nothing (just a constant offset). So we compute residuals
at the model parameters of interest, then we then do the fit as above,
obtaining a maximum-likelihood :math:`b` and a set of whitened residuals
:math:`n`. We then report, as our goodness of fit,

.. math::

    \chi_G^2 = n^T N n + b^T B^{-1} b

.. _Coles_et_al_2011: https://ui.adsabs.harvard.edu/abs/2011MNRAS.418..561C/abstract
.. _Lentati_et_al_2013: https://ui.adsabs.harvard.edu/abs/2013PhRvD..87j4021L/abstract
.. _van_Haasteren_and_Vallisneri_2015: https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1170V/abstract
.. _NANOGrav_9-year: https://ui.adsabs.harvard.edu/abs/2015ApJ...813...65N/abstract

Fitting algorithms
''''''''''''''''''

PINT is designed to be able to offer several alternative algorithms to arrive
at the best-fit model. This both because fitting can be a time-consuming
process if a suboptimal algorithm is chosen, and because different kinds of
model and data require different calculations - narrowband (TOA-only) versus
wideband (TOA and DM measurements) and uncorrelated errors versus correlated
errors.

The TEMPO/TEMPO2 and default PINT fitting algorithms (:class:`pint.fitter.WidebandTOAFitter`, for example), 
leaving aside the rank-reduced case, proceed like:

1. Evaluate the model and its derivatives at the starting point :math:`x`, producing a set of residuals :math:`\delta y` and a Jacobian `M`.
2. Compute :math:`\delta x` to minimize :math:`\left| M\delta x - \delta y \right|_C`, where :math:`\left| \cdot \right|_C` is the squared amplitude of a vector with respect to the data uncertainties/covariance :math:`C`.
3. Update the starting point by :math:`\delta x`.

TEMPO and TEMPO2 can check whether the predicted improvement of chi-squared, assuming 
the linear model is correct, is enough to warrant continuing; if so, they jump back to 
step 1 unless the maximum number of iterations is reached. PINT does not contain this check.

This algorithm is the Gauss-Newton_algorithm_ for solving nonlinear
least-squares problems, and even in one-complex-dimensional cases can exhibit
convergence behavior that is literally chaotic_. For TEMPO/TEMPO2 and PINT, the
problem is that the model is never actually evaluated at the updated starting
point before committing to it; it can be invalid (ECC > 1) or the step can be
large enough that the derivative does not match the function and thus the
chi-squared value after the step can be worse than the initial chi-squared.
These issues particularly arise with poorly constrained parameters like M2 or
SINI. Users experienced with pulsar timing are frequently all too familiar with
this phenomenon and have a collection of tricks for evading it.

PINT contains a slightly more sophisticated algorithm, implemented in
:class:`pint.fitter.DownhillFitter`, that takes more careful steps:

1. Evaluate the model and its derivatives at the starting point :math:`x`, producing a set of residuals :math:`\delta y` and a Jacobian `M`.
2. Compute :math:`\delta x` to minimize :math:`\left| M\delta x - \delta y \right|_C`, where :math:`\left| \cdot \right|_C` is the squared amplitude of a vector with respect to the data uncertainties/covariance :math:`C`.
3. Set :math:`\lambda` to 1.
4. Evaluate the model at the starting point plus :math:`\lambda \delta x`. If this is invalid or worse than the starting point, divide :math:`\lambda` by two and repeat this step. If :math:`\lambda` is too small, accept the best point seen to date and exit without convergence.
5. If the model improved but only slightly with :math:`\lambda=1`, exit with convergence. If the maximum number of iterations was reached, exit without convergence. Otherwise update the starting point and return to step 1.

This ensures that PINT tries taking smaller steps if problems arise, and claims convergence 
only if a normal step worked. It does not solve the problems that arise if some parameters are 
nearly degenerate, enough to cause problems with the numerical linear algebra.

As a rule, this kind of problem is addressed with the Levenberg-Marquardt algorithm, which 
operates on the same principle of taking reduced steps when the derivative appears not to 
match the function, but does so in a way that also reduces issues with degenerate parameters; 
unfortunately it is not clear how to adapt this problem to the rank-reduced case. Nevertheless, 
PINT contains an implementation in :class:`pint.fitter.WidebandLMFitter`, but it does not perform as 
well as one might hope in practice and must be considered experimental.

.. _Gauss-Newton_algorithm: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
.. _chaotic: https://en.wikipedia.org/wiki/Newton_fractal

Coding Style
------------

We would like `PINT` to be easy to use and easy to contribute to. To
this end we'd like to ask that if you're going to contribute code or
documentation that you try to follow the below style advice. We know
that not all of the existing code does this, and it's something we'd
like to change.

For a specific listing of the rules we try to write PINT code by, please see
:ref:`CodingStyle`.

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
'''''''''''''''''
by Tim Peters::

   >>> import this
   The Zen of Python, by Tim Peters

   Beautiful is better than ugly.
   Explicit is better than implicit.
   Simple is better than complex.
   Complex is better than complicated.
   Flat is better than nested.
   Sparse is better than dense.
   Readability counts.
   Special cases aren't special enough to break the rules.
   Although practicality beats purity.
   Errors should never pass silently.
   Unless explicitly silenced.
   In the face of ambiguity, refuse the temptation to guess.
   There should be one-- and preferably only one --obvious way to do it.
   Although that way may not be obvious at first unless you're Dutch.
   Now is better than never.
   Although never is often better than *right* now.
   If the implementation is hard to explain, it's a bad idea.
   If the implementation is easy to explain, it may be a good idea.
   Namespaces are one honking great idea -- let's do more of those!

.. _guidelines: https://numpy.org/devdocs/docs/howto_document.html
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _pytest: https://docs.pytest.org/en/latest/
.. _pdb: https://docs.python.org/3/library/pdb.html
