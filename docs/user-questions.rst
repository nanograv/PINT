.. highlight:: shell
.. _user-questions:

How to do a number of things users have asked about
===================================================

Quick solutions to common tasks (taken from #pint and elsewhere)

How to upgrade PINT
-------------------

With ``pip``::

    pip install -U pint-pulsar

With ``conda``::

    conda update pint-pulsar


How to check out some user's particular branch for testing:
-----------------------------------------------------------

If you wish to checkout branch ``testbranch`` from user ``pintuser``::

    git checkout -b pintuser-testbranch master
    git pull https://github.com/pintuser/PINT.git testbranch

The first command makes a new local branch with name ``pintuser-testbranch`` from the ``master`` branch.  
The second pulls the remote branch from the desired user's fork into that local branch.  
You may still need to install/reinstall that branch, depending on how you have things set up 
(so ``pip install .`` or ``pip install -e .``, where the later keeps the files in-place for faster developing).


How to go to a specific version of PINT
---------------------------------------

With ``pip``::

    pip install -U pint-pulsar==0.8.4

or similar.

With ``conda``::

    conda install pint-pulsar=0.8.4

or similar.

Find data files for testing/tutorials
-------------------------------------

.. highlight:: python

The data files (par and tim) associated with the tutorials and other examples
can be located via :func:`pint.config.examplefile` (available via the
:mod:`pint.config` module)::

    import pint.config
    fullfilename = pint.config.examplefile(filename)

For example, the file ``NGC6440E.par`` from the `Time a Pulsar <https://nanograv-pint.readthedocs.io/en/latest/examples/time_a_pulsar.html>`_ notebook can be found via::

    import pint
    fullfilename = pint.config.examplefile("NGC6440E.par")


Load a par file
---------------

To load a par file::

    from pint.models import get_model
    m = get_model(parfile)


Load a tim file
---------------

To load a tim file::

    from pint.toa import get_TOAs
    t = get_TOAs(timfile)

Note that a par file may contain information, like which solar system ephemeris to use, that affects how a tim file should be loaded::

    t = get_TOAs(timfile, model=model)

Load a tim and par file together
--------------------------------

To load both::

    from pint.models import get_model_and_toas
    m, t = get_model_and_toas(parfile, timfile)


Create TOAs from an array of times
--------------------------------
A :class:`pint.toa.TOA` object represents a *single* TOA as an object that contains 
both a time and a location, along with optional information like frequency, measurement error, etc.  
So each :class:`~pint.toa.TOA` object should only contain a single time, since otherwise the location information would be ambiguous.
If you wish to create TOAs from a :class:`astropy.time.Time` object containing multiple times,
you can do::

    import numpy as np
    from astropy import units as u, constants as c
    from pint import pulsar_mjd
    from astropy.time import Time
    from pint import toa

    t = Time(np.array([55000, 56000]), scale="utc", format="pulsar_mjd")
    obs = "gbt"

    toas = toa.get_TOAs_array(t, obs)

Note that we import :mod:`pint.pulsar_mjd` to allow the 
``pulsar_mjd`` format, designed to deal properly with leap seconds.  
We use :func:`pint.toa.get_TOAs_array` to make sure clock corrections are 
applied when constructing the TOAs.  
Other information like ``errors``, ``frequencies``, and ``flags`` can be added.  
You can also merge multiple data-sets with :func:`pint.toa.merge_TOAs`


Get the red noise basis functions and the corresponding coefficients out of a PINT fitter object
------------------------------------------------------------------------------------------------

...?

Select TOAs
-----------

You can index by column name into the TOAs object, so you can do ``toas["observatory"]`` or whatever the column is called; and that's an array, so you can do ``toas["observatory"]=="arecibo"`` to get a Boolean array; and you can index with boolean arrays, so you can do ``toas[toas["observatory"]=="arecibo"]`` to get a new TOAs object referencing a subset.

Modify TOAs
-----------

The TOAs have a table with ``mjd``, ``mjd_float``, ``tdb``, and ``tdbld`` columns.  To modify them all safely and consistently the best way is to use::

    t.adjust_TOAs(dt)

where ``dt`` is an :class:`astropy.time.TimeDelta` object.  This function does not
change the pulse numbers column, if present, but does recompute ``mjd_float``,
the TDB times, and the observatory positions and velocities.


Avoid "KeyError: 'obs_jupiter_pos' error when trying to grab residuals?"
------------------------------------------------------------------------

You need to have the TOAs object compute the positions of the planets and add them to the table::

    ts.compute_posvels(ephem,planets=True)

This should be done automatically if you load your TOAs with the
:func:`pint.toa.get_TOAs`  or
:func:`pint.models.model_builder.get_model_and_toas`

Convert from ELAT/ELONG <-> RA/DEC if I have a timing model
-----------------------------------------------------------

If ``model`` is in ecliptic coordinates::

    model.as_ICRS(epoch=epoch)

which will give it to you as a model with
:class:`pint.models.astrometry.AstrometryEquatorial` components at the
requested epoch. Similarly::

    model.as_ECL(epoch=epoch)

does the same for :class:`pint.models.astrometry.AstrometryEcliptic` (with an
optional specification of the obliquity).

Convert between binary models
-----------------------------

If ``m`` is your initial model, say an ELL1 binary::

    from pint import binaryconvert
    m2 = binaryconvert.convert_binary(m, "DD")

will convert it to a DD binary.  

Some binary types need additional parameters.  For ELL1H, you can set the number of harmonics and whether to use H4 or STIGMA::

    m2 = binaryconvert.convert_binary(m, "ELL1H", NHARMS=3, useSTIGMA=True)

For DDK, you can set OM (known as ``KOM``)::

    m2 = binaryconvert.convert_binary(mDD, "DDK", KOM=12 * u.deg)

Parameter values and uncertainties will be converted.  It will also make a best-guess as to which parameters should be frozen, but 
it can still be useful to refit with the new model and check which parameters are fit.

.. note::
    The T2 model from tempo2 is not implemented, as this is a complex model that actually encapsulates several models.  The best practice is to 
    change the model to the actual underlying model (ELL1, DD, BT, etc).

These conversions can also be done on the command line using ``convert_parfile``::

    convert_parfile --binary=DD ell1.par -o dd.par

Add a jump programmatically
---------------------------

``PINT`` can handle jumps in the model outside a ``par`` file.  An example is::

    import numpy as np
    from astropy import units as u, constants as c
    from pint.models import get_model, get_model_and_toas, parameter
    from pint import fitter
    from pint.models import PhaseJump
    import pint.config

    m, t = get_model_and_toas(pint.config.examplefile("NGC6440E.par"),
                              pint.config.examplefile("NGC6440E.tim"))

    # fit the nominal model
    f = fitter.WLSFitter(toas=t, model=m)
    f.fit_toas()

    # group TOAs: find clusters with gaps of <2h
    clusters = t.get_clusters(add_column=True)

    # put in the pulse numbers based on the previous fit
    t.compute_pulse_numbers(f.model)
    # just for a test, add an offset to a set of TOAs
    t['delta_pulse_number'][clusters==3]+=3

    # now fit without a jump
    fnojump = fitter.WLSFitter(toas=t, model=m, track_mode="use_pulse_numbers")
    fnojump.fit_toas()


    # add the Jump Component to the model
    m.add_component(PhaseJump(), validate=False)

    # now add the actual jump
    # it can be keyed on any parameter that maskParameter will accept
    # here we will use a range of MJDs
    par = parameter.maskParameter(
        "JUMP",
        key="mjd",
        value=0.0,
        key_value=[t[clusters==3].get_mjds().min().value,
                   t[clusters==3].get_mjds().max().value],
        units=u.s,
        frozen=False,
        )
    m.components['PhaseJump'].add_param(par, setup=True)

    # you can also do it indirectly through the flags as:
    # m.components["PhaseJump"].add_jump_and_flags(t.table["flags"][clusters == 3])

    # and fit with a jump
    fjump = fitter.WLSFitter(toas=t, model=m, track_mode="use_pulse_numbers")
    fjump.fit_toas()

    print(f"Original chi^2 = {f.resids.calc_chi2():.2f} for {f.resids.dof} DOF")
    print(f"After adding 3 rotations to some TOAs, chi^2 = {fnojump.resids.calc_chi2():.2f} for {fnojump.resids.dof} DOF")
    print(f"Then after adding a jump to those TOAs, chi^2 = {fjump.resids.calc_chi2():.2f} for {fjump.resids.dof} DOF")
    print(f"Best-fit value of the jump is {fjump.model.JUMP1.quantity} +/- {fjump.model.JUMP1.uncertainty} ({(fjump.model.JUMP1.quantity*fjump.model.F0.quantity).decompose():.3f} +/- {(fjump.model.JUMP1.uncertainty*fjump.model.F0.quantity).decompose():.3f} rotations)")

which returns::

    Original chi^2 = 59.57 for 56 DOF
    After adding 3 rotations to some TOAs, chi^2 = 19136746.30 for 56 DOF
    Then after adding a jump to those TOAs, chi^2 = 56.60 for 55 DOF
    Best-fit value of the jump is -0.048772786677935796 s +/- 1.114921182802775e-05 s (-2.999 +/- 0.001 rotations)

showing that the offset we applied has been absorbed by the jump (plus a little extra, so chi^2 has actually improved).

See :class:`pint.models.parameter.maskParameter` documentation on the ways to select the TOAs.

Choose a fitter
---------------

Use :func:`pint.fitter.Fitter.auto`::

    f = pint.fitter.Fitter.auto(toas, model)

Include logging in a script
---------------------------

PINT now uses `loguru <https://github.com/Delgan/loguru>`_ for its logging.  To get this working within a script, try::

    import pint.logging
    from loguru import logger as log

    pint.logging.setup(sink=sys.stderr, level="WARNING", usecolors=True)

That sets up the logging and ensures it will play nicely with the rest of PINT.
You can customize the level, the destination (e.g., file, ``stderr``, ...) and
format.  The :class:`pint.logging.LogFilter`
suppresses some INFO/DEBUG messages that can clog up your screen: you can make
a custom filter as well to add/remove messages.

If you want to include a standard way to control the level using command line arguments, you can do::

    parser.add_argument(
        "--log-level",
        type=str,
        choices=("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),
        default=pint.logging.script_level,
        help="Logging level",
        dest="loglevel",
    )
    ...
    pint.logging.setup(level=args.loglevel, ...)

assuming you are using ``argparse``.  Note that ``loguru`` doesn't let you
change existing loggers: you should just remove and add (which the
:func:`pint.logging.setup` function does).

Make PINT stop reporting a particular warning
---------------------------------------------

If PINT keeps emitting a warning you know is irrelevant from somewhere inside your code, you can disable that specific warning coming from that place. For example if you are reading a par file with ``T2CMETHOD`` set but you know that's fine, you can shut off the message about ``T2CMETHOD`` while you're loading the file::

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*T2CMETHOD.*")
        model = get_model(os.path.join(datadir, "J1614-2230_NANOGrav_12yv3.wb.gls.par"))

