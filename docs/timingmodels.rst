.. _`Timing Models`:

Timing Models
=============

PINT, like TEMPO and TEMPO2, support many different ways of calculating pulse
arrival times. The key tool for doing this is a
:class:`~pint.models.timing_model.TimingModel` object, through which a whole
range of :class:`~pint.models.parameter.Parameter` are accessible. The actual
computation is done by pieces of code that live in
:class:`~pint.models.timing_model.Component`; during the parsing of a parameter
file, these are selected based on the parameters present. Binary models are
selected explicitly using the ``BINARY`` parameter, while each non-binary
component is selected if some parameter unique to it is included (for example
if ``ELAT`` is present, :class:`~pint.models.astrometry.AstrometryEcliptic` is
selected). Ambiguous or contradictory parameter files are possible, and for
these PINT raises an exception.

.. componentlist::

.. _`Supported Parameters`:

Supported Parameters
--------------------

The following table lists all the parameters that PINT can understand (along
with their aliases). The model components that use them (linked below) should
give more information about how they are interpreted.

Some parameters PINT understands have aliases - for example, the parameter PINT
calls "ECC" may also be written as "E" in parameter files. PINT will understand
these parameter files, but will always refer to this parameter internally as
"ECC". By default, though, when PINT reads a parameter file, PINT will remember
the alias that was used, and PINT will write the model out using the same
alias. This can be controlled by the ``use_alias`` attribute of
:class:`~pint.models.parameter.Parameter` objects.

PINT support for families of parameters, either specified by prefix (``F0``,
``F1``, ``F2``, ... or ``DMX_0017``, ``DMX_0123``, ...) or selecting subsets of
parameters based on flags (``JUMP -tel AO``). These are indicated in the table
with square brackets. Note that like the frequency derivatives, these families
may have units that vary in a systematic way.

Parameters can also have different types. Most are long double floating point,
with or without units; these can be specified in the usual ``1.234e5`` format,
although they also support ``1.234d5`` as well as capitalized versions for
compatibility. One or two parameters - notably ``A1DOT`` - can accept a value
scaled by ``1e12``, automatically rescaling upon read; although this is
confusing, it is necessary because TEMPO does this and so there are parameter
files "in the wild" that use this feature. Other data types allow input of
different formats, for example ``RAJ 10:23:47.67``; boolean parameters allow
``1``/``0``, ``Y``/``N``, ``T/F``, ``YES``/``NO``, ``TRUE``/``FALSE``, or
lower-case versions of these.

.. paramtable::

For comparison, there is a `table of parameters that TEMPO supports <http://tempo.sourceforge.net/ref_man_sections/binary.txt>`_.
