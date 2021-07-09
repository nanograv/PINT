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

Supported Parameters
--------------------

The following table lists all the parameters that PINT can understand (along with their aliases). The model components that use them (linked below) should give more information about how they are interpreted.

.. paramtable::

