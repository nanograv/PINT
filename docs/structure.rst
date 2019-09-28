Structure of Pulsar Timing Data Formats
=======================================

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
--------------------------

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

Timing files (``.tim``)
-----------------------

There are several commonly-used timing file formats. These are collections of
lines, but in some cases they can contain structure in the form of blocks that
are meant to be omitted from reading or have their time adjusted. We recommend
use of the most flexible format, that defined by TEMPO2 and now also supported
(to the extent that the engine permits) by TEMPO.
