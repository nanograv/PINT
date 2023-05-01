"""Timing model parameters encapsulated as objects.

Defines Parameter class for timing model parameters. These objects keep
track of values, uncertainties, and units. They can hold a variety of
types, both numeric - python floats and numpy longdoubles - and other -
string, angles, times.

These classes also contain code to allow them to read and write values
in both exact and human-readable forms, as well as detecting when they
have occurred in ``.par`` files.

One major complication is that timing models can often have variable
numbers of parameters: for example the ``DMX`` family of parameters
can have one parameter for each group of TOAs in the input, allowing
potentially very many. These are handled in two separate ways, as "prefix
parameters" (:class:`pint.models.parameter.prefixParameter`) and
"mask parameters" (:class:`pint.models.parameter.maskParameter`)
depending on how they occur in the ``.par`` and ``.tim`` files.

See :ref:`Supported Parameters` for an overview, including a table of all the
parameters PINT understands.

"""

from pint.models.parameter.angle_parameter import AngleParameter
from pint.models.parameter.bool_parameter import boolParameter
from pint.models.parameter.float_parameter import floatParameter
from pint.models.parameter.func_parameter import funcParameter
from pint.models.parameter.int_parameter import intParameter
from pint.models.parameter.mask_parameter import maskParameter
from pint.models.parameter.mjd_parameter import MJDParameter
from pint.models.parameter.pair_parameter import pairParameter
from pint.models.parameter.param_base import Parameter, parfile_formats
from pint.models.parameter.prefix_parameter import prefixParameter
from pint.models.parameter.str_parameter import strParameter
