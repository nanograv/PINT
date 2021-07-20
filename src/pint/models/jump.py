"""Phase jumps. """
import logging

import astropy.units as u
import numpy

from pint.models.parameter import maskParameter
from pint.models.timing_model import DelayComponent, MissingParameter, PhaseComponent

log = logging.getLogger(__name__)

__all__ = ["PhaseJump"]


class DelayJump(DelayComponent):
    """Delay jumps

    This is a different kind of JUMP, conceptually affecting the TOAs directly,
    so that a 60s jump would advance the pulsar around its orbit by 60 extra
    seconds. The conventional JUMP affects only the observed phase, as you
    would expect if they are due to a redefinition of phase zero in a template.

    Parameters supported:

    .. paramtable::
        :class: pint.models.jump.DelayJump

    Note
    ----
    This component is disabled for now, since we don't have any method
    to identify the phase jumps and delay jumps.
    """

    register = False
    category = "delay_jump"

    def __init__(self):
        super().__init__()
        self.add_param(maskParameter(name="JUMP", units="second"))
        self.delay_funcs_component += [self.jump_delay]

    def setup(self):
        super().setup()
        self.jumps = []
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("JUMP"):
                self.jumps.append(mask_par)
        for j in self.jumps:
            self.register_deriv_funcs(self.d_delay_d_jump, j)

    def jump_delay(self, toas, acc_delay=None):
        """This method returns the jump delays for each toas section collected by
        jump parameters. The delay value is determined by jump parameter value
        in the unit of seconds.
        """
        tbl = toas.table
        jdelay = numpy.zeros(len(tbl))
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            mask = jump_par.select_toa_mask(toas)
            # NOTE: Currently parfile jump value has opposite sign with our
            # delay calculation.
            jdelay[mask] += -jump_par.value
        return jdelay * u.second

    def d_delay_d_jump(self, toas, jump_param, acc_delay=None):
        tbl = toas.table
        d_delay_d_j = numpy.zeros(len(tbl))
        jpar = getattr(self, jump_param)
        mask = jpar.select_toa_mask(toas)
        d_delay_d_j[mask] = -1.0
        return d_delay_d_j * u.second / jpar.units

    def print_par(self):
        result = ""
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            result += jump_par.as_parfile_line()
        return result


class PhaseJump(PhaseComponent):
    """Arbitrary jumps in pulse phase.

    A JUMP adds a constant amount to the observed phase of all the TOAs it
    applies to. JUMPs are specified using TOA flags::

        JUMP -fish carp 0.1

    will select all TOAs that have the flag ``-fish`` with the value ``carp``
    and add 0.1 seconds times ``F0`` to the phase observed at each of them.
    This would frequently be used with a flag specifying the receiver or front
    end to JUMP all TOAs coming from a particular combination of telescope and
    frequency band. Users can of course add their own flags to allow the
    selection of appropriate subsets of TOAs.

    An example of how you could JUMP a particular set of TOAs::

        >>> toa_index_list = [1,3,5]
        >>> for i in toa_index_list:
        ...     toas.table['flags'][i]['fish'] = 'carp'
        >>> model

    Jumps are specified by :class:`~pint.models.parameter.maskParameter`
    objects, so there is further documentation there on how these parameters
    and their selection criteria work. In brief, in addition to matching
    specific flags, these can also match MJD ranges (``JUMP mjd 57000 58000
    0.1``), telescopes (``JUMP tel ao 0.1``), or frequency ranges
    (``JUMP freq 1000 2000 0.1``).

    The set of TOAs matched by a particular jump, say ``JUMP1``, can be retrieved
    by :func:`~pint.models.parameter.maskParameter.select_toa_mask` as in
    ``model.JUMP1.select_toa_mask(toas)``.

    The original TEMPO supported JUMPs encoded in ``.tim`` files - a line
    containing the command JUMP indicates the beginning of a block of TOAs, and
    the next occurrence indicates the end of the block. The block of TOAs so
    defined would then have an unnamed JUMP parameter associated with it and
    fit for. When PINT encounters such a command, the affected TOAs get a flag
    ``-jump N``, where N increases by 1 for each group encountered. A par file
    can take advantage of this by including a line ``JUMP -jump 1 0.1``; such parameters
    can be automatically added with the
    :func:`~pint.models.timing_model.TimingModel.jump_flags_to_params`
    function.

    Parameters supported:

    .. paramtable::
        :class: pint.models.jump.PhaseJump

    Note
    ----

    In spite of the name, the amounts here are specified in seconds and
    converted to phase using F0. They are treated as applying to the observed
    phase, so these JUMPs do not affect where the pulsar is in its orbit when
    the TOA was observed. This is more appropriate for things like a
    redefinition in the zero of phase due to a change in pulse profile template
    than for actual time delays. Unfortunately no standard way of specifying
    that other kind of JUMPs exists, so although PINT contains code to
    implement them they cannot be used in par files.
    """

    register = True
    category = "phase_jump"

    def __init__(self):
        super().__init__()
        self.add_param(
            maskParameter(
                name="JUMP",
                units="second",
                description="Amount to jump the selected TOAs by.",
            )
        )
        self.phase_funcs_component += [self.jump_phase]

    def setup(self):
        """Set up support data structures to reflect parameters as set."""
        super().setup()
        self.jumps = {}
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("JUMP"):
                mp = getattr(self, mask_par)
                self.jumps[mp.key, mp.key_value] = mp
        for pm in self.jumps.values():
            j = pm.name
            # prevents duplicates from being added to phase_deriv_funcs
            if j in self.deriv_funcs:
                del self.deriv_funcs[j]
            self.register_deriv_funcs(self.d_phase_d_jump, j)

    def jump_phase(self, toas, delay):
        """The extra phase contributed by the JUMPs.

        This method returns the jump phase for each toas section collected by
        jump parameters. The phase value is determined by jump parameter times
        F0.

        Parameters
        ----------
        toas : pint.toa.TOAs
            The TOAs for which the JUMP is to be computed.
        delay : array-like
            Ignored.

        Returns
        -------
        astropy.units.Quantity
            The phase shift for each TOA.
        """
        tbl = toas.table
        jphase = numpy.zeros(len(tbl)) * (self.JUMP1.units * self._parent.F0.units)
        for jump, jump_par in self.jumps.items():
            mask = jump_par.select_toa_mask(toas)
            # NOTE: Currently parfile jump value has opposite sign with our
            # phase calculation.
            jphase[mask] += jump_par.quantity * self._parent.F0.quantity
        return jphase

    def d_phase_d_jump(self, toas, jump_param, delay):
        """Derivative of phase with respect to the JUMP argument.

        Parameters
        ----------
        toas : pint.toa.TOAs
            The TOAs for which the JUMP is to be computed.
        jump_param : pint.models.parameter.maskParameter
            The jump parameter to differentiate with respect to.
        delay : array-like
            Ignored.

        Returns
        -------
        astropy.units.Quantity
            The derivative of phase shift with respect to the parameter for each TOA.
        """
        tbl = toas.table
        jpar = getattr(self, jump_param)
        d_phase_d_j = numpy.zeros(len(tbl))
        mask = jpar.select_toa_mask(toas)
        d_phase_d_j[mask] = self._parent.F0.value
        return (d_phase_d_j * self._parent.F0.units).to(1 / u.second)

    def print_par(self):
        """Return a string representation of all JUMP parameters appropriate for a par file."""
        result = ""
        for jump, jump_par in self.jumps.items():
            result += jump_par.as_parfile_line()
        return result

    def get_jump_param_objects(self):
        """Returns a list of the maskParameter objects for all JUMPs."""
        jump_obs = list(self.jump.values())
        return jump_obs

    def add_jump_and_flags(self, toa_table, key="gui_jump", key_value=None):
        """Add jump object to PhaseJump and appropriate flags to TOA tables.

        Given a subset of TOAs (specified by a reference to their flags objects),
        create a new JUMP and assign flags to those TOAs so that they are selected
        by it.

        This will add a parameter to the model corresponding to::

            JUMP -gui_jump N 0 1

        where ``N`` some number not currently in use by any JUMP. This
        function will also add the flag ``-gui_jump N`` to all the TOAs in the segment
        of the table that is passed to this function.

        Parameters
        ----------
        toa_table: list
            The TOA table which must be modified. In pintk (pulsar.py), this will
            be all_toas.table["flags"][selected]
        key: str
            The name of the flag to use for the JUMP.
        key_value: str or None
            The flag value to associate with this JUMP; if not specified, find the first
            integer N not associated with a JUMP and use its string representation.

        Returns
        -------
        str
            The name of the new JUMP parameter.
        """
        in_use = set()
        for pm in self.jumps.values():
            if pm.key == "-" + key:
                in_use.add(pm.key_value)
        if key_value is None:
            i = 1
            while True:
                key_value = str(i)
                if key_value not in in_use:
                    break
                i += 1
        elif key_value in in_use:
            raise ValueError(f"A JUMP -{key} {key_value} is already present.")

        used_indices = set()
        for pm in self.jumps.values():
            used_indices.add(pm.index)
        i = 1
        while i in used_indices:
            i += 1

        param = maskParameter(
            name="JUMP",
            index=i,
            key="-" + key,
            key_value=key_value,
            value=0.0,
            units="second",
            frozen=False,
        )
        name = param.name
        for d in toa_table:
            if key in d:
                raise ValueError(
                    "The selected toa(s) overlap an existing jump. Remove all "
                    "interfering jumps before attempting to jump these toas."
                )
        self.add_param(param)
        self.setup()
        # add appropriate flags to TOA table to link jump with appropriate TOA
        for d in toa_table:
            d[key] = key_value
        return name
