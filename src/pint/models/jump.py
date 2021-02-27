"""Phase jumps. """
# phase_jump.py
# Defines PhaseJump timing model class
import astropy.units as u
import numpy

from pint.models.parameter import maskParameter
from pint.models.timing_model import DelayComponent, MissingParameter, PhaseComponent
from astropy import log


class DelayJump(DelayComponent):
    """Phase jumps

    Note
    ----
    This component is disabled for now, since we don't have any method
    to identify the phase jumps and delay jumps.
    """

    register = False
    category = "delay_jump"

    def __init__(self):
        super(DelayJump, self).__init__()
        self.add_param(maskParameter(name="JUMP", units="second"))
        self.delay_funcs_component += [self.jump_delay]

    def setup(self):
        super(DelayJump, self).setup()
        self.jumps = []
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("JUMP"):
                self.jumps.append(mask_par)
        for j in self.jumps:
            self.register_deriv_funcs(self.d_delay_d_jump, j)

    def validate(self):
        super(DelayJump, self).validate()

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
    """A class to implement phase jumps."""

    register = True
    category = "phase_jump"

    def __init__(self):
        super(PhaseJump, self).__init__()
        self.add_param(maskParameter(name="JUMP", units="second"))
        self.phase_funcs_component += [self.jump_phase]

    def setup(self):
        super(PhaseJump, self).setup()
        self.jumps = []
        for mask_par in self.get_params_of_type("maskParameter"):
            if mask_par.startswith("JUMP"):
                self.jumps.append(mask_par)
        for j in self.jumps:
            # prevents duplicates from being added to phase_deriv_funcs
            if j in self.deriv_funcs.keys():
                del self.deriv_funcs[j]
            self.register_deriv_funcs(self.d_phase_d_jump, j)

    def validate(self):
        super(PhaseJump, self).validate()

    def jump_phase(self, toas, delay):
        """This method returns the jump phase for each toas section collected by
        jump parameters. The phase value is determined by jump parameter times
        F0.
        """
        tbl = toas.table
        jphase = numpy.zeros(len(tbl)) * (self.JUMP1.units * self._parent.F0.units)
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            mask = jump_par.select_toa_mask(toas)
            # NOTE: Currently parfile jump value has opposite sign with our
            # phase calculation.
            jphase[mask] += jump_par.quantity * self._parent.F0.quantity
        return jphase

    def d_phase_d_jump(self, toas, jump_param, delay):
        tbl = toas.table
        jpar = getattr(self, jump_param)
        d_phase_d_j = numpy.zeros(len(tbl))
        mask = jpar.select_toa_mask(toas)
        d_phase_d_j[mask] = self._parent.F0.value
        return (d_phase_d_j * self._parent.F0.units).to(1 / u.second)

    def print_par(self):
        result = ""
        for jump in self.jumps:
            jump_par = getattr(self, jump)
            result += jump_par.as_parfile_line()
        return result

    def get_number_of_jumps(self):
        """Returns the number of jumps contained in this PhaseJump object."""
        return len(self.jumps)

    def get_jump_param_objects(self):
        """
        Returns a list of the maskParameter objects representing the jumps
        in this PhaseJump object.
        """
        jump_obs = [getattr(self, jump) for jump in self.jumps]
        return jump_obs

    def jump_params_to_flags(self, toas):
        """Take jumps created from .par file and add appropriate flags to toa table.

        This function was made specifically with pintk in mind for a way to properly
        load jump flags at the same time a .par file with jumps is loaded (like how
        jump_flags_to_params loads jumps from .tim files).

        Parameters
        ----------
        toas: TOAs object
            The TOAs which contain the TOA table to be modified
        """
        # for every jump, set appropriate flag for TOAs it jumps
        for jump_par in self.get_jump_param_objects():
            # find TOAs jump applies to
            mask = jump_par.select_toa_mask(toas)
            # apply to dictionaries
            for dict in toas.table["flags"][mask]:
                if "jump" in dict.keys():
                    # check if jump flag already added - don't add flag twice
                    if jump_par.index in dict["jump"]:
                        continue
                    dict["jump"].append(jump_par.index)  # otherwise, add jump flag
                else:
                    dict["jump"] = [jump_par.index]

    def add_jump_and_flags(self, toa_tables):
        """Add jump object to PhaseJump and appropriate flags to TOA tables (helper
        function for pintk).

        Primarily to be used when applying a jump through pintk to TOAs - since these
        jumps don't have keys that match to preexisting flags in the TOA tables,
        we must add the flags when adding the jump.

        Parameters
        ----------
        toa_tables: list object
            The TOA tables which must be modified. In pintk (pulsar.py), this will
            be a list of TOA tables:
            [all_toas.table["flags"][selected], selected_toas.table["flags"]]
        """
        ind = None  # index of jump
        # check if this is first jump added
        if len(self.jumps) == 0 or (
            len(self.jumps) == 1 and getattr(self, "JUMP1").key == None
        ):
            param = maskParameter(
                name="JUMP",
                index=1,
                key="-gui_jump",
                key_value=1,
                value=0.0,
                units="second",
                frozen=False,
                aliases=["JUMP"],
            )
            self.add_param(param)
            ind = 1
        # otherwise add on jump with next index
        else:
            # first, search for TOAs already jumped in inputted selection - pintk does not allow jumps added through GUI to overlap with existing jumps
            for dict in toa_tables[0]:  # just need to check overall toa table
                if "jump" in dict.keys():
                    log.warning(
                        "The selected toa(s) overlap an existing jump. Remove all interfering jumps before attempting to jump these toas."
                    )
                    return
            param = maskParameter(
                name="JUMP",
                index=len(self.jumps) + 1,
                key="-gui_jump",
                key_value=1,
                value=0.0,
                units="second",
                frozen=False,
                aliases=["JUMP"],
            )
            self.add_param(param)
            ind = param.index
        self.setup()
        # add appropriate flags to TOA table(s) to link jump with appropriate TOA
        for dict1, dict2 in zip(toa_tables[0], toa_tables[1]):
            if "jump" in dict1.keys():
                dict1["jump"].append(ind)  # toa can have multiple jumps
            else:
                dict1["jump"] = [ind]
            dict1["gui_jump"] = ind  # toa can only have one gui_jump
            if "jump" in dict2.keys():
                dict2["jump"].append(ind)
            else:
                dict2["jump"] = [ind]
            dict2["gui_jump"] = ind

    '''
    def delete_jump_and_flags(self, toa_tables, toa_indeces, jump_num):
        """Delete jump object from PhaseJump and remove its flags from TOA tables
        (helper function for pintk).

        Parameters
        ----------
        toa_tables: list object
            The TOA tables which must be modified. In pintk (pulsar.py), this will
            be a list of TOA tables:
            [all_toas.table["flags"], selected_toas.table["flags"]]
        toa_indeces: list object
            A list of ints corresponding to the indeces of the selected TOAs (in the GUI).
        jump_num: int
            Specifies the index of the jump to be deleted.
        """
        # remove jump of specified index
        self.remove_param("JUMP" + str(jump_num))

        # remove jump flags from selected TOA tables
        for dict1, dict2 in zip(toa_tables[0][toa_indeces], toa_tables[1]):
            if "jump" in dict1.keys() and jump_num in dict1["jump"]:
                if len(dict1["jump"]) == 1:
                    del dict1["jump"]
                else:
                    dict1["jump"].remove(jump_num)
            if "jump" in dict2.keys() and jump_num in dict2["jump"]:
                if len(dict2["jump"]) == 1:
                    del dict2["jump"]
                else:
                    dict2["jump"].remove(jump_num)
            if "gui_jump" in dict1.keys() and dict1["gui_jump"] == jump_num:
                del dict1["gui_jump"]
            if "gui_jump" in dict2.keys() and dict2["gui_jump"] == jump_num:
                del dict2["gui_jump"]

        for dict1 in toa_tables[0]:
            # renumber jump flags at higher jump indeces in whole TOA table
            if "jump" in dict1.keys():
                dict1["jump"] = [
                    num - 1 if num > jump_num else num for num in dict1["jump"]
                ]
            if "gui_jump" in dict1.keys() and dict1["gui_jump"] > jump_num:
                cur_val = dict1["gui_jump"]
                dict1["gui_jump"] = cur_val - 1

        # reindex jump objects
        if len(self.jumps) == 0:

        for i in range(jump_num + 1, len(self.jumps) + 2):
            cur_jump = getattr(self, "JUMP" + str(i))
            new_jump = cur_jump.new_param(index=(i - 1), copy_all=True)
            self.add_param(new_jump)
            self.remove_param(cur_jump.name)
    '''
