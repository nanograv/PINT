"""A wrapper around pulsar functions for pintkinter to use.

This object will be shared between widgets in the main frame
and will contain the pre/post fit model, toas,
pre/post fit residuals, and other useful information.
self.selected_toas = selected toas, self.all_toas = all toas in tim file
"""
from __future__ import division, print_function

import copy
from enum import Enum

import astropy.units as u
import numpy as np
from astropy import log

import pint.fitter
import pint.models
from pint.pulsar_mjd import Time
from pint.random_models import random_models
from pint.residuals import Residuals
from pint.toa import get_TOAs

plot_labels = [
    "pre-fit",
    "post-fit",
    "mjd",
    "year",
    "orbital phase",
    "serial",
    "day of year",
    "frequency",
    "TOA error",
    "rounded MJD",
]

# Some parameters we do not want to add a fitting checkbox for:
nofitboxpars = [
    "PSR",
    "START",
    "FINISH",
    "POSEPOCH",
    "PEPOCH",
    "DMEPOCH",
    "EPHVER",
    "TZRMJD",
    "TZRFRQ",
    "TRES",
    "PLANET_SHAPIRO",
]


class Fitters(Enum):
    POWELL = 0
    WLS = 1
    GLS = 2


class Pulsar(object):
    """
    Wrapper class for a pulsar. Contains the toas, model, residuals, and fitter
    """

    def __init__(self, parfile=None, timfile=None, ephem=None):
        super(Pulsar, self).__init__()

        print("STARTING LOADING OF PULSAR %s" % str(parfile))

        if parfile is not None and timfile is not None:
            self.parfile = parfile
            self.timfile = timfile
        else:
            raise ValueError("No valid pulsar to load")

        self.prefit_model = pint.models.get_model(self.parfile)

        if ephem is not None:
            # TODO: EPHEM overwrite message?
            self.all_toas = get_TOAs(self.timfile, ephem=ephem, planets=True)
            self.prefit_model.EPHEM.value = ephem
        elif getattr(self.prefit_model, "EPHEM").value is not None:
            self.all_toas = get_TOAs(
                self.timfile, ephem=self.prefit_model.EPHEM.value, planets=True
            )
        else:
            self.all_toas = get_TOAs(self.timfile, planets=True)

        # turns pre-existing jump flags in toas.table['flags'] into parameters in parfile
        self.prefit_model.jump_flags_to_params(self.all_toas)
        self.selected_toas = copy.deepcopy(self.all_toas)
        print("prefit_model.as_parfile():")
        print(self.prefit_model.as_parfile())

        self.all_toas.print_summary()

        self.prefit_resids = Residuals(self.selected_toas, self.prefit_model)
        print(
            "RMS PINT residuals are %.3f us\n"
            % self.prefit_resids.time_resids.std().to(u.us).value
        )
        self.fitter = Fitters.WLS
        self.fitted = False

    @property
    def name(self):
        return getattr(self.prefit_model, "PSR").value

    def __getitem__(self, key):
        try:
            return getattr(self.prefit_model, key)
        except AttributeError:
            log.error(
                "Parameter %s was not found in pulsar model %s" % (key, self.name)
            )
            return None

    def __contains__(self, key):
        return key in self.prefit_model.params

    def reset_model(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.update_resids()

    def reset_TOAs(self):
        if getattr(self.prefit_model, "EPHEM").value is not None:
            self.all_toas = get_TOAs(
                self.timfile, ephem=self.prefit_model.EPHEM.value, planets=True
            )
        else:
            self.all_toas = get_TOAs(self.timfile, planets=True)
        self.selected_toas = copy.deepcopy(self.all_toas)
        self.update_resids()

    def resetAll(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.reset_TOAs()

    def update_resids(self):
        # update the pre and post fit residuals using all_toas
        self.prefit_resids = Residuals(self.all_toas, self.prefit_model)
        if self.fitted:
            self.postfit_resids = Residuals(self.all_toas, self.postfit_model)

    def orbitalphase(self):
        """
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of zeros
        """
        if "PB" not in self:
            log.warn("This is not a binary pulsar")
            return np.zeros(len(self.selected_toas))

        toas = self.selected_toas.get_mjds()

        if "T0" in self and not self["T0"].quantity is None:
            tpb = (toas.value - self["T0"].value) / self["PB"].value
        elif "TASC" in self and not self["TASC"].quantity is None:
            tpb = (toas.value - self["TASC"].value) / self["PB"].value
        else:
            log.error("Neither T0 nor TASC set")
            return np.zeros(len(toas))

        phase = np.modf(tpb)[0]
        phase[phase < 0] += 1
        return phase

    def dayofyear(self):
        """
        Return the day of the year for all the TOAs of this pulsar
        """
        t = Time(self.selected_toas.get_mjds(), format="mjd")
        year = Time(np.floor(t.decimalyear), format="decimalyear")
        return (t.mjd - year.mjd) * u.day

    def year(self):
        """
        Return the decimal year for all the TOAs of this pulsar
        """
        t = Time(self.selected_toas.get_mjds(), format="mjd")
        return (t.decimalyear) * u.year

    def write_fit_summary(self):
        """
        Summarize fitting results
        """
        if self.fitted:
            chi2 = self.postfit_resids.chi2
            wrms = np.sqrt(chi2 / self.selected_toas.ntoas)
            print("Post-Fit Chi2:\t\t%.8g us^2" % chi2)
            print("Post-Fit Weighted RMS:\t%.8g us" % wrms)
            print(
                "%19s  %24s\t%24s\t%16s  %16s  %16s"
                % (
                    "Parameter",
                    "Pre-Fit",
                    "Post-Fit",
                    "Uncertainty",
                    "Difference",
                    "Diff/Unc",
                )
            )
            print("-" * 132)
            fitparams = [
                p
                for p in self.prefit_model.params
                if not getattr(self.prefit_model, p).frozen
            ]
            for key in fitparams:
                line = "%8s " % key
                pre = getattr(self.prefit_model, key)
                post = getattr(self.postfit_model, key)
                line += "%10s  " % ("" if post.units is None else str(post.units))
                if post.quantity is not None:
                    line += "%24s\t" % pre.print_quantity(pre.quantity)
                    line += "%24s\t" % post.print_quantity(post.quantity)
                    try:
                        line += "%16.8g  " % post.uncertainty.value
                    except:
                        line += "%18s" % ""
                    try:
                        diff = post.value - pre.value
                        line += "%16.8g  " % diff
                        if pre.uncertainty is not None:
                            line += "%16.8g" % (diff / pre.uncertainty.value)
                    except:
                        pass
                print(line)
        else:
            log.warn("Pulsar has not been fitted yet!")

    def add_phase_wrap(self, selected, phase):
        """
        Add a phase wrap to selected points in the TOAs object

        Turn on pulse number tracking in the model, if it isn't already

        :param selected: boolean array to apply to toas, True = selected toa
        :param phase: phase diffeence to be added, i.e.  -0.5, +2, etc.
        """
        # Check if pulse numbers are in table already, if not, make the column
        if (
            "pn" not in self.all_toas.table.colnames
            or "pn" not in self.selected_toas.table.colnames
        ):
            self.all_toas.compute_pulse_numbers(self.prefit_model)
            self.selected_toas.compute_pulse_numbers(self.prefit_model)
        if (
            "delta_pulse_number" not in self.all_toas.table.colnames
            or "delta_pulse_number" not in self.selected_toas.table.colnames
        ):
            self.all_toas.table["delta_pulse_number"] = np.zeros(
                len(self.all_toas.get_mjds())
            )
            self.selected_toas.table["delta_pulse_number"] = np.zeros(
                len(self.selected_toas.get_mjds())
            )

        # add phase wrap
        self.all_toas.table["delta_pulse_number"][selected] += phase
        self.selected_toas.table["delta_pulse_number"] += phase

        self.update_resids()

    def add_jump(self, selected):
        """
        jump the toas selected or unjump them if already jumped

        :param selected: boolean array to apply to toas, True = selected toa
        """
        # TODO: split into two functions
        if "PhaseJump" not in self.prefit_model.components:
            # if no PhaseJump component, add one
            log.info("PhaseJump component added")
            a = pint.models.jump.PhaseJump()
            a.setup()
            self.prefit_model.add_component(a, order=-1)
            self.prefit_model.remove_param("JUMP1")
            param = pint.models.parameter.maskParameter(
                name="JUMP", index=1, key="jump", key_value=1, value=0.0, units="second"
            )
            self.prefit_model.add_param_from_top(param, "PhaseJump")
            getattr(self.prefit_model, param.name).frozen = False
            self.prefit_model.components["PhaseJump"]._parent = self.prefit_model
            if self.fitted:
                self.postfit_model.add_component(a)
            for dict1, dict2 in zip(
                self.all_toas.table["flags"][selected],
                self.selected_toas.table["flags"],
            ):
                dict1["jump"] = 1
                dict2["jump"] = 1
            return param.name
        # if gets here, has at least one jump param already
        # if doesnt overlap or cancel, add the param
        jump_nums = [
            int(dict["jump"]) if "jump" in dict.keys() else np.nan
            for dict in self.all_toas.table["flags"]
        ]
        for num in range(1, int(np.nanmax(jump_nums) + 1)):
            num = int(num)
            jump_select = [num == jump_num for jump_num in jump_nums]
            if np.array_equal(jump_select, selected):
                # if current jump exactly matches selected, remove it
                self.prefit_model.remove_param("JUMP" + str(num))
                if self.fitted:
                    self.postfit_model.remove_param("JUMP" + str(num))
                for dict1, dict2 in zip(
                    self.all_toas.table["flags"][selected],
                    self.selected_toas.table["flags"],
                ):
                    if "jump" in dict1.keys() and dict1["jump"] == num:
                        del dict1["jump"]  # somehow deletes from both
                nums_subset = range(num + 1, int(np.nanmax(jump_nums) + 1))
                for n in nums_subset:
                    # iterate through jump params and rename them so that they are always in numerical order starting with JUMP1
                    n = int(n)
                    for dict in self.all_toas.table["flags"]:
                        if "jump" in dict.keys() and dict["jump"] == n:
                            dict["jump"] = n - 1
                    param = pint.models.parameter.maskParameter(
                        name="JUMP",
                        index=int(n - 1),
                        key="jump",
                        key_value=int(n - 1),
                        value=getattr(self.prefit_model, "JUMP" + str(n)).value,
                        units="second",
                    )
                    self.prefit_model.add_param_from_top(param, "PhaseJump")
                    getattr(self.prefit_model, param.name).frozen = getattr(
                        self.prefit_model, "JUMP" + str(n)
                    ).frozen
                    self.prefit_model.remove_param("JUMP" + str(n))
                    if self.fitted:
                        self.postfit_model.add_param_from_top(param, "PhaseJump")
                        getattr(self.postfit_model, param.name).frozen = getattr(
                            self.postfit_model, "JUMP" + str(n)
                        ).frozen
                        self.postfit_model.remove_param("JUMP" + str(n))
                if "JUMP1" not in self.prefit_model.params:
                    # remove PhaseJump component if no jump params
                    comp_list = getattr(self.prefit_model, "PhaseComponent_list")
                    for item in comp_list:
                        if isinstance(item, pint.models.jump.PhaseJump):
                            comp_list.remove(item)
                            break
                    self.prefit_model.setup_components(comp_list)
                    if self.fitted:
                        self.postfit_model.setup_components(comp_list)
                else:
                    self.prefit_model.components["PhaseJump"].setup()
                    if self.fitted:
                        self.postfit_model.components["PhaseJump"].setup()
                log.info("removed param", "JUMP" + str(num))
                return jump_select
            elif True in [a and b for a, b in zip(jump_select, selected)]:
                # if current jump overlaps selected, raise and error and end
                log.warn(
                    "The selected toa(s) overlap an existing jump. Remove all interfering jumps before attempting to jump these toas."
                )
                return None
        # if here, then doesn't overlap or match anything
        for dict1, dict2 in zip(
            self.all_toas.table["flags"][selected], self.selected_toas.table["flags"]
        ):
            dict1["jump"] = int(np.nanmax(jump_nums)) + 1
            dict2["jump"] = int(np.nanmax(jump_nums)) + 1
        param = pint.models.parameter.maskParameter(
            name="JUMP",
            index=int(np.nanmax(jump_nums)) + 1,
            key="jump",
            key_value=int(np.nanmax(jump_nums)) + 1,
            value=0.0,
            units="second",
            aliases=["JUMP"],
        )
        self.prefit_model.add_param_from_top(param, "PhaseJump")
        getattr(self.prefit_model, param.name).frozen = False
        self.prefit_model.components["PhaseJump"].setup()
        if (
            self.fitted
            and not self.prefit_model.components["PhaseJump"]
            == self.postfit_model.components["PhaseJump"]
        ):
            self.postfit_model.add_param_from_top(param, "PhaseJump")
            getattr(self.postfit_model, param.name).frozen = False
            self.postfit_model.components["PhaseJump"].setup()
        return param.name

    def fit(self, selected, iters=1):
        """
        Run a fit using the specified fitter
        """
        # Select all the TOAs if none are explicitly set
        if not any(selected):
            selected = ~selected

        """JUMP check, TODO: put in fitter?"""
        if "PhaseJump" in self.prefit_model.components:
            # if attempted fit (selected)
            # A) contains only jumps, don't do the fit and return an error
            # B) excludes a jump, turn that jump off
            # C) partially contains a jump, redefine that jump only with the overlap
            fit_jumps = []
            for param in self.prefit_model.params:
                if getattr(
                    self.prefit_model, param
                ).frozen == False and param.startswith("JUMP"):
                    fit_jumps.append(int(param[4:]))
            jumps = [
                True if "jump" in dict.keys() and dict["jump"] in fit_jumps else False
                for dict in self.selected_toas.table["flags"]
            ]
            if all(jumps):
                log.warn(
                    "toas being fit must not all be jumped. Remove or uncheck at least one jump in the selected toas before fitting."
                )
                return None
            sel_jump_nums = [
                dict["jump"] if "jump" in dict.keys() else np.nan
                for dict in self.selected_toas.table["flags"]
            ]
            full_jump_nums = [
                dict["jump"] if "jump" in dict.keys() else np.nan
                for dict in self.all_toas.table["flags"]
            ]
            for num in range(1, int(np.nanmax(full_jump_nums) + 1)):
                num = int(num)
                if num not in sel_jump_nums:
                    getattr(self.prefit_model, "JUMP" + str(num)).frozen = True
                    continue
                jump_select = [num == jump_num for jump_num in full_jump_nums]
                overlap = [a and b for a, b in zip(jump_select, selected)]
                # remove the jump flags for that num
                for dict in self.all_toas.table["flags"]:
                    if "jump" in dict.keys() and dict["jump"] == num:
                        del dict["jump"]
                # re-add the jump using overlap as 'selected'
                for dict in self.all_toas.table["flags"][overlap]:
                    dict["jump"] = num

        if self.fitted:
            self.prefit_model = self.postfit_model
            self.prefit_resids = self.postfit_resids

        if self.fitter == Fitters.POWELL:
            fitter = pint.fitter.PowellFitter(self.selected_toas, self.prefit_model)
        elif self.fitter == Fitters.WLS:
            fitter = pint.fitter.WLSFitter(self.selected_toas, self.prefit_model)
        elif self.fitter == Fitters.GLS:
            fitter = pint.fitter.GLSFitter(self.selected_toas, self.prefit_model)
        chi2 = self.prefit_resids.chi2
        wrms = np.sqrt(chi2 / self.selected_toas.ntoas)
        print("Pre-Fit Chi2:\t\t%.8g us^2" % chi2)
        print("Pre-Fit Weighted RMS:\t%.8g us" % wrms)

        fitter.fit_toas(maxiter=1)
        self.postfit_model = fitter.model
        self.postfit_resids = Residuals(
            self.all_toas, self.postfit_model, set_pulse_nums=True
        )
        self.fitted = True
        self.write_fit_summary()

        # TODO: set pulse nums above not working to reset delta pulse nums, have to force it here
        # self.fulltoas.table['delta_pulse_numbers'] = np.zeros(self.fulltoas.ntoas)
        self.selected_toas.table["delta_pulse_number"] = np.zeros(
            self.selected_toas.ntoas
        )

        # plot the prefit without jumps
        pm_no_jumps = copy.deepcopy(self.postfit_model)
        for param in pm_no_jumps.params:
            if param.startswith("JUMP"):
                getattr(pm_no_jumps, param).value = 0.0
                getattr(pm_no_jumps, param).frozen = True
        self.prefit_resids_no_jumps = Residuals(
            self.all_toas, pm_no_jumps, set_pulse_nums=True
        )

        f = copy.deepcopy(fitter)
        no_jumps = [
            False if "jump" in dict.keys() else True for dict in f.toas.table["flags"]
        ]
        f.toas.select(no_jumps)

        selectedMJDs = self.selected_toas.get_mjds()
        if all(no_jumps):
            q = list(self.all_toas.get_mjds())
            index = q.index(
                [i for i in self.all_toas.get_mjds() if i > selectedMJDs.min()][0]
            )
            rs_mean = (
                Residuals(self.all_toas, f.model, set_pulse_nums=True)
                .phase_resids[index : index + len(selectedMJDs)]
                .mean()
            )
        else:
            rs_mean = self.prefit_resids_no_jumps.phase_resids[no_jumps].mean()

        # determines how far on either side fake toas go
        # TODO: hard limit on how far fake toas can go --> can get clkcorr
        # errors if go before GBT existed, etc.
        minMJD, maxMJD = selectedMJDs.min(), selectedMJDs.max()
        spanMJDs = maxMJD - minMJD
        if spanMJDs < 30 * u.d:
            redge = ledge = 4
            npoints = 400
        elif spanMJDs < 90 * u.d:
            redge = ledge = 2
            npoints = 300
        elif spanMJDs < 200 * u.d:
            redge = ledge = 1
            npoints = 300
        elif spanMJDs < 400 * u.d:
            redge = ledge = 0.5
            npoints = 200
        else:
            redge = ledge = 0.2
            npoints = 150
        # Check to see if too recent
        nowish = (Time.now().mjd - 40) * u.d
        if maxMJD + spanMJDs * redge > nowish:
            redge = (nowish - maxMJD) / spanMJDs
            if redge < 0.0:
                redge = 0.0
        f_toas, rs, mrands = random_models(
            f,
            rs_mean=rs_mean,
            redge_multiplier=redge,
            ledge_multiplier=ledge,
            npoints=npoints,
            iter=10,
        )
        self.random_resids = rs
        self.fake_toas = f_toas
