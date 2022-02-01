"""A wrapper around pulsar functions for pintkinter to use.

This object will be shared between widgets in the main frame
and will contain the pre/post fit model, toas,
pre/post fit residuals, and other useful information.
self.selected_toas = selected toas, self.all_toas = all toas in tim file
"""
import copy
import logging
from enum import Enum

import astropy.units as u
import numpy as np

import pint.fitter
import pint.models
from pint.pulsar_mjd import Time
from pint.simulation import make_fake_toas_uniform, calculate_random_models
from pint.residuals import Residuals
from pint.toa import get_TOAs
from pint.phase import Phase

log = logging.getLogger(__name__)

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


class Pulsar:
    """Wrapper class for a pulsar.

    Contains the toas, model, residuals, and fitter
    """

    def __init__(self, parfile=None, timfile=None, ephem=None):
        super(Pulsar, self).__init__()

        log.info("STARTING LOADING OF PULSAR %s" % str(parfile))

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
        # TODO: fix jump_flags_to_params
        self.prefit_model.jump_flags_to_params(self.all_toas)
        # adds flags to toas.table for existing jump parameters from .par file
        if "PhaseJump" in self.prefit_model.components:
            self.prefit_model.jump_params_to_flags(self.all_toas)
        self.selected_toas = copy.deepcopy(self.all_toas)
        print("prefit_model.as_parfile():")
        print(self.prefit_model.as_parfile())

        self.all_toas.print_summary()

        self.prefit_resids = Residuals(self.selected_toas, self.prefit_model)
        print(
            "RMS PINT residuals are %.3f us\n"
            % self.prefit_resids.rms_weighted().to(u.us).value
        )
        self.fitter = Fitters.WLS
        self.fitted = False
        self.use_pulse_numbers = False

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
        self.use_pulse_numbers = False
        self.reset_TOAs()

    def update_resids(self):
        # update the pre and post fit residuals using all_toas
        if self.use_pulse_numbers:
            self.prefit_resids = Residuals(
                self.all_toas, self.prefit_model, track_mode="use_pulse_numbers"
            )
            if self.fitted:
                self.postfit_resids = Residuals(
                    self.all_toas, self.postfit_model, track_mode="use_pulse_numbers"
                )
        else:
            self.prefit_resids = Residuals(self.all_toas, self.prefit_model)
            if self.fitted:
                self.postfit_resids = Residuals(self.all_toas, self.postfit_model)

    def orbitalphase(self):
        """
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of unitless quantities of zeros
        """
        if not self.prefit_model.is_binary:
            log.warn("This is not a binary pulsar")
            return u.Quantity(np.zeros(self.selected_toas.ntoas))

        toas = self.selected_toas
        if self.fitted:
            phase = self.postfit_model.orbital_phase(toas, anom="mean")
        else:
            phase = self.prefit_model.orbital_phase(toas, anom="mean")
        return phase / (2 * np.pi * u.rad)

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
            wrms = self.postfit_resids.rms_weighted()
            print("Post-Fit Chi2:\t\t%.8g" % chi2)
            print("Post-Fit Weighted RMS:\t%.8g us" % wrms.to(u.us).value)
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
                    line += "%24s\t" % pre.str_quantity(pre.quantity)
                    line += "%24s\t" % post.str_quantity(post.quantity)
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
            "pulse_number" not in self.all_toas.table.colnames
            or "pulse_number" not in self.selected_toas.table.colnames
        ):
            if self.fitted:
                self.all_toas.compute_pulse_numbers(self.postfit_model)
                self.selected_toas.compute_pulse_numbers(self.postfit_model)
            else:
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

        self.use_pulse_numbers = True

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
            self.prefit_model.add_component(a)
            retval = self.prefit_model.add_jump_and_flags(
                self.all_toas.table["flags"][selected]
            )
            if self.fitted:
                self.postfit_model.add_component(a)
            return retval
        # if gets here, has at least one jump param already
        # if doesnt overlap or cancel, add the param
        numjumps = self.prefit_model.components["PhaseJump"].get_number_of_jumps()
        if numjumps == 0:
            log.warn(
                "There are no jumps (maskParameter objects) in PhaseJump. Please delete the PhaseJump object and try again. "
            )
            return None
        # delete jump if perfectly overlaps any existing jump
        for num in range(1, numjumps + 1):
            # create boolean array corresponding to TOAs to be jumped
            toas_jumped = [
                "jump" in dict.keys() and str(num) in dict["jump"]
                for dict in self.all_toas.table["flags"]
            ]
            if np.array_equal(toas_jumped, selected):
                # if current jump exactly matches selected, remove it
                self.prefit_model.delete_jump_and_flags(
                    self.all_toas.table["flags"], num
                )
                if self.fitted:
                    self.postfit_model.delete_jump_and_flags(None, num)
                log.info("removed param", "JUMP" + str(num))
                return toas_jumped
        # if here, then doesn't match anything
        # add jump flags to selected TOAs at their perspective indices in the TOA tables
        retval = self.prefit_model.add_jump_and_flags(
            self.all_toas.table["flags"][selected]
        )
        if (
            self.fitted
            and not self.prefit_model.components["PhaseJump"]
            == self.postfit_model.components["PhaseJump"]
        ):
            param = self.prefit_model.components[
                "PhaseJump"
            ].get_jump_param_objects()  # array of jump objects
            self.postfit_model.add_param_from_top(
                param[-1], "PhaseJump"
            )  # add last (newest) jump
            getattr(self.postfit_model, param[-1].name).frozen = False
            self.postfit_model.components["PhaseJump"].setup()
        return retval

    def fit(self, selected, iters=1):
        """
        Run a fit using the specified fitter
        """
        # Select all the TOAs if none are explicitly set
        if not any(selected):
            selected = ~selected

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
        wrms = self.prefit_resids.rms_weighted()
        print("Pre-Fit Chi2:\t\t%.8g" % chi2)
        print("Pre-Fit Weighted RMS:\t%.8g us" % wrms.to(u.us).value)

        fitter.fit_toas(maxiter=1)
        self.postfit_model = fitter.model
        self.postfit_resids = Residuals(self.all_toas, self.postfit_model)
        self.fitted = True
        self.write_fit_summary()

        # TODO: delta_pulse_numbers need some work. They serve both for PHASE and -padd functions from the TOAs
        # as well as for phase jumps added manually in the GUI. They really should not be zeroed out here because
        # that will wipe out preexisting values
        self.all_toas.table["delta_pulse_numbers"] = np.zeros(self.all_toas.ntoas)
        self.selected_toas.table["delta_pulse_number"] = np.zeros(
            self.selected_toas.ntoas
        )

        # plot the prefit without jumps
        pm_no_jumps = copy.deepcopy(self.prefit_model)
        for param in pm_no_jumps.params:
            if param.startswith("JUMP"):
                getattr(pm_no_jumps, param).value = 0.0
                getattr(pm_no_jumps, param).frozen = True
        self.prefit_resids_no_jumps = Residuals(self.selected_toas, pm_no_jumps)

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
                Residuals(self.all_toas, f.model)
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
            redge = ledge = 1.0
            npoints = 250
        # Check to see if too recent
        nowish = (Time.now().mjd - 40) * u.d
        if maxMJD + spanMJDs * redge > nowish:
            redge = (nowish - maxMJD) / spanMJDs
            if redge < 0.0:
                redge = 0.0
        f_toas = make_fake_toas_uniform(
            minMJD - spanMJDs * ledge, maxMJD + spanMJDs * redge, npoints, f.model
        )
        rs = calculate_random_models(
            f, f_toas, Nmodels=10, keep_models=False, return_time=True
        )

        # subtract the mean residual of each random model from the respective residual set
        # based ONLY on the mean of the random residuals in the real data range
        start_index = np.where(abs(f_toas.get_mjds() - minMJD) < 1 * u.d)
        end_index = np.where(abs(f_toas.get_mjds() - maxMJD) < 1 * u.d)
        for i in range(len(rs)):
            # use start_index[0][0] since np.where returns np.array([], dtype), extract index from list in array
            rs_mean = rs[i][start_index[0][0] : end_index[0][0]].mean()
            rs[i][:] = [resid - rs_mean for resid in rs[i]]

        self.random_resids = rs
        self.fake_toas = f_toas

    def fake_year(self):
        """
        Function to support plotting of random models on multiple x-axes.
        Return the decimal year for all the TOAs of this pulsar
        """
        t = Time(self.fake_toas.get_mjds(), format="mjd")
        return (t.decimalyear) * u.year
