"""A wrapper around pulsar functions for `pintk` to use.

This object will be shared between widgets in the main frame
and will contain the pre/post fit model, toas,
pre/post fit residuals, and other useful information.
self.selected_toas = selected toas, self.all_toas = all toas in tim file
"""
import copy

import astropy.units as u
import numpy as np

import pint.fitter
import pint.models
from pint.pulsar_mjd import Time
from pint.simulation import (
    make_fake_toas_uniform,
    calculate_random_models,
    zero_residuals,
)
from pint.residuals import Residuals
from pint.toa import get_TOAs, merge_TOAs
from pint.utils import FTest, akaike_information_criterion

import pint.logging
from loguru import logger as log


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
    "model DM",
    "WB DM",
    "WB DM res",
    "WB DM err",
    "elongation",
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


class Pulsar:
    """Wrapper class for a pulsar.

    Contains the toas, model, residuals, and fitter
    """

    def __init__(self, parfile=None, timfile=None, ephem=None, fitter="GLSFitter"):
        super().__init__()

        log.info(f"Loading pulsar parfile: {str(parfile)}")

        if parfile is None or timfile is None:
            raise ValueError("No valid pulsar model and/or TOAs to load")

        self.parfile = parfile
        self.timfile = timfile
        self.prefit_model = pint.models.get_model(self.parfile)

        if ephem is not None:
            log.info(
                f"Overriding model ephemeris {self.prefit_model.EPHEM.value} with {ephem}"
            )
            self.prefit_model.EPHEM.value = ephem
        self.all_toas = get_TOAs(self.timfile, model=self.prefit_model, usepickle=True)
        self.all_toas.table.sort("index")
        self.all_toas.get_clusters(add_column=True)
        # Make sure that if we used a model, that any phase jumps from
        # the parfile have their flags updated in the TOA table
        if "PhaseJump" in self.prefit_model.components:
            self.prefit_model.jump_params_to_flags(self.all_toas)
        # turns pre-existing jump flags in toas.table['flags'] into parameters in parfile
        self.prefit_model.jump_flags_to_params(self.all_toas)
        self.selected_toas = copy.deepcopy(self.all_toas)
        print("The prefit model as a parfile:")
        print(self.prefit_model.as_parfile())
        # adds extra prefix params for fitting
        self.add_model_params()

        self.all_toas.print_summary()

        self.prefit_resids = Residuals(self.all_toas, self.prefit_model)
        self.selected_prefit_resids = self.prefit_resids
        print(
            "RMS pre-fit PINT residuals are %.3f us\n"
            % self.prefit_resids.rms_weighted().to(u.us).value
        )
        # Set of indices from original list that are deleted
        self.deleted = set([])
        if fitter == "notdownhill":
            self.fit_method = self.getDefaultFitter(downhill=False)
            log.info(
                f"Since wideband={self.all_toas.wideband} and correlated={self.prefit_model.has_correlated_errors}, selecting fitter={self.fit_method}"
            )
        elif fitter == "downhill":
            self.fit_method = self.getDefaultFitter(downhill=True)
            log.info(
                f"Since wideband={self.all_toas.wideband} and correlated={self.prefit_model.has_correlated_errors}, selecting Downhill fitter={self.fit_method}"
            )
        else:
            self.fit_method = fitter
        self.fitter = None
        self.fitted = False
        self.stashed = None  # for temporarily stashing some TOAs
        self.faketoas1 = None  # for random models
        self.faketoas = None  # for random models
        self.use_pulse_numbers = False

    @property
    def name(self):
        return getattr(self.prefit_model, "PSR").value

    def __getitem__(self, key):
        try:
            return getattr(self.prefit_model, key)
        except AttributeError:
            log.error(f"Parameter {key} was not found in pulsar model {self.name}")
            return None

    def __contains__(self, key):
        return key in self.prefit_model.params

    def reset_model(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.add_model_params()
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.update_resids()

    def reset_TOAs(self):
        self.all_toas = get_TOAs(self.timfile, model=self.prefit_model, usepickle=True)
        # Make sure that if we used a model, that any phase jumps from
        # the parfile have their flags updated in the TOA table
        if "PhaseJump" in self.prefit_model.components:
            self.prefit_model.jump_params_to_flags(self.all_toas)
        # turns pre-existing jump flags in toas.table['flags'] into parameters in parfile
        self.prefit_model.jump_flags_to_params(self.all_toas)
        self.selected_toas = copy.deepcopy(self.all_toas)
        self.deleted = set([])
        self.stashed = None
        self.update_resids()

    def resetAll(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.use_pulse_numbers = False
        self.reset_TOAs()

    def _delete_TOAs(self, toa_table):
        del_inds = np.in1d(toa_table["index"], np.array(list(self.deleted)))
        return toa_table[~del_inds] if del_inds.sum() < len(toa_table) else None

    def delete_TOAs(self, indices, selected):
        # note: indices should be a list or an array
        self.deleted |= set(indices)  # update the deleted indices
        if selected is not None:
            self.selected_toas.table = self._delete_TOAs(self.selected_toas.table)
        # Now delete from all_toas
        self.all_toas.table = self._delete_TOAs(self.all_toas.table)
        if self.selected_toas.table is None:  # all selected were deleted
            self.selected_toas = copy.deepcopy(self.all_toas)
            selected = np.zeros(self.selected_toas.ntoas, dtype=bool)
        else:
            # Make a new selected list by adding a value if the table
            # index at that position is not in the new indices to
            # delete, with a value that is the same as the previous
            # selected array
            newselected = [
                sel
                for idx, sel in zip(self.all_toas.table["index"], selected)
                if idx not in indices
            ]
            selected = np.asarray(newselected, dtype=bool)
            self.selected_toas = self.all_toas[selected]
        # delete the TOAs from the stashed list also
        if self.stashed:
            self.stashed.table = self._delete_TOAs(self.stashed.table)
        return selected

    def update_resids(self):
        # update the pre and post fit residuals using all_toas
        track_mode = "use_pulse_numbers" if self.use_pulse_numbers else None
        self.prefit_resids = Residuals(
            self.all_toas, self.prefit_model, track_mode=track_mode
        )
        if self.selected_toas.ntoas and self.selected_toas.ntoas != self.all_toas.ntoas:
            self.selected_prefit_resids = Residuals(
                self.selected_toas, self.prefit_model, track_mode=track_mode
            )
        else:
            self.selected_prefit_resids = self.prefit_resids
        if self.fitted:
            self.postfit_resids = Residuals(
                self.all_toas, self.postfit_model, track_mode=track_mode
            )

    def orbitalphase(self):
        """
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of unitless quantities of zeros
        """
        if not self.prefit_model.is_binary:
            log.warning("This is not a binary pulsar")
            return u.Quantity(np.zeros(self.all_toas.ntoas))

        toas = self.all_toas
        if self.fitted:
            phase = self.postfit_model.orbital_phase(toas, anom="mean")
        else:
            phase = self.prefit_model.orbital_phase(toas, anom="mean")
        return phase / (2 * np.pi * u.rad)

    def dayofyear(self):
        """
        Return the day of the year for all the TOAs of this pulsar
        """
        t = Time(self.all_toas.get_mjds(), format="mjd")
        year = Time(np.floor(t.decimalyear), format="decimalyear")
        return np.asarray(t.mjd - year.mjd) << u.day

    def year(self):
        """
        Return the decimal year for all the TOAs of this pulsar
        """
        t = Time(self.all_toas.get_mjds(), format="mjd")
        return np.asarray(t.decimalyear) << u.year

    def add_model_params(self):
        """This automatically adds the next available unfit prefix
        parameters to the model so they show up on the GUI
        """
        m = self.prefit_model
        # Add next spin freq deriv
        if "Spindown" in m.components:
            c = m.components["Spindown"]
            n = len(c.get_prefix_mapping_component("F"))
            if f"F{n-1}" in m.free_params and not hasattr(m, f"F{n}"):
                c.add_param(m.F0.new_param(n), setup=True)
                log.debug(f"Adding F{n} to prefit model")
                p = getattr(m, f"F{n}")
                p.quantity = 0.0 * p.units
                p.frozen = True
        # Add next orbital freq deriv
        if "BinaryBT" in m.components:
            c = m.components["BinaryBT"]
            n = len(c.get_prefix_mapping_component("FB"))
            if f"FB{n-1}" in m.free_params and not hasattr(m, f"FB{n}"):
                c.add_param(m.FB0.new_param(n), setup=True)
                log.debug(f"Adding FB{n} to prefit model")
                p = getattr(m, f"FB{n}")
                p.quantity = 0.0 * p.units
                p.frozen = True
        # Add dispersion expansion component
        if "DispersionDM" in m.components:
            c = m.components["DispersionDM"]
            n = len(c.get_prefix_mapping_component("DM")) + 1
            # DM1 is always added, but might be unset
            if n == 2 and m.DM1.value is None:
                p = m.DM1
                p.quantity = 0.0 * p.units
                p.frozen = True
            if f"DM{n-1}" in m.free_params and not hasattr(m, f"DM{n}"):
                c.add_param(m.DM1.new_param(n), setup=True)
                log.debug(f"Adding DM{n} to prefit model")
                p = getattr(m, f"DM{n}")
                p.quantity = 0.0 * p.units
                p.frozen = True
        m.setup()  # Not sure if this is necessary
        m.validate()

    def write_fit_summary(self):
        """
        Summarize fitting results
        """
        if self.fitted:
            wrms = self.selected_postfit_resids.rms_weighted()
            print("Post-Fit Chi2:          %.8g" % self.selected_postfit_resids.chi2)
            print("Post-Fit DOF:            %8d" % self.selected_postfit_resids.dof)
            print(
                "Post-Fit Reduced-Chi2:  %.8g"
                % self.selected_postfit_resids.reduced_chi2
            )
            print("Post-Fit Weighted RMS:  %.8g us" % wrms.to(u.us).value)
            print("------------------------------------")
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
            for key in self.prefit_model.free_params:
                line = "%8s " % key
                pre = getattr(self.prefit_model, key)
                post = getattr(self.postfit_model, key)
                line += "%10s  " % ("" if post.units is None else str(post.units))
                if post.quantity is not None:
                    line += "%24s\t" % pre.str_quantity(pre.quantity)
                    line += "%24s\t" % post.str_quantity(post.quantity)
                    try:
                        line += "%16.8g  " % post.uncertainty.value
                    except Exception:
                        line += "%18s" % ""
                    diff = post.value - pre.value
                    line += "%16.8g  " % diff
                    if pre.uncertainty is not None and pre.uncertainty.value != 0.0:
                        line += "%16.8g" % (diff / pre.uncertainty.value)
                print(line)
        else:
            log.warning("Pulsar has not been fitted yet!")

    def add_phase_wrap(self, selected, phase):
        """
        Add a phase wrap to selected points in the TOAs object

        Turn on pulse number tracking in the model, if it isn't already

        :param selected: boolean array to apply to toas, True = selected toa
        :param phase: phase difference to be added, i.e.  -0.5, +2, etc.
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
            self.all_toas.table["delta_pulse_number"] = np.zeros(self.all_toas.ntoas)
            self.selected_toas.table["delta_pulse_number"] = np.zeros(
                self.selected_toas.ntoas
            )

        # add phase wrap and update
        self.all_toas.table["delta_pulse_number"][selected] += phase
        self.use_pulse_numbers = True
        self.update_resids()

    def add_jump(self, selected):
        """
        jump the toas selected or un-jump them if already jumped

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
            log.info(f"New jump {retval} added for {selected.sum()} toas.")
            return retval
        # if gets here, has at least one jump param already
        # and iif it doesn't overlap or cancel, add the param
        numjumps = self.prefit_model.components["PhaseJump"].get_number_of_jumps()
        if numjumps == 0:
            log.warning(
                "There are no jumps (maskParameter objects) in PhaseJump. Please delete the PhaseJump object and try again. "
            )
            return None
        # delete the jump ad flags if the selected TOAs exactly overlap;
        # else just delete the jump flag from the selected TOAs
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
                log.info("removed param", f"JUMP{str(num)}")
                return toas_jumped

            # Has to be some overlap between jumps and selected TOAs
            elif np.any(toas_jumped & selected):
                # if not, then they don't exactly match, delete the common subset
                jumped_selected = toas_jumped & selected
                # Post fit model and prefit model share the same TOA table, so as long as we
                # don't delete the jump altogether, modifying prefit model table flags is fine.
                self.prefit_model.delete_not_all_jump_toas(
                    self.all_toas.table["flags"][jumped_selected], num
                )
                log.info(
                    f"Removed existing jump JUMP{str(num)} from {jumped_selected.astype(int).sum()} TOAs"
                )
                return list(jumped_selected)
        # if here, then doesn't match anything
        # add jump flags to selected TOAs at their perspective indices in the TOA tables
        retval = self.prefit_model.add_jump_and_flags(
            self.all_toas.table["flags"][selected]
        )
        log.info(f"New jump {retval} added for {selected.sum()} toas.")
        if (
            self.fitted
            and self.prefit_model.components["PhaseJump"]
            != self.postfit_model.components["PhaseJump"]
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

    def getDefaultFitter(self, downhill=False):
        if self.all_toas.wideband:
            return "WidebandDownhillFitter" if downhill else "WidebandTOAFitter"
        if self.prefit_model.has_correlated_errors:
            return "DownhillGLSFitter" if downhill else "GLSFitter"
        else:
            return "DownhillWLSFitter" if downhill else "WLSFitter"

    def print_chi2(self, selected):
        # Select all the TOAs if none are explicitly set
        if not np.any(selected):
            selected = ~selected

        if self.fitted:
            self.prefit_model = self.postfit_model
            self.prefit_resids = self.postfit_resids
            self.add_model_params()

        self.selected_resids = Residuals(self.selected_toas, self.prefit_model)

        wrms = self.selected_resids.rms_weighted()
        print("------------------------------------")
        print("Selected TOAs:           %8d" % self.selected_toas.ntoas)
        print("Selected Chi2:          %.8g" % self.selected_resids.chi2)
        print(
            "Selected Chi2/Ntoa:     %.8g"
            % (self.selected_resids.chi2 / self.selected_toas.ntoas)
        )
        print("Selected Weighted RMS:  %.8g us" % wrms.to(u.us).value)
        print("------------------------------------")

    def fit(self, selected, iters=4, compute_random=False):
        """
        Run a fit using the specified fitter
        """
        # Select all the TOAs if none are explicitly set
        if not np.any(selected):
            selected = ~selected

        if self.fitted:
            self.prefit_model = self.postfit_model
            self.prefit_resids = self.postfit_resids
            self.add_model_params()

        if self.selected_toas.ntoas != self.all_toas.ntoas:
            self.selected_prefit_resids = Residuals(
                self.selected_toas, self.prefit_model
            )
        else:
            self.selected_prefit_resids = self.prefit_resids

        # Have to change the fitter for each fit since TOAs and models change
        log.info(f"Using {self.fit_method}")
        self.fitter = getattr(pint.fitter, self.fit_method)(
            self.selected_toas, self.prefit_model
        )

        wrms = self.selected_prefit_resids.rms_weighted()
        print("\n------------------------------------")
        print(" Pre-Fit Chi2:          %.8g" % self.selected_prefit_resids.chi2)
        print(" Pre-Fit reduced-Chi2:  %.8g" % self.selected_prefit_resids.reduced_chi2)
        print(" Pre-Fit Weighted RMS:  %.8g us" % wrms.to(u.us).value)
        print("------------------------------------")

        # Do the actual fit and mark things as being fit
        self.fitter.fit_toas(maxiter=iters)
        self.fitter.update_model()
        self.postfit_model = self.fitter.model
        self.fitted = True

        # Zero out all of the "delta_pulse_numbers" if they are set
        if np.any(self.all_toas.table["delta_pulse_number"]):
            self.all_toas.table["delta_pulse_number"] = np.zeros(self.all_toas.ntoas)
            self.selected_toas.table["delta_pulse_number"] = np.zeros(
                self.selected_toas.ntoas
            )
        # Re-calculate the pulse numbers here
        self.all_toas.compute_pulse_numbers(self.postfit_model)
        self.selected_toas.compute_pulse_numbers(self.postfit_model)

        # Compute the residuals using correct pulse numbers
        self.postfit_resids = Residuals(self.all_toas, self.postfit_model)
        self.selected_postfit_resids = (
            self.postfit_resids
            if np.all(selected)
            else Residuals(self.selected_toas, self.postfit_model)
        )

        # Need this since it isn't updated using self.fitter.update_model()
        self.fitter.model.CHI2.value = self.selected_postfit_resids.chi2
        # And print the summary
        self.write_fit_summary()

        # Check to see if we should calculate an F-test
        if (
            hasattr(self, "lastfit")
            and (len(self.postfit_model.free_params) > len(self.lastfit["free_params"]))
            and (self.lastfit["ntoas"] == self.fitter.toas.ntoas)
        ):
            prob = FTest(
                self.lastfit["chi2"],
                self.lastfit["dof"],
                self.selected_postfit_resids.chi2,
                self.selected_postfit_resids.dof,
            )
            new_params = set(self.postfit_model.free_params) - set(
                self.lastfit["free_params"]
            )
            print(
                f"F-test comparing post- to pre-fit models for addition of {new_params}:\n"
                f"  P = {prob:.3g} that the improvement is due to noise."
            )

        # plot the prefit without jumps
        pm_no_jumps = copy.deepcopy(self.prefit_model)
        for param in pm_no_jumps.params:
            if param.startswith("JUMP"):
                getattr(pm_no_jumps, param).value = 0.0
                getattr(pm_no_jumps, param).frozen = True
        self.prefit_resids_no_jumps = Residuals(self.all_toas, pm_no_jumps)

        # Store some key params for possible F-testing
        self.lastfit = {
            "free_params": self.fitter.model.free_params,
            "dof": self.selected_postfit_resids.dof,
            "chi2": self.selected_postfit_resids.chi2,
            "ntoas": self.fitter.toas.ntoas,
        }

        # adds extra prefix params for fitting
        self.add_model_params()

        print(
            f"Akaike information criterion = {akaike_information_criterion(self.fitter.model, self.fitter.toas)}"
        )

    def random_models(self, selected):
        """Compute and plot random models"""
        log.info("Computing random models based on parameter covariance matrix.")
        if [p for p in self.postfit_model.free_params if p.startswith("DM")]:
            log.warning(
                "Fitting for DM while using random models can cause strange behavior."
            )

        # These are the currently selected TOAs in the fit
        sim_sel = copy.deepcopy(self.selected_toas)
        # These are single TOAs from each cluster of TOAs
        inds = np.zeros(sim_sel.ntoas, dtype=bool)
        inds[np.unique(sim_sel.get_clusters(), return_index=True)[1]] |= True
        sim_sel = sim_sel[inds]
        # Get the range of MJDs we are using in the fit
        mjds = sim_sel.get_mjds().value
        minselMJD, maxselMJD = mjds.min(), mjds.max()

        extra = 0.1  # Fraction beyond TOAs to plot or calculate random models
        if self.faketoas1 is None:
            mjds = self.all_toas.get_mjds().value
            minallMJD, maxallMJD = mjds.min(), mjds.max()
            spanMJD = maxallMJD - minallMJD
            # Select appropriate number of fake TOAs to generate.
            if spanMJD < 1000:
                Ntoas = 400
            elif spanMJD < 4000:
                Ntoas = 750
            else:
                Ntoas = 1500
            log.debug(
                f"Generating {Ntoas} fake TOAs for the random models over MJD {minallMJD - extra * spanMJD} to {minallMJD + extra * spanMJD}"
            )
            # By default we will use TOAs from the TopoCenter.  This gets done only once.
            self.faketoas1 = make_fake_toas_uniform(
                minallMJD - extra * spanMJD,
                maxallMJD + extra * spanMJD,
                Ntoas,
                self.postfit_model,
                obs="coe",
                freq=1 * u.THz,  # effectively infinite frequency
                include_bipm=sim_sel.clock_corr_info["include_bipm"],
                include_gps=sim_sel.clock_corr_info["include_gps"],
            )
        self.faketoas1.compute_pulse_numbers(self.postfit_model)
        self.faketoas1.get_clusters(add_column=True)

        # Combine our TOAs
        toas = merge_TOAs([sim_sel, self.faketoas1])
        zero_residuals(toas, self.postfit_model)

        # Get a selection array to select the non-fake TOAs
        refs = np.asarray(toas.get_flag_value("name")[0]) != "fake"

        # Compute the new random timing models
        rs = calculate_random_models(
            self.fitter,
            toas,
            Nmodels=15,
            keep_models=False,
            return_time=True,
        )

        # Get a selection array for the fake TOAs that covers the fit TOAs (plus extra)
        mjds = toas.get_mjds().value
        spanMJD = maxselMJD - minselMJD
        toplot = np.bitwise_and(
            mjds > (minselMJD - extra * spanMJD), mjds < (maxselMJD + extra * spanMJD)
        )

        # This is the mean of the reference resids for the selected TOAs
        if selected.sum():  # shorthand for having some selected
            ref_mean = self.postfit_resids.time_resids[selected][inds].mean()
        else:
            ref_mean = self.postfit_resids.time_resids[inds].mean()
        # This is the means of the corresponding resids from the random models
        ran_mean = rs[:, refs].mean(axis=1)
        #  Now adjust each set of random resids so that the ran_mean == ref_mean
        rs -= ran_mean[:, np.newaxis]
        rs += ref_mean
        # And store the key things for plotting
        self.faketoas = toas
        self.random_resids = rs
