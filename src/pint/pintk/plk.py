"""
Interactive emulator of tempo2 plk
"""

from __future__ import division, print_function

import copy
import os
import sys

import astropy.units as u
import matplotlib as mpl
import numpy as np
from astropy import log
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pint.pintk.pulsar as pulsar

try:
    # Python2
    import Tkinter as tk
    import tkFileDialog
    import tkMessageBox
except ImportError:
    # Python3
    import tkinter as tk
    import tkinter.filedialog as tkFileDialog
    import tkinter.messagebox as tkMessageBox

log.setLevel("INFO")
log.info("This should show up")

try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
except ImportError:
    from matplotlib.backends.backend_tkagg import (
        NavigationToolbar2TkAgg as NavigationToolbar2Tk,
    )


log.info(
    "This should also show up. test click revert, turn params on and off, and prefit model"
)

plotlabels = {
    "pre-fit": [
        r"Pre-fit residual ($\mu$s)",
        "Pre-fit residual (phase)",
        "Pre-fit residual (us)",
    ],
    "post-fit": [
        r"Post-fit residual ($\mu$s)",
        "Post-fit residual (phase)",
        "Post-fit residual (us)",
    ],
    "mjd": r"MJD",
    "orbital phase": "Orbital Phase",
    "serial": "TOA number",
    "day of year": "Day of the year",
    "year": "Year",
    "frequency": r"Observing Frequency (MHz)",
    "TOA error": r"TOA uncertainty ($\mu$s)",
    "rounded MJD": r"MJD",
}

helpstring = """The following interactions are currently supported by the Plk pane in the PINTkinter GUI:

Left click:     Select a point

Right click:    (NON-OP) Delete a point

r:              Reset the pane - undo all deletions, selections, etc.

k:              (K)orrect the pane - rescale the axes

f:              Perform a fit on the selected points

d:              Delete the highlighted points

u:              Undo the most recent selection

c:              Clear highlighter from map

j:              Jump the selected points, or unjump them if already jumped

v:              Jump all TOA groups except those selected

i:              Print the prefit model as of this moment

o:              Print the postfit model as of this moment (if it exists)

p:              Print info about highlighted points (or all, if none are selected)

t:              Print the range of MJDs with the highest density of TOAs

+:              Increase pulse number for selected points

-:              Decrease pulse number for selected points

>:              Increase pulse number for all points to the right of selection

<:              Decrease pulse number for all points to the right of selection

h:              Print help
"""

clickDist = 0.0005


class State:
    """class used by revert to save the state of the system before each fit"""

    pass


class PlkFitBoxesWidget(tk.Frame):
    """
    Allows one to select which parameters to fit for
    """

    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)
        self.boxChecked = None
        self.maxcols = 8

    def setCallbacks(self, boxChecked):
        """
        Set the callback functions
        """
        self.boxChecked = boxChecked

    def addFitCheckBoxes(self, model):
        """
        Add the fitting checkboxes for the given model to the frame
        """
        self.deleteFitCheckBoxes()

        self.compGrids = []
        self.compCBs = []
        self.compVisible = []
        self.parVars = {}

        ii = 0
        comps = model.components.keys()
        fitparams = [p for p in model.params if not getattr(model, p).frozen]
        for comp in comps:
            showpars = [
                p
                for p in model.components[comp].params
                if not p in pulsar.nofitboxpars
                and getattr(model, p).quantity is not None
            ]
            # Don't bother showing components without any fittable parameters
            if len(showpars) == 0:
                continue

            self.compVisible.append(tk.IntVar())
            self.compCBs.append(
                tk.Checkbutton(
                    self,
                    text=comp,
                    variable=self.compVisible[ii],
                    command=self.updateLayout,
                )
            )

            self.compGrids.append([])
            for pp, par in enumerate(showpars):
                self.parVars[par] = tk.IntVar()
                self.compGrids[ii].append(
                    tk.Checkbutton(
                        self,
                        text=par,
                        variable=self.parVars[par],
                        command=lambda p=par: self.changedFitCheckBox(p),
                    )
                )
                if par in fitparams:
                    self.compCBs[ii].select()
                    self.compGrids[ii][pp].select()
            ii += 1

        self.updateLayout()

    def deleteFitCheckBoxes(self):
        for widget in self.winfo_children():
            widget.destroy()

    def clear_grid(self):
        for widget in self.winfo_children():
            widget.grid_forget()

    def updateLayout(self):
        self.clear_grid()
        rowCount = 0
        for ii in range(len(self.compGrids)):
            self.compCBs[ii].grid(row=rowCount, column=0, sticky="W")
            if self.compVisible[ii].get():
                for pp, cb in enumerate(self.compGrids[ii]):
                    row = int(pp / self.maxcols)
                    col = pp % self.maxcols + 1
                    cb.grid(row=rowCount + row, column=col, sticky="W")
                rowCount += int(len(self.compGrids[ii]) / self.maxcols)
            rowCount += 1

    def changedFitCheckBox(self, par):
        if self.boxChecked is not None:
            self.boxChecked(par, bool(self.parVars[par].get()))
        log.info("%s set to %d" % (par, self.parVars[par].get()))


class PlkXYChoiceWidget(tk.Frame):
    """
    Allows one to choose which quantities to plot against one another
    """

    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)

        self.xvar = tk.StringVar()
        self.yvar = tk.StringVar()

        self.initPlkXYChoice()

    def initPlkXYChoice(self):
        labellength = 3

        label = tk.Label(self, text="X")
        label.grid(row=0, column=1)
        label = tk.Label(self, text="Y")
        label.grid(row=0, column=2)

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.xbuttons = []
        self.ybuttons = []

        for ii, choice in enumerate(pulsar.plot_labels):
            label = tk.Label(self, text=choice)
            label.grid(row=ii + 1, column=0)

            self.xbuttons.append(
                tk.Radiobutton(
                    self, variable=self.xvar, value=choice, command=self.updateChoice
                )
            )
            self.xbuttons[ii].grid(row=ii + 1, column=1)

            self.ybuttons.append(
                tk.Radiobutton(
                    self, variable=self.yvar, value=choice, command=self.updateChoice
                )
            )
            self.ybuttons[ii].grid(row=ii + 1, column=2)

    def setChoice(self, xid="mjd", yid="pre-fit"):
        for ii, choice in enumerate(pulsar.plot_labels):
            if choice.lower() == xid:
                self.xbuttons[ii].select()
            if choice.lower() == yid:
                self.ybuttons[ii].select()

    def setCallbacks(self, updatePlot):
        """
        Set the callback functions
        """
        self.updatePlot = updatePlot

    def plotIDs(self):
        return self.xvar.get(), self.yvar.get()

    def updateChoice(self):
        if self.updatePlot is not None:
            self.updatePlot()


class PlkToolbar(NavigationToolbar2Tk):
    """
    A modification of the stock Matplotlib toolbar to perform the
    necessary selections/unselections on points
    """

    toolitems = [
        t
        for t in NavigationToolbar2Tk.toolitems
        if t[0] in ("Home", "Back", "Forward", "Pan", "Zoom", "Save")
    ]


class PlkActionsWidget(tk.Frame):
    """
    Shows action items like re-fit, write par, write tim, etc.
    """

    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)

        self.fit_callback = None
        self.clearAll_callback = None
        self.writePar_callback = None
        self.writeTim_callback = None
        self.saveFig_callback = None
        self.revert_callback = None

        self.initPlkActions()

    def initPlkActions(self):
        self.fitbutton = tk.Button(self, text="Fit", command=self.fit)
        self.fitbutton.grid(row=0, column=0)

        button = tk.Button(self, text="Reset", command=self.reset)
        button.grid(row=0, column=1)

        button = tk.Button(self, text="Write par", command=self.writePar)
        button.grid(row=0, column=2)

        button = tk.Button(self, text="Write tim", command=self.writeTim)
        button.grid(row=0, column=3)

        button = tk.Button(self, text="Revert", command=self.revert)
        button.grid(row=0, column=4)

    def setCallbacks(self, fit, reset, writePar, writeTim, revert):
        """
        Callback functions
        """
        self.fit_callback = fit
        self.reset_callback = reset
        self.writePar_callback = writePar
        self.writeTim_callback = writeTim
        self.revert_callback = revert

    def setFitButtonText(self, text):
        self.fitbutton.config(text=text)

    def fit(self):
        if self.fit_callback is not None:
            self.fit_callback()

    def writePar(self):
        if self.writePar_callback is not None:
            self.writePar_callback()
        log.info("Write Par clicked")

    def writeTim(self):
        if self.writeTim_callback is not None:
            self.writeTim_callback()
        log.info("Write Tim clicked")

    def reset(self):
        if self.reset_callback is not None:
            self.reset_callback()
        log.info("Reset clicked")

    def revert(self):
        if self.revert_callback is not None:
            self.revert_callback()
        log.info("Revert clicked")


class PlkWidget(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)

        self.initPlk()
        self.initPlkLayout()
        self.current_state = State()
        self.state_stack = []

        self.update_callbacks = None

        self.press = False
        self.move = False

        self.psr = None

    def initPlk(self):
        self.fitboxesWidget = PlkFitBoxesWidget(master=self)
        self.xyChoiceWidget = PlkXYChoiceWidget(master=self)
        self.actionsWidget = PlkActionsWidget(master=self)

        self.plkDpi = 100
        self.plkFig = mpl.figure.Figure(dpi=self.plkDpi)
        self.plkCanvas = FigureCanvasTkAgg(self.plkFig, self)
        self.plkCanvas.mpl_connect("button_press_event", self.canvasClickEvent)
        self.plkCanvas.mpl_connect("button_release_event", self.canvasReleaseEvent)
        self.plkCanvas.mpl_connect("motion_notify_event", self.canvasMotionEvent)
        self.plkCanvas.mpl_connect("key_press_event", self.canvasKeyEvent)
        self.plkToolbar = PlkToolbar(self.plkCanvas, tk.Frame(self))

        self.plkAxes = self.plkFig.add_subplot(111)  # 111
        self.plkAx2x = self.plkAxes.twinx()
        self.plkAx2y = self.plkAxes.twiny()
        self.plkAxes.set_zorder(0.1)

        self.drawSomething()

    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel("MJD")
        self.plkAxes.set_ylabel("Residual ($\mu$s)")
        self.plkFig.tight_layout()
        self.plkToolbar.push_current()
        self.plkCanvas.draw()

    def initPlkLayout(self):
        self.plkToolbar.master.grid(row=1, column=1, sticky="nesw")
        self.xyChoiceWidget.grid(row=2, column=0, sticky="nw")
        self.plkCanvas.get_tk_widget().grid(row=2, column=1, sticky="nesw")
        self.actionsWidget.grid(row=3, column=0, columnspan=2, sticky="W")

        self.grid_columnconfigure(1, weight=10)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=10)
        self.grid_rowconfigure(3, weight=1)

    def update(self):
        if self.psr is not None:
            self.psr.update_resids()
            self.selected = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
            self.jumped = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
            self.actionsWidget.setFitButtonText("Fit")
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            self.xyChoiceWidget.setChoice()
            self.updatePlot(keepAxes=True)
            self.plkToolbar.update()
            # reset state stack
            self.state_stack = [self.base_state]
            self.current_state = State()

    def setPulsar(self, psr, updates):
        self.psr = psr
        # self.selected & self.jumped = boolean arrays, len = all_toas, True = selected/jumped
        self.selected = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
        self.jumped = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
        # update jumped with any jump params already in the file
        for param in self.psr.prefit_model.params:
            if (
                param.startswith("JUMP")
                and getattr(self.psr.prefit_model, param).frozen == False
            ):
                self.updateJumped(getattr(self.psr.prefit_model, param).name)
        self.update_callbacks = updates

        if not hasattr(self, "base_state"):
            self.base_state = State()
            self.base_state.psr = copy.deepcopy(self.psr)
            self.base_state.ft_flags = copy.deepcopy(self.psr.all_toas.table["flags"])
            self.base_state.t_flags = copy.deepcopy(
                self.psr.selected_toas.table["flags"]
            )
            self.base_state.selected = copy.deepcopy(self.selected)
            self.base_state.jumped = copy.deepcopy(self.jumped)
            self.state_stack.append(self.base_state)

        self.fitboxesWidget.setCallbacks(self.fitboxChecked)
        self.xyChoiceWidget.setCallbacks(self.updatePlot)
        self.actionsWidget.setCallbacks(
            self.fit, self.reset, self.writePar, self.writeTim, self.revert
        )

        self.fitboxesWidget.grid(row=0, column=0, columnspan=2, sticky="W")
        self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
        self.xyChoiceWidget.setChoice()
        self.updatePlot(keepAxes=False)

    def call_updates(self):
        if not self.update_callbacks is None:
            for ucb in self.update_callbacks:
                ucb()

    def fitboxChecked(self, parchanged, newstate):
        """
        When a fitbox is (un)checked, this callback function is called

        @param parchanged:  Which parameter has been (un)checked
        @param newstate:    The new state of the checkbox (True if model should be fit)
        """
        getattr(self.psr.prefit_model, parchanged).frozen = not newstate
        if self.psr.fitted:
            getattr(self.psr.postfit_model, parchanged).frozen = not newstate
        if parchanged.startswith("JUMP"):
            self.updateJumped(parchanged)
        self.call_updates()
        self.updatePlot(keepAxes=True)

    def unselect(self):
        """
        Undo a selection (but not deletes)
        """
        self.psr.selected_toas = copy.deepcopy(self.psr.all_toas)
        self.selected = np.zeros(self.psr.selected_toas.ntoas, dtype=bool)
        self.updatePlot(keepAxes=True)
        self.call_updates()

    def fit(self):
        """
        fit the selected points using the current pre-fit model
        """
        if not self.psr is None:
            # check jumps wont cancel fit, if so, exit here
            if self.check_jump_invalid() == True:
                return None
            if self.psr.fitted:
                # append the current state to the state stack
                self.current_state.psr = copy.deepcopy(self.psr)
                self.current_state.ft_flags = copy.deepcopy(
                    self.psr.all_toas.table["flags"]
                )
                self.current_state.t_flags = copy.deepcopy(
                    self.psr.selected_toas.table["flags"]
                )
                self.current_state.jumped = copy.deepcopy(self.jumped)
                self.state_stack.append(copy.deepcopy(self.current_state))
            self.psr.fit(self.selected)
            self.current_state.selected = copy.deepcopy(self.selected)
            self.actionsWidget.setFitButtonText("Re-fit")
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            xid, yid = self.xyChoiceWidget.plotIDs()
            self.xyChoiceWidget.setChoice(xid=xid, yid="post-fit")
            self.jumped = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
            for param in self.psr.prefit_model.params:
                if (
                    param.startswith("JUMP")
                    and getattr(self.psr.prefit_model, param).frozen == False
                ):
                    self.updateJumped(getattr(self.psr.prefit_model, param).name)
            self.updatePlot(keepAxes=True)
        self.call_updates()

    def reset(self):
        """
        Reset all plot changes for this pulsar
        """
        self.psr.reset_TOAs()
        self.psr.fitted = False
        self.psr = copy.deepcopy(self.base_state.psr)
        # must specifically copy flags because deepcopy doesn't work with numpy Tables
        self.psr.all_toas.table["flags"] = copy.deepcopy(self.base_state.ft_flags)
        self.psr.selected_toas.table["flags"] = copy.deepcopy(self.base_state.t_flags)
        self.selected = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
        self.jumped = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
        for param in self.psr.prefit_model.params:
            if (
                param.startswith("JUMP")
                and getattr(self.psr.prefit_model, param).frozen == False
            ):
                self.updateJumped(param)
        self.actionsWidget.setFitButtonText("Fit")
        self.fitboxesWidget.addFitCheckBoxes(self.base_state.psr.prefit_model)
        self.xyChoiceWidget.setChoice()
        self.updatePlot(keepAxes=False)
        self.plkToolbar.update()
        self.current_state = State()
        self.state_stack = [self.base_state]
        self.call_updates()

    def writePar(self):
        """
        Write the fit parfile to a file
        """
        filename = tkFileDialog.asksaveasfilename(title="Choose output par file")
        try:
            fout = open(filename, "w")
            if self.psr.fitted:
                fout.write(self.psr.postfit_model.as_parfile())
                log.info("Saved post-fit parfile to %s" % filename)
            else:
                fout.write(self.psr.prefit_model.as_parfile())
                log.warn(
                    "Pulsar has not been fitted! Saving pre-fit parfile to %s"
                    % filename
                )
            fout.close()
        except:
            log.error("Could not save parfile to filename:\t%s" % filename)

    def writeTim(self):
        """
        Write the current timfile to a file
        """
        filename = tkFileDialog.asksaveasfilename(title="Choose output tim file")
        try:
            log.info("Choose output file %s" % filename)
            self.psr.all_toas.write_TOA_file(filename, format="TEMPO2")
        except:
            log.error("Count not save file to filename:\t%s" % filename)

    def revert(self):
        """
        revert to the state of the model and toas right before the last fit
        """
        if len(self.state_stack) > 0 and self.psr.fitted and self.psr is not None:
            c_state = self.state_stack.pop()
            self.psr = copy.deepcopy(c_state.psr)
            self.psr.all_toas.table["flags"] = copy.deepcopy(c_state.ft_flags)
            self.psr.selected_toas.table["flags"] = copy.deepcopy(c_state.t_flags)
            self.jumped = copy.deepcopy(c_state.jumped)
            self.selected = copy.deepcopy(c_state.selected)
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            if len(self.state_stack) == 0:
                self.state_stack.append(self.base_state)
                self.actionsWidget.setFitButtonText("Fit")
            self.psr.update_resids()
            self.updatePlot(keepAxes=True)
        else:
            log.warn("No model to revert to")

    def updatePlot(self, keepAxes=False):
        """
        Update the plot/figure

        @param keepAxes: Set to True whenever we want to preserve zoom
        """
        if self.psr is not None:
            # Get a mask for the plotting points
            # msk = self.psr.mask('plot')

            # Get the IDs of the X and Y axis
            self.xid, self.yid = self.xyChoiceWidget.plotIDs()

            # Retrieve the data
            x, xerr = self.psr_data_from_label(self.xid)
            y, yerr = self.psr_data_from_label(self.yid)
            if x is not None and y is not None:
                self.xvals = x
                self.yvals = y
                self.yerrs = yerr
                self.plotResiduals(keepAxes=keepAxes)
            else:
                raise ValueError("Nothing to plot!")

        self.plkFig.tight_layout()
        self.plkCanvas.draw()

    def plotErrorbar(self, selected, color):
        """
        For some reason, errorbar breaks completely when the plotting array is
        of length 2. So this workaround is needed
        """
        if selected.sum() != 2:
            self.plkAxes.errorbar(
                self.xvals[selected].reshape([-1, 1]),
                self.yvals[selected].reshape([-1, 1]),
                yerr=self.yerrs[selected].reshape([-1, 1]),
                fmt=".",
                color=color,
            )
        else:
            self.plkAxes.errorbar(
                self.xvals[selected][0].reshape([-1, 1]),
                self.yvals[selected][0].reshape([-1, 1]),
                yerr=self.yerrs[selected][0].reshape([-1, 1]),
                fmt=".",
                color=color,
            )
            self.plkAxes.errorbar(
                self.xvals[selected][1].reshape([-1, 1]),
                self.yvals[selected][1].reshape([-1, 1]),
                yerr=self.yerrs[selected][1].reshape([-1, 1]),
                fmt=".",
                color=color,
            )

    def plotResiduals(self, keepAxes=False):
        """
        Update the plot, given all the plotting info
        """
        if keepAxes:
            xmin, xmax = self.plkAxes.get_xlim()
            ymin, ymax = self.plkAxes.get_ylim()
        else:
            xave = 0.5 * (np.max(self.xvals) + np.min(self.xvals))
            xmin = xave - 1.10 * (xave - np.min(self.xvals))
            xmax = xave + 1.10 * (np.max(self.xvals) - xave)
            if self.yerrs is None:
                yave = 0.5 * (np.max(self.yvals) + np.min(self.yvals))
                ymin = yave - 1.10 * (yave - np.min(self.yvals))
                ymax = yave + 1.10 * (np.max(self.yvals) - yave)
            else:
                yave = 0.5 * (
                    np.max(self.yvals + self.yerrs) + np.min(self.yvals - self.yerrs)
                )
                ymin = yave - 1.10 * (yave - np.min(self.yvals - self.yerrs))
                ymax = yave + 1.10 * (np.max(self.yvals + self.yerrs) - yave)
            xmin, xmax = xmin.value, xmax.value
            ymin, ymax = ymin.value, ymax.value

        self.plkAxes.clear()
        self.plkAx2x.clear()
        self.plkAx2y.clear()
        self.plkAxes.grid(True)

        if self.yerrs is None:
            self.plkAxes.scatter(
                self.xvals[~self.selected],
                self.yvals[~self.selected],
                marker=".",
                color="blue",
            )
            self.plkAxes.scatter(
                self.xvals[self.jumped],
                self.yvals[self.jumped],
                marker=".",
                color="red",
            )
            self.plkAxes.scatter(
                self.xvals[self.selected],
                self.yvals[self.selected],
                marker=".",
                color="orange",
            )
        else:
            self.plotErrorbar(~self.selected, color="blue")
            self.plotErrorbar(self.jumped, color="red")
            self.plotErrorbar(self.selected, color="orange")
        self.plkAxes.axis([xmin, xmax, ymin, ymax])
        self.plkAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.plkAx2y.set_visible(False)
        self.plkAx2x.set_visible(False)
        # clears the views stack and puts the scaled view on top, fixes toolbar problems
        # self.plkToolbar._views.clear()
        self.plkToolbar.push_current()

        if self.xid in ["pre-fit", "post-fit"]:
            self.plkAxes.set_xlabel(plotlabels[self.xid][0])
            m = (
                self.psr.prefit_model
                if self.xid == "pre-fit" or not self.psr.fitted
                else self.psr.postfit_model
            )
            if hasattr(m, "F0"):
                self.plkAx2y.set_visible(True)
                self.plkAx2y.set_xlabel(plotlabels[self.xid][1])
                f0 = m.F0.quantity.to(u.MHz).value
                self.plkAx2y.set_xlim(xmin * f0, xmax * f0)
                self.plkAx2y.xaxis.set_major_locator(
                    mpl.ticker.FixedLocator(self.plkAxes.get_xticks() * f0)
                )
        else:
            self.plkAxes.set_xlabel(plotlabels[self.xid])

        if self.yid in ["pre-fit", "post-fit"]:
            self.plkAxes.set_ylabel(plotlabels[self.yid][0])
            try:
                r = (
                    self.psr.prefit_resids
                    if self.yid == "pre-fit" or not self.psr.fitted
                    else self.psr.postfit_resids
                )
                f0 = r.get_PSR_freq().to(u.MHz).value
                self.plkAx2x.set_visible(True)
                self.plkAx2x.set_ylabel(plotlabels[self.yid][1])
                self.plkAx2x.set_ylim(ymin * f0, ymax * f0)
                self.plkAx2x.yaxis.set_major_locator(
                    mpl.ticker.FixedLocator(self.plkAxes.get_yticks() * f0)
                )
            except:
                pass
        else:
            self.plkAxes.set_ylabel(plotlabels[self.yid])

        self.plkAxes.set_title(self.psr.name, y=1.1)

        # plot random models
        if self.psr.fitted == True:
            # TODO: add random models on/off button
            log.info("plotting random models")
            f_toas = self.psr.fake_toas
            print("Computing random models based on parameter covariance matrix...")
            rs = self.psr.random_resids
            for i in range(len(rs)):
                self.plkAxes.plot(f_toas, rs[i], "-k", alpha=0.3)

    def print_info(self):
        """
        Write information about the current selection, or all points
        Format is:
        TOA_index   X_val   Y_val

        or, if residuals:
        TOA_index   X_val   time_resid  phase_resid
        """
        if np.sum(self.selected) == 0:
            selected = np.ones(self.psr.selected_toas.ntoas, dtype=bool)
        else:
            selected = self.selected

        header = "%6s" % "TOA"

        f0x, f0y = None, None
        xf, yf = False, False
        if self.xid in ["pre-fit", "post-fit"]:
            header += " %16s" % plotlabels[self.xid][2]
            try:
                r = (
                    self.psr.prefit_resids
                    if self.xid == "pre-fit" or not self.psr.fitted
                    else self.psr.postfit_resids
                )
                f0x = r.get_PSR_freq().to(u.MHz).value
                header += " %16s" % plotlabels[self.xid][1]
                xf = True
            except:
                pass
        else:
            header += " %16s" % plotlabels[self.xid]
        if self.yid in ["pre-fit", "post-fit"]:
            header += " %16s" % plotlabels[self.yid][2]
            try:
                r = (
                    self.psr.prefit_resids
                    if self.xid == "pre-fit" or not self.psr.fitted
                    else self.psr.postfit_resids
                )
                f0y = r.get_PSR_freq().to(u.MHz).value
                header += " %16s" % plotlabels[self.yid][1]
                yf = True
            except:
                pass
        else:
            header += "%12s" % plotlabels[self.yid]

        print(header)
        print("-" * len(header))

        xs = self.xvals[selected].value
        ys = self.yvals[selected].value
        inds = self.psr.selected_toas.table["index"][selected]

        for i in range(len(xs)):
            line = "%6d" % inds[i]
            line += " %16.8g" % xs[i]
            if xf:
                line += " %16.8g" % (xs[i] * f0x)
            line += " %16.8g" % ys[i]
            if yf:
                line += " %16.8g" % (ys[i] * f0y)
            print(line)

    def psr_data_from_label(self, label):
        """
        Given a label, get the corresponding data from the pulsar

        @param label: The label for the data we want
        @return:    data, error
        """
        data, error = None, None
        if label == "pre-fit":
            if self.psr.fitted:
                # TODO: may want to include option for prefit resids to include jumps
                data = self.psr.prefit_resids_no_jumps.time_resids.to(u.us)
                error = self.psr.all_toas.get_errors().to(u.us)
                return data, error
            data = self.psr.prefit_resids.time_resids.to(u.us)
            error = self.psr.all_toas.get_errors().to(u.us)
        elif label == "post-fit":
            if self.psr.fitted:
                data = self.psr.postfit_resids.time_resids.to(u.us)
            else:
                log.warn("Pulsar has not been fitted yet! Giving pre-fit residuals")
                data = self.psr.prefit_resids.time_resids.to(u.us)
            error = self.psr.all_toas.get_errors().to(u.us)
        elif label == "mjd":
            data = self.psr.all_toas.get_mjds()
            error = self.psr.all_toas.get_errors()
        elif label == "orbital phase":
            data = self.psr.orbitalphase()
            error = None
        elif label == "serial":
            data = np.arange(self.psr.all_toas.ntoas) * u.m / u.m
            error = None
        elif label == "day of year":
            data = self.psr.dayofyear()
            error = None
        elif label == "year":
            data = self.psr.year()
            error = None
        elif label == "frequency":
            data = self.psr.all_toas.get_freqs()
            error = None
        elif label == "TOA error":
            data = self.psr.all_toas.get_errors().to(u.us)
            error = None
        elif label == "rounded MJD":
            data = np.floor(self.psr.all_toas.get_mjds() + 0.5 * u.d)
            error = self.psr.all_toas.get_errors().to(u.d)
        return data, error

    def coordToPoint(self, cx, cy):
        """
        Given a set of x-y coordinates, get the TOA index closest to it
        """
        ind = None

        if self.psr is not None:
            x = self.xvals.value
            y = self.yvals.value

            xmin, xmax, ymin, ymax = self.plkAxes.axis()
            dist = ((x - cx) / (xmax - xmin)) ** 2.0 + ((y - cy) / (ymax - ymin)) ** 2.0
            ind = np.argmin(dist)
            # print('Closest point is %d:(%s, %s) at d=%f' % (ind, self.xvals[ind], self.yvals[ind], dist[ind]))

            if dist[ind] > clickDist:
                log.warn("Not close enough to a point")
                ind = None

        return ind

    def check_jump_invalid(self):
        """checks if jumps will cancel the attempted fit"""
        if "PhaseJump" not in self.psr.prefit_model.components:
            return False
        fit_jumps = []
        for param in self.psr.prefit_model.params:
            if getattr(
                self.psr.prefit_model, param
            ).frozen == False and param.startswith("JUMP"):
                fit_jumps.append(int(param[4:]))
        jumps = [
            True if "jump" in dict.keys() and dict["jump"] in fit_jumps else False
            for dict in self.psr.selected_toas.table["flags"]
        ]
        if all(jumps):
            log.warn(
                "toas being fit must not all be jumped. Remove or uncheck at least one jump in the selected toas before fitting."
            )
            return True

    def updateJumped(self, jump_name):
        """update self.jumped for the jump given"""
        # if removing a jump, add_jump returns a boolean array rather than a name
        if type(jump_name) == list:
            self.jumped[jump_name] = False
            return None
        elif type(jump_name) != str:
            log.error(jump_name, "is not a string")
            return None
        num = int(jump_name[4:])
        jump_select = [
            num == jump_num
            for jump_num in [
                int(dict["jump"]) if "jump" in dict.keys() else np.nan
                for dict in self.psr.all_toas.table["flags"]
            ]
        ]
        self.jumped[jump_select] = ~self.jumped[jump_select]

    def canvasClickEvent(self, event):
        """
        Call this function when the figure/canvas is clicked
        """
        self.plkCanvas.get_tk_widget().focus_set()
        if event.inaxes == self.plkAxes:
            self.press = True
            self.pressEvent = event

    def canvasMotionEvent(self, event):
        """
        Call this function when mouse is moved in the figure/canvas
        """
        if event.inaxes == self.plkAxes and self.press:
            self.move = True
            # Draw bounding box
            if self.plkToolbar._active is None:
                x0, x1 = self.pressEvent.x, event.x
                y0, y1 = self.pressEvent.y, event.y
                height = self.plkFig.bbox.height
                y0 = height - y0
                y1 = height - y1
                if hasattr(self, "brect"):
                    self.plkCanvas._tkcanvas.delete(self.brect)
                self.brect = self.plkCanvas._tkcanvas.create_rectangle(x0, y0, x1, y1)

    def canvasReleaseEvent(self, event):
        """
        Call this function when the figure/canvas is released
        """
        if self.press and not self.move:
            self.stationaryClick(event)
        elif self.press and self.move:
            self.clickAndDrag(event)
        self.press = False
        self.move = False

    def stationaryClick(self, event):
        """
        Call this function when the mouse is clicked but not moved
        """
        if event.inaxes == self.plkAxes:
            ind = self.coordToPoint(event.xdata, event.ydata)
            if ind is not None:
                # TODO: right click to delete doesn't work, needs to be reinstated
                if event.button == 2:
                    # Right click is delete
                    log.error("right click to delete is non-operational")
                #    self.psr.toas.table.remove_row(ind)
                #    self.psr.toas.table = self.psr.toas.table.group_by('obs')
                #    if hasattr(self.psr.toas, 'table_selects'):
                #        for i in range(len(self.psr.toas.table_selects)):
                #            self.psr.toas.table_selects[i].remove_row(ind)
                #            self.psr.toas.table_selects[i] = \
                #                self.psr.toas.table_selects[i].group_by('obs')
                #    self.selected = np.delete(self.selected, ind)
                #    self.psr.update_resids()
                #    self.updatePlot(keepAxes=True)
                #    self.call_updates()
                if event.button == 1 and self.plkToolbar._active is None:
                    # Left click is select
                    self.selected[ind] = not self.selected[ind]
                    self.updatePlot(keepAxes=True)

    def clickAndDrag(self, event):
        """
        Call this function when the mouse is clicked and dragged
        """
        if event.inaxes == self.plkAxes and self.plkToolbar._active is None:
            xmin, xmax = self.pressEvent.xdata, event.xdata
            ymin, ymax = self.pressEvent.ydata, event.ydata
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            self.selected = (self.xvals.value > xmin) & (self.xvals.value < xmax)
            self.selected &= (self.yvals.value > ymin) & (self.yvals.value < ymax)
            self.updatePlot(keepAxes=True)
            self.plkCanvas._tkcanvas.delete(self.brect)
            self.psr.selected_toas = copy.deepcopy(self.psr.all_toas)
            self.psr.selected_toas.select(self.selected)
            self.psr.update_resids()
            self.call_updates()

    def canvasKeyEvent(self, event):
        """
        A key is pressed. Handle all the shortcuts here
        """
        fkey = event.key
        xpos, ypos = event.xdata, event.ydata
        ukey = ord(fkey[-1])

        if ukey == ord("r"):
            # Reset the pane
            self.reset()
        elif ukey == ord("k"):
            # Rescale axes
            self.updatePlot(keepAxes=False)
        elif ukey == ord("f"):
            self.fit()
        elif ukey == ord("-"):
            self.psr.add_phase_wrap(self.selected, -1)
            self.updatePlot(keepAxes=True)
            self.call_updates()
        elif ukey == ord("+"):
            self.psr.add_phase_wrap(self.selected, 1)
            self.updatePlot(keepAxes=True)
            self.call_updates()
        elif ukey == ord(">"):
            if np.sum(self.selected) > 0:
                selected = copy.deepcopy(self.selected)
                ind = np.nonzero(selected)[0][0]
                selected[ind:] = True
                self.psr.add_phase_wrap(selected, 1)
                self.updatePlot(keepAxes=False)
                self.call_updates()
        elif ukey == ord("<"):
            if np.sum(self.selected) > 0:
                selected = copy.deepcopy(self.selected)
                ind = np.nonzero(selected)[0][0]
                selected[ind:] = True
                self.psr.add_phase_wrap(selected, -1)
                self.updatePlot(keepAxes=False)
                self.call_updates()
        elif ukey == ord("d"):
            # if any of the points are jumped, tell the user to delete the jump(s) first
            jumped_copy = copy.deepcopy(self.jumped)
            for param in self.psr.prefit_model.params:
                if (
                    param.startswith("JUMP")
                    and getattr(self.psr.prefit_model, param).frozen == True
                ):
                    self.updateJumped(param)
            all_jumped = copy.deepcopy(self.jumped)
            self.jumped = jumped_copy
            if True in [a and b for a, b in zip(self.selected, all_jumped)]:
                log.warn(
                    "cannot delete jumped toas. Delete interfering jumps before deleting toas."
                )
                return None
            # Delete the selected points
            self.psr.all_toas.table = self.psr.all_toas.table[~self.selected].group_by(
                "obs"
            )
            self.psr.selected_toas = copy.deepcopy(self.psr.all_toas)
            if hasattr(self.psr.all_toas, "table_selects"):
                for i in range(len(self.psr.all_toas.table_selects)):
                    self.psr.all_toas.table_selects[
                        i
                    ] = self.psr.all_toas.table_selects[i][~self.selected].group_by(
                        "obs"
                    )
            self.jumped = self.jumped[~self.selected]
            self.selected = np.zeros(self.psr.all_toas.ntoas, dtype=bool)
            self.psr.update_resids()
            self.updatePlot(keepAxes=True)
            self.call_updates()
        elif ukey == ord("u"):
            self.unselect()
        elif ukey == ord("j"):
            # jump the selected points, or unjump if already jumped
            jump_name = self.psr.add_jump(self.selected)
            self.updateJumped(jump_name)
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            self.updatePlot(keepAxes=True)
            self.call_updates()
        elif ukey == ord("v"):
            # jump all groups except the one(s) selected, or jump all groups if none selected
            jumped_copy = copy.deepcopy(self.jumped)
            for param in self.psr.prefit_model.params:
                if (
                    param.startswith("JUMP")
                    and getattr(self.psr.prefit_model, param).frozen == True
                ):
                    self.updateJumped(param)
            all_jumped = copy.deepcopy(self.jumped)
            self.jumped = jumped_copy
            groups = list(self.psr.all_toas.table["groups"])
            # jump each group, check doesn't overlap with existing jumps and selected
            for num in np.arange(max(groups) + 1):
                group_bool = [
                    num == group for group in self.psr.all_toas.table["groups"]
                ]
                if True in [
                    a and b for a, b in zip(group_bool, self.selected)
                ] or True in [a and b for a, b in zip(group_bool, all_jumped)]:
                    continue
                self.psr.selected_toas = copy.deepcopy(self.psr.all_toas)
                self.psr.selected_toas.select(group_bool)
                jump_name = self.psr.add_jump(group_bool)
                self.updateJumped(jump_name)
            self.psr.selected_toas = copy.deepcopy(self.psr.all_toas)
            if (
                self.selected is not None
                and self.selected is not []
                and all(self.selected) is not False
            ):
                self.psr.selected_toas.select(self.selected)
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            self.updatePlot(keepAxes=True)
            self.call_updates()
        elif ukey == ord("c"):
            self.selected = np.zeros(self.psr.selected_toas.ntoas, dtype=bool)
            self.updatePlot(keepAxes=True)
        elif ukey == ord("i"):
            log.info("PREFIT MODEL")
            log.info(self.psr.prefit_model.as_parfile())
        elif ukey == ord("o"):
            if self.psr.fitted:
                log.info("POSTFIT MODEL")
                log.info(self.psr.postfit_model.as_parfile())
            else:
                log.warn("No postfit model to show")
        elif ukey == ord("p"):
            self.print_info()
        elif ukey == ord("h"):
            print(helpstring)
        elif ukey == ord("t"):
            print(self.psr.all_toas.get_highest_density_range())
