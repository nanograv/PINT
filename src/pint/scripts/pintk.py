#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
"""Tkinter interactive interface for PINT pulsar timing tool"""
import argparse

import sys
import os

import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
import matplotlib as mpl
from loguru import logger as log

import pint.logging

pint.logging.setup(level=pint.logging.script_level)

import pint
from pint.pintk.paredit import ParWidget
from pint.pintk.plk import PlkWidget, helpstring
from pint.pintk.pulsar import Pulsar
from pint.pintk.timedit import TimWidget


__all__ = ["main"]


class PINTk:
    """Main PINTk window."""

    def __init__(
        self,
        master,
        parfile=None,
        timfile=None,
        fitter="downhill",
        ephem=None,
        loglevel=None,
        **kwargs,
    ):
        self.master = master
        self.master.title("Tkinter interface to PINT")

        self.mainFrame = tk.Frame(master=self.master)
        self.mainFrame.grid(row=0, column=0, sticky="nesw")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.loglevel = loglevel
        self.maxcols = 2

        self.createWidgets()
        if parfile is not None and timfile is not None:
            self.openPulsar(
                parfile=parfile, timfile=timfile, fitter=fitter, ephem=ephem
            )

        self.initUI()
        self.updateLayout()

    def initUI(self):
        # Create top level menus
        top = self.mainFrame.winfo_toplevel()
        self.menuBar = tk.Menu(top)
        top["menu"] = self.menuBar

        self.fileMenu = tk.Menu(self.menuBar)
        self.fileMenu.add_command(
            label="Open par/tim",
            command=self.openParTim,
            underline=0,
            accelerator="Ctrl+O",
        )
        self.fileMenu.add_command(label="Switch model", command=self.switchModel)
        self.fileMenu.add_command(label="Switch TOAs", command=self.switchTOAs)
        parfile_submenu = tk.Menu(self.fileMenu)
        parfile_submenu.add_command(
            label="Write par (pint format)", command=self.writeParPINT
        )
        parfile_submenu.add_command(
            label="Write par (tempo2 format)", command=self.writeParTempo2
        )
        parfile_submenu.add_command(
            label="Write par (tempo format)", command=self.writeParTempo
        )
        self.fileMenu.add_cascade(label="Write par...", menu=parfile_submenu)
        timfile_submenu = tk.Menu(self.fileMenu)
        timfile_submenu.add_command(
            label="Write tim (tempo2 format)", command=self.writeTimTempo2
        )
        timfile_submenu.add_command(
            label="Write tim (tempo format)", command=self.writeTimTempo
        )
        self.fileMenu.add_cascade(label="Write tim...", menu=timfile_submenu)
        self.fileMenu.add_command(label="Exit", command=top.destroy, accelerator="q")
        self.menuBar.add_cascade(label="File", menu=self.fileMenu, underline=0)

        self.viewMenu = tk.Menu(self.menuBar)
        self.viewMenu.add_checkbutton(
            label="Plk",
            command=self.updateLayout,
            variable=self.active["plk"],
            accelerator="Ctrl+p",
        )
        self.viewMenu.add_checkbutton(
            label="Model Editor",
            command=self.updateLayout,
            variable=self.active["par"],
            accelerator="Ctrl+m",
        )
        self.viewMenu.add_checkbutton(
            label="TOAs Editor",
            command=self.updateLayout,
            variable=self.active["tim"],
            accelerator="Ctrl+t",
        )
        self.menuBar.add_cascade(label="View", menu=self.viewMenu)

        self.helpMenu = tk.Menu(self.menuBar)
        self.helpMenu.add_command(label="About", command=self.about)
        self.helpMenu.add_command(
            label="PINTk Help", command=lambda: print(helpstring), accelerator="h"
        )
        self.menuBar.add_cascade(label="Help", menu=self.helpMenu)

        # Key bindings
        top.bind("<Control-p>", lambda e: self.toggle("plk"))
        top.bind("<Control-m>", lambda e: self.toggle("par"))
        top.bind("<Control-t>", lambda e: self.toggle("tim"))
        top.bind("<Control-o>", lambda e: self.openParTim())

    def createWidgets(self):
        self.widgets = {
            "plk": PlkWidget(master=self.mainFrame, loglevel=self.loglevel),
            "par": ParWidget(master=self.mainFrame),
            "tim": TimWidget(master=self.mainFrame),
        }
        self.active = {"plk": tk.IntVar(), "par": tk.IntVar(), "tim": tk.IntVar()}
        self.active["plk"].set(1)

    def updateLayout(self):
        for widget in self.mainFrame.winfo_children():
            widget.grid_forget()

        visible = 0
        for key in self.active.keys():
            if self.active[key].get():
                row = int(visible / self.maxcols)
                col = visible % self.maxcols
                self.widgets[key].grid(row=row, column=col, sticky="nesw")
                self.mainFrame.grid_rowconfigure(row, weight=1)
                self.mainFrame.grid_columnconfigure(col, weight=1)
                visible += 1

    def openPulsar(self, parfile, timfile, fitter="downhill", ephem=None):
        self.psr = Pulsar(parfile, timfile, ephem, fitter=fitter)
        self.widgets["plk"].setPulsar(
            self.psr,
            updates=[self.widgets["par"].set_model, self.widgets["tim"].set_toas],
        )
        self.widgets["par"].setPulsar(self.psr, updates=[self.widgets["plk"].update])
        self.widgets["tim"].setPulsar(self.psr, updates=[self.widgets["plk"].update])

    def switchModel(self):
        parfile = tkFileDialog.askopenfilename(title="Open par file")
        self.psr.parfile = parfile
        self.psr.reset_model()
        self.widgets["plk"].update()
        self.widgets["par"].set_model()

    def switchTOAs(self):
        timfile = tkFileDialog.askopenfilename()
        self.psr.timfile = timfile
        self.psr.reset_TOAs()
        self.widgets["plk"].update()
        self.widgets["tim"].set_toas()

    def openParTim(self):
        parfile = tkFileDialog.askopenfilename(title="Open par file")
        timfile = tkFileDialog.askopenfilename(title="Open tim file")
        self.openPulsar(parfile, timfile)

    def writeParPINT(self):
        self.widgets["plk"].writePar(format="pint")

    def writeParTempo(self):
        self.widgets["plk"].writePar(format="tempo")

    def writeParTempo2(self):
        self.widgets["plk"].writePar(format="tempo2")

    def writeTimTempo(self):
        self.widgets["plk"].writeTim(format="tempo")

    def writeTimTempo2(self):
        self.widgets["plk"].writeTim(format="tempo2")

    def toggle(self, key):
        self.active[key].set((self.active[key].get() + 1) % 2)
        self.updateLayout()

    def about(self):
        tkMessageBox.showinfo(
            title="About PINTk",
            message=f"A Tkinter based graphical interface to PINT (version={pint.__version__}), using matplotlib (version={mpl.__version__}) and the {mpl.get_backend()} backend",
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tkinter interface for PINT pulsar timing tool"
    )
    parser.add_argument("parfile", help="parfile to use")
    parser.add_argument("timfile", help="timfile to use")
    parser.add_argument("--ephem", help="Ephemeris to use", default=None)
    parser.add_argument(
        "--test",
        help="Build UI and exit. Just for unit testing.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--fitter",
        type=str,
        choices=(
            "notdownhill",
            "downhill",
            "WLSFitter",
            "GLSFitter",
            "WidebandTOAFitter",
            "PowellFitter",
            "DownhillWLSFitter",
            "DownhillGLSFitter",
            "WidebandDownhillFitter",
            "WidebandLMFitter",
        ),
        default="downhill",
        help="PINT Fitter to use [default='downhill'].  'notdownhill' will choose WLS/GLS/WidebandTOA depending on TOA/model properties.  'downhill' will do the same for Downhill versions.",
    )
    parser.add_argument(
        "--version",
        action="version",
        help="Print version info and  exit.",
        version=f"This is PINT version {pint.__version__}, using matplotlib (version={mpl.__version__})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=pint.logging.levels,
        default=pint.logging.script_level,
        help="Logging level",
        dest="loglevel",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", default=0, action="count", help="Decrease output verbosity"
    )

    args = parser.parse_args(argv)
    pint.logging.setup(
        level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet)
    )
    # see if the arguments were flipped
    if (
        os.path.splitext(args.parfile)[1] == ".tim"
        and os.path.splitext(args.timfile)[1] == ".par"
    ):
        log.debug(
            f"Swapping inputs: parfile='{args.timfile}' and timfile='{args.parfile}'"
        )
        args.parfile, args.timfile = args.timfile, args.parfile
    else:
        if os.path.splitext(args.timfile)[1] != ".tim":
            log.info(
                f"Input timfile '{args.timfile}' has unusual extension '{os.path.splitext(args.timfile)[1]}': is this intended?"
            )
        if os.path.splitext(args.parfile)[1] != ".par":
            log.info(
                f"Input parfile '{args.parfile}' has unusual extension '{os.path.splitext(args.parfile)[1]}': is this intended?"
            )

    root = tk.Tk()
    root.minsize(1000, 800)
    if not args.test:
        app = PINTk(
            root,
            parfile=args.parfile,
            timfile=args.timfile,
            fitter=args.fitter,
            ephem=args.ephem,
            loglevel=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet),
        )
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        img = tk.Image("photo", file=pint.pintk.plk.icon_img)

        root.tk.call("wm", "iconphoto", root._w, img)
        tk.mainloop()


if __name__ == "__main__":
    main()
