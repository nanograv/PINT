import copy
import os
import tempfile

import astropy.time
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox

import pint


class TimActionsWidget(tk.Frame):
    """
    Allows the user to reset the model, apply changes, or save to a parfile
    """

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.reset_callback = None
        self.remove_callback = None
        self.apply_callback = None
        self.write_callback = None

        self.initLayout()

    def initLayout(self):
        button = tk.Button(self, text="Reset TOAs", command=self.resetTimfile)
        button.grid(row=0, column=0)

        button = tk.Button(self, text="Remove Changes", command=self.removeChanges)
        button.grid(row=0, column=1)

        button = tk.Button(self, text="Apply Changes", command=self.applyChanges)
        button.grid(row=0, column=2)

        button = tk.Button(self, text="Write Tim", command=self.writeTim)
        button.grid(row=0, column=3)

    def setCallbacks(self, resetTimfile, removeChanges, applyChanges, writeTim):
        self.reset_callback = resetTimfile
        self.remove_callback = removeChanges
        self.apply_callback = applyChanges
        self.write_callback = writeTim

    def resetTimfile(self):
        if self.reset_callback is not None:
            self.reset_callback()
        print("Reset clicked")

    def removeChanges(self):
        if self.remove_callback is not None:
            self.remove_callback()
        print("Remove clicked")

    def applyChanges(self):
        if self.apply_callback is not None:
            self.apply_callback()
        print("Apply clicked")

    def writeTim(self):
        if self.write_callback is not None:
            self.write_callback()
        print("Write clicked")


class TimWidget(tk.Frame):
    """
    A widget that allows editing and saving of a pulsar timfile
    """

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.psr = None
        self.update_callbacks = None
        self.initLayout()

    def initLayout(self):
        self.editor = tk.Text(self)
        self.editor.grid(row=0, column=0, sticky="nesw")

        scrollbar = tk.Scrollbar(self, command=self.editor.yview)
        scrollbar.grid(row=0, column=1, sticky="nesw")
        self.editor["yscrollcommand"] = scrollbar.set

        self.actionsWidget = TimActionsWidget(master=self)
        self.actionsWidget.grid(row=1, column=0, sticky="w")

        self.grid_rowconfigure(0, weight=10)
        self.grid_rowconfigure(1, weight=1)

    def setPulsar(self, psr, updates):
        self.psr = psr

        self.actionsWidget.setCallbacks(
            self.reset, self.set_toas, self.applyChanges, self.writeTim
        )
        self.set_toas()
        self.update_callbacks = updates

    def call_updates(self):
        if not self.update_callbacks is None:
            for ucb in self.update_callbacks:
                ucb()

    def reset(self):
        self.psr.reset_TOAs()
        self.set_toas()
        self.call_updates()

    def set_toas(self, newpsr=None):
        # if pulsar updated in pintk, update here
        if newpsr != None:
            self.psr = newpsr

        self.editor.delete("1.0", tk.END)

        # Pretty much copying TOAs.write_TOA_file here but without creating an
        # intermediate filename
        toas = self.psr.selected_toas
        asfile = "FORMAT 1\n"
        pnChange = False
        if "pn" in toas.table.colnames:
            pnChange = True
            for i in range(len(toas.table["flags"])):
                toas.table["flags"][i]["pn"] = toas.table["pn"][i]
        for time, err, freq, obs, flags in zip(
            toas.table["mjd"],
            toas.table["error"].quantity,
            toas.table["freq"].quantity,
            toas.table["obs"],
            toas.table["flags"],
        ):
            obs_obj = pint.observatory.Observatory.get(obs)
            if "clkcorr" in flags.keys():
                time_out = time - astropy.time.TimeDelta(flags["clkcorr"])
            else:
                time_out = time
            asfile += pint.toa.format_toa_line(
                time_out, err, freq, obs_obj, name="pint", flags=flags, format="TEMPO2"
            )
        if pnChange:
            for flags in toas.table["flags"]:
                del flags["pn"]

        self.editor.insert("1.0", asfile)

    def applyChanges(self):
        tfilename = tempfile.mkstemp()[1]
        tfile = open(tfilename, "w")
        tfile.write(self.editor.get("1.0", "end-1c"))
        tfile.close()
        self.psr.selected_toas = pint.toa.get_TOAs(tfilename)
        os.remove(tfilename)
        self.call_updates()

    def writeTim(self):
        filename = tkFileDialog.asksaveasfilename(title="Choose output tim file")
        try:
            fout = open(filename, "w")
            fout.write(self.editor.get("1.0", "end-1c"))
            fout.close()
            print("Saved timfile to %s" % filename)
        except:
            if filename == () or filename == "":
                print("Write Tim cancelled.")
            else:
                print("Could not save timfile to filename:\t%s" % filename)
