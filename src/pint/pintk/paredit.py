import copy
import os
import tempfile

import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox

import pint.models


class ParChoiceWidget(tk.Frame):
    """
    Lets the user select between the pre-fit and post-fit model for the
    loaded pulsar
    """

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.choose_callback = None
        self.initLayout()

    def initLayout(self):
        self.choice = tk.StringVar()
        self.prefit = tk.Radiobutton(
            self,
            text="Pre-Fit",
            command=self.choose,
            variable=self.choice,
            value="prefit",
        )
        self.prefit.select()
        self.prefit.grid(row=0, column=0)

        self.postfit = tk.Radiobutton(
            self,
            text="Post-Fit",
            command=self.choose,
            variable=self.choice,
            value="postfit",
        )
        self.postfit.grid(row=0, column=1)

    def setCallbacks(self, choose):
        self.choose_callback = choose

    def choose(self):
        self.choose_callback()


class ParActionsWidget(tk.Frame):
    """
    Allows the user to reset the model, apply changes, or save to a parfile
    """

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.reset_callback = None
        self.remove_callback = None
        self.apply_callback = None
        self.write_callback = None
        self.centerPE_callback = None
        self.centerPO_callback = None
        self.centerT0_callback = None

        self.initLayout()

    def initLayout(self):
        button = tk.Button(self, text="Reset Model", command=self.resetParfile)
        button.grid(row=0, column=0)

        button = tk.Button(self, text="Remove Changes", command=self.removeChanges)
        button.grid(row=0, column=1)

        button = tk.Button(self, text="Apply Changes", command=self.applyChanges)
        button.grid(row=0, column=2)

        button = tk.Button(self, text="Write Par", command=self.writePar)
        button.grid(row=0, column=3)

        button = tk.Button(self, text="Center PEPOCH", command=self.centerPEPOCH)
        button.grid(row=0, column=4)

        button = tk.Button(self, text="Center POSEPOCH", command=self.centerPOSEPOCH)
        button.grid(row=1, column=1)

        button = tk.Button(self, text="Center T0", command=self.centerT0)
        button.grid(row=1, column=2)

    def setCallbacks(
        self,
        resetParfile,
        removeChanges,
        applyChanges,
        writePar,
        centerPEPOCH,
        centerPOSEPOCH,
        centerT0,
    ):
        self.reset_callback = resetParfile
        self.remove_callback = removeChanges
        self.apply_callback = applyChanges
        self.write_callback = writePar
        self.centerPE_callback = centerPEPOCH
        self.centerPO_callback = centerPOSEPOCH
        self.centerT0_callback = centerT0

    def resetParfile(self):
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

    def writePar(self):
        if self.write_callback is not None:
            self.write_callback()
        print("Write clicked")

    def centerPEPOCH(self):
        if self.centerPE_callback is not None:
            self.centerPE_callback()
        print("Center PEPOCH clicked")

    def centerPOSEPOCH(self):
        if self.centerPO_callback is not None:
            self.centerPO_callback()
        print("Center POSEPOCH clicked")

    def centerT0(self):
        if self.centerT0_callback is not None:
            self.centerT0_callback()
        print("Center T0 clicked")


class ParWidget(tk.Frame):
    """
    A widget that allows editing and saving of a pulsar parfile
    """

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.psr = None
        self.update_callbacks = None
        self.initLayout()

    def initLayout(self):
        self.choiceWidget = ParChoiceWidget(master=self)
        self.choiceWidget.grid(row=0, column=0, sticky="nw")

        self.editor = tk.Text(self)
        self.editor.grid(row=1, column=0, sticky="nesw")

        scrollbar = tk.Scrollbar(self, command=self.editor.yview)
        scrollbar.grid(row=1, column=1, sticky="nesw")
        self.editor["yscrollcommand"] = scrollbar.set

        self.actionsWidget = ParActionsWidget(master=self)
        self.actionsWidget.grid(row=2, column=0, sticky="w")

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=10)
        self.grid_rowconfigure(2, weight=1)

    def setPulsar(self, psr, updates):
        self.psr = psr

        self.choiceWidget.setCallbacks(self.set_model)
        self.actionsWidget.setCallbacks(
            self.reset,
            self.set_model,
            self.applyChanges,
            self.writePar,
            self.centerPEPOCH,
            self.centerPOSEPOCH,
            self.centerT0,
        )
        self.set_model()
        self.update_callbacks = updates

    def call_updates(self):
        if not self.update_callbacks is None:
            for ucb in self.update_callbacks:
                ucb()

    def reset(self):
        self.choiceWidget.prefit.select()
        self.psr.reset_model()
        self.set_model()
        self.call_updates()

    def set_model(self, newpsr=None):
        # if the pulsar was updated in pintk, update here
        if newpsr != None:
            self.psr = newpsr
        choice = self.choiceWidget.choice.get()
        if choice == "postfit":
            if self.psr.fitted:
                self.editor.delete("1.0", tk.END)
                self.editor.insert("1.0", self.psr.postfit_model.as_parfile())
            else:
                print("There is no postfit model yet!")
                self.choiceWidget.prefit.select()
        elif choice == "prefit":
            self.editor.delete("1.0", tk.END)
            self.editor.insert("1.0", self.psr.prefit_model.as_parfile())

    def applyChanges(self):
        pfilename = tempfile.mkstemp()[1]
        pfile = open(pfilename, "w")
        pfile.write(self.editor.get("1.0", "end-1c"))
        pfile.close()
        if self.psr.fitted:
            # if pulsar already fitted, add changes to postfit model as well
            self.psr.postfit_model = pint.models.get_model(pfilename)
        self.psr.prefit_model = pint.models.get_model(pfilename)
        os.remove(pfilename)
        self.call_updates()

    def writePar(self):
        filename = tkFileDialog.asksaveasfilename(title="Choose output par file")
        try:
            fout = open(filename, "w")
            fout.write(self.editor.get("1.0", "end-1c"))
            fout.close()
            print("Saved parfile to %s" % filename)
        except:
            if filename == () or filename == "":
                print("Write Par cancelled.")
            else:
                print("Could not save parfile to filename:\t%s" % filename)

    def centerPEPOCH(self):
        if not hasattr(self.psr.prefit_model, "PEPOCH"):
            print("No PEPOCH to center.")
            return
        mintime, maxtime = (
            self.psr.all_toas.get_mjds().min(),
            self.psr.all_toas.get_mjds().max(),
        )
        midpoint = (mintime + maxtime) / 2
        if self.psr.fitted:
            self.psr.postfit_model.change_pepoch(midpoint)
        self.psr.prefit_model.change_pepoch(midpoint)
        self.set_model()
        self.applyChanges()

    def centerPOSEPOCH(self):
        if not hasattr(self.psr.prefit_model, "POSEPOCH"):
            print("No POSEPOCH to center.")
            return
        mintime, maxtime = (
            self.psr.all_toas.get_mjds().min(),
            self.psr.all_toas.get_mjds().max(),
        )
        midpoint = (mintime + maxtime) / 2
        if self.psr.fitted:
            self.psr.postfit_model.change_posepoch(midpoint)
        self.psr.prefit_model.change_posepoch(midpoint)
        self.set_model()
        self.applyChanges()

    def centerT0(self):
        if not hasattr(self.psr.prefit_model, "T0"):
            print("No T0 to center.")
            return
        mintime, maxtime = (
            self.psr.all_toas.get_mjds().min(),
            self.psr.all_toas.get_mjds().max(),
        )
        midpoint = (mintime + maxtime) / 2
        if self.psr.fitted:
            self.psr.postfit_model.change_binary_epoch(midpoint)
        self.psr.prefit_model.change_binary_epoch(midpoint)
        self.set_model()
        self.applyChanges()
