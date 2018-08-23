'''
Interactive emulator of tempo2 plk
'''

from __future__ import print_function, division
import os, sys

import Tkinter as tk
import tkFileDialog
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import copy

import pulsar as pu

class PlkFitBoxesWidget(tk.Frame):
    '''
    Allows one to select which parameters to fit for
    '''
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)
        self.boxChecked = None
        self.maxcols = 8
    
    def setCallbacks(self, boxChecked, model):
        '''
        Set the callback functions
        @param boxChecked:  Callback function when box is checked
        @param model:       psr.model
        '''
        self.boxChecked = boxChecked
        self.addFitCheckBoxes(model)

    def addFitCheckBoxes(self, model):
        '''
        Add the fitting checkboxes to the frame
        '''
        self.deleteFitCheckBoxes()

        self.compGrids = []
        self.compCBs = []
        self.compVisible = []
        self.parVars = {}

        ii = 0
        comps = model.components.keys()
        fitparams = [p for p in model.params if not getattr(model, p).frozen]
        print(fitparams)
        for comp in comps:
            showpars = [p for p in model.components[comp].params \
                if not p in pu.nofitboxpars and getattr(model, p).quantity is not None]
            #Don't bother showing components without any fittable parameters
            if len(showpars) == 0:
                continue

            self.compVisible.append(tk.IntVar())
            self.compCBs.append(tk.Checkbutton(self, text=comp,
                variable=self.compVisible[ii], command=self.updateLayout))
            
            self.compGrids.append([])
            for pp, par in enumerate(showpars):
                self.parVars[par] = tk.IntVar()
                self.compGrids[ii].append(tk.Checkbutton(self, text=par,
                    variable=self.parVars[par], 
                    command=lambda p=par: self.changedFitCheckBox(p)))
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
            self.compCBs[ii].grid(row=rowCount, column=0, sticky='W')
            if self.compVisible[ii].get():
                for pp, cb in enumerate(self.compGrids[ii]):
                    row = int(pp / self.maxcols)
                    col = pp % self.maxcols + 1
                    cb.grid(row=rowCount+row, column=col, sticky='W')
                rowCount += int(len(self.compGrids[ii]) / self.maxcols) 
            rowCount += 1

    def changedFitCheckBox(self, par):
        if self.boxChecked is not None:
            self.boxChecked(par, bool(self.parVars[par].get()))
        print('%s set to %d' % (par, self.parVars[par].get()))

class PlkXYChoiceWidget(tk.Frame):
    '''
    Allows one to choose which quantities to plot against one another
    '''
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

        for ii, choice in enumerate(pu.plot_labels):
            label = tk.Label(self, text=choice)
            label.grid(row=ii+1, column=0)

            radio = tk.Radiobutton(self, variable=self.xvar, value=choice, 
                                   command=self.updateChoice)
            if choice.lower() == 'mjd':
                radio.select()
            radio.grid(row=ii+1, column=1)

            radio = tk.Radiobutton(self, variable=self.yvar, value=choice, 
                                   command=self.updateChoice)
            if choice.lower() == 'post-fit':
                radio.select()
            radio.grid(row=ii+1, column=2)

    def setCallbacks(self, updatePlot):
        '''
        Set the callback functions
        '''
        self.updatePlot = updatePlot

    def plotids(self):
        return self.xvar.get(), self.yvar.get()

    def updateChoice(self):
        if self.updatePlot is not None:
            self.updatePlot()

class PlkActionsWidget(tk.Frame):
    '''
    Shows action items like re-fit, write par, write tim, etc.
    '''
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)

        self.updatePlot=None
        self.reFit_callback=None
        self.clearAll_callback=None
        self.writePar_callback=None
        self.writeTim_callback=None
        self.saveFig_callback=None

        self.initPlkActions()
    
    def initPlkActions(self):
        button = tk.Button(self, text='Re-fit', command=self.reFit)
        button.grid(row=0, column=0)

        button = tk.Button(self, text='Clear', command=self.clearAll)
        button.grid(row=0, column=1)

        button = tk.Button(self, text='Write par', command=self.writePar)
        button.grid(row=0, column=2)

        button = tk.Button(self, text='Write tim', command=self.writeTim)
        button.grid(row=0, column=3)
        
        button = tk.Button(self, text='Save fig', command=self.saveFig)
        button.grid(row=0, column=4)
    
    def setCallbacks(self, updatePlot, reFit, writePar, writeTim, saveFig):
        """
        Callback functions
        """
        self.updatePlot = updatePlot
        self.reFit_callback = reFit
        self.writePar_callback = writePar
        self.writeTim_callback = writeTim
        self.saveFig_callback = saveFig
        
    def reFit(self):
        if self.reFit_callback is not None:
            self.reFit_callback()

    def writePar(self):
        if self.writePar_callback is not None:
            self.writePar_callback()
        print("Write Par clicked")

    def writeTim(self):
        if self.writeTim_callback is not None:
            self.writeTim_callback()
        print("Write Tim clicked")

    def clearAll(self):
        print("Clear clicked")

    def saveFig(self):
        if self.saveFig_callback is not None:
            self.saveFig_callback()
        print("Save fig clicked")

class PlkWidget(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)

        self.initPlk()
        self.initPlkLayout()

        self.psr = None

    def initPlk(self):
        self.fitboxesWidget = PlkFitBoxesWidget(master=self)
        self.xyChoiceWidget = PlkXYChoiceWidget(master=self)
        self.actionsWidget = PlkActionsWidget(master=self)

        self.plkDpi = 100
        self.plkFig = mpl.figure.Figure(dpi=self.plkDpi)
        self.plkCanvas = FigureCanvasTkAgg(self.plkFig, self)

        self.plkAxes = self.plkFig.add_subplot(111)

        self.drawSomething()

    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel('MJD')
        self.plkAxes.set_ylabel('Residual ($\mu$s)')
        self.plkCanvas.draw()

    def initPlkLayout(self):
        self.xyChoiceWidget.grid(row=1, column=0)
        self.plkCanvas.get_tk_widget().grid(row=1, column=1)
        self.actionsWidget.grid(row=2, column=0, columnspan=2, sticky='W')

        self.grid_columnconfigure(1, weight=10)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=10)
        self.grid_rowconfigure(2, weight=1)

    def setPulsar(self, psr):
        self.psr = psr

        self.fitboxesWidget.setCallbacks(self.fitboxChecked, psr._model)
        self.xyChoiceWidget.setCallbacks(self.updatePlot)
        self.actionsWidget.setCallbacks(self.updatePlot, self.reFit, 
            self.writePar, self.writeTim, self.saveFig)

        self.fitboxesWidget.grid(row=0, column=0, columnspan=2, sticky='W')

        self.xyChoiceWidget.updateChoice()
    
    def fitboxChecked(self, parchanged, newstate):
        """
        When a fitbox is (un)checked, this callback function is called

        @param parchanged:  Which parameter has been (un)checked
        @param newstate:    The new state of the checkbox
        """
        self.psr.set_fit_state(parchanged, newstate)

    def reFit(self):
        """
        We need to re-do the fit for this pulsar
        """
        if not self.psr is None:
            self.psr.fit()
            self.updatePlot()

    def writePar(self):
        '''
        Write the fit parfile to a file
        '''
        filename = tkFileDialog.asksaveasfilename(title='Choose output par file')
        try:
            fout = open(filename, 'w')
            fout.write(self.psr._fitter.model.as_parfile())
            fout.close()
            print('Saved post-fit parfile to %s' % filename)
        except:
            print('Count not save parfile to filename:\t%s' % filename)

    def writeTim(self):
        '''
        Write the current timfile to a file
        '''
        filename = tkFileDialog.asksaveasfilename(title='Choose output tim file')
        try:
            print('Chose output file %s' % filename)
        except:
            print('Count not save file to filename:\t%s' % filename)

    def saveFig(self):
        '''
        Save the current plot view to a file
        '''
        filename = tkFileDialog.asksaveasfilename(title='Choose output file')
        try:
            self.plkFig.savefig(filename)
            print('Saved image to %s' % filename)
        except:
            print('Could not save figure to filename:\t%s' % filename)

    def newFitParameters(self):
        """
        This function is called when we have new fitparameters

        TODO: callback not used right now
        """
        pass
    
    def updatePlot(self):
        """
        Update the plot/figure
        """
        self.plkAxes.clear()
        self.plkAxes.grid(True)

        if self.psr is not None:
            # Get a mask for the plotting points
            msk = self.psr.mask('plot')

            #print("Mask has {0} toas".format(np.sum(msk)))

            # Get the IDs of the X and Y axis
            xid, yid = self.xyChoiceWidget.plotids()

            # Retrieve the data
            x, xerr, xlabel = self.psr.data_from_label(xid)
            y, yerr, ylabel = self.psr.data_from_label(yid)

            if x is not None and y is not None and np.sum(msk) > 0:
                xp = x[msk]
                yp = y[msk]

                if yerr is not None:
                    yerrp = yerr[msk]
                else:
                    yerrp = None

                self.plotResiduals(xp, yp, yerrp, xlabel, ylabel, self.psr.name)

                if xid in ['mjd', 'year', 'rounded MJD']:
                    self.plotPhaseJumps(self.psr.phasejumps())
            else:
                raise ValueError("Nothing to plot!")

        self.plkCanvas.draw()

    def plotResiduals(self, x, y, yerr, xlabel, ylabel, title):
        """
        Update the plot, given all the plotting info
        """
        xave = 0.5 * (np.max(x) + np.min(x))
        xmin = xave - 1.05 * (xave - np.min(x))
        xmax = xave + 1.05 * (np.max(x) - xave)
        if yerr is None:
            yave = 0.5 * (np.max(y) + np.min(y))
            ymin = yave - 1.05 * (yave - np.min(y))
            ymax = yave + 1.05 * (np.max(y) - yave)
            self.plkAxes.scatter(x, y, marker='.', color='blue')
        else:
            yave = 0.5 * (np.max(y+yerr) + np.min(y-yerr))
            ymin = yave - 1.05 * (yave - np.min(y-yerr))
            ymax = yave + 1.05 * (np.max(y+yerr) - yave)
            self.plkAxes.errorbar(x.reshape([-1, 1]), y.reshape([-1, 1]), \
                                  yerr=yerr.reshape([-1, 1]), fmt='.', color='blue')

        self.plkAxes.axis([xmin.value, xmax.value, ymin.value, ymax.value])
        self.plkAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.plkAxes.set_xlabel(xlabel)
        self.plkAxes.set_ylabel(ylabel)
        self.plkAxes.set_title(title, y=1.03)
    
    def plotPhaseJumps(self, phasejumps):
        """
        Plot the phase jump lines, if we have any
        """
        xmin, xmax, ymin, ymax = self.plkAxes.axis()
        dy = 0.01 * (ymax-ymin)

        if len(phasejumps) > 0:
            phasejumps = np.array(phasejumps)

            for ii in range(len(phasejumps)):
                if phasejumps[ii,1] != 0:
                    self.plkAxes.vlines(phasejumps[ii,0], ymin, ymax,
                            color='darkred', linestyle='--', linewidth=0.5)

                    if phasejumps[ii,1] < 0:
                        jstr = str(phasejumps[ii,1])
                    else:
                        jstr = '+' + str(phasejumps[ii,1])

                    # Print the jump size above the plot
                    ann = self.plkAxes.annotate(jstr, \
                            xy=(phasejumps[ii,0], ymax+dy), xycoords='data', \
                            annotation_clip=False, color='darkred', \
                            size=7.0)
                    
    def setFocusToCanvas(self):
        """
        Set the focus to the plk Canvas
        """
        self.plkCanvas.setFocus()
