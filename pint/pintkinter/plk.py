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
import astropy.units as u
import copy

import pint.pintkinter.pulsar as pu

class PlkFitBoxesWidget(tk.Frame):
    '''
    Allows one to select which parameters to fit for
    '''
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)
        self.boxChecked = None
        self.maxcols = 8
    
    def setCallbacks(self, boxChecked):
        '''
        Set the callback functions
        '''
        self.boxChecked = boxChecked

    def addFitCheckBoxes(self, model):
        '''
        Add the fitting checkboxes for the given model to the frame
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

        self.xbuttons = []
        self.ybuttons = []

        for ii, choice in enumerate(pu.plot_labels):
            label = tk.Label(self, text=choice)
            label.grid(row=ii+1, column=0)

            self.xbuttons.append(tk.Radiobutton(self, variable=self.xvar, value=choice, 
                                  command=self.updateChoice))
            self.xbuttons[ii].grid(row=ii+1, column=1)

            self.ybuttons.append(tk.Radiobutton(self, variable=self.yvar, value=choice, 
                                 command=self.updateChoice))
            self.ybuttons[ii].grid(row=ii+1, column=2)

    def setChoice(self, xid='mjd', yid='pre-fit'):
        for ii, choice in enumerate(pu.plot_labels):
            if choice.lower() == xid:
                self.xbuttons[ii].select()
            if choice.lower() == yid:
                self.ybuttons[ii].select()

    def setCallbacks(self, updatePlot):
        '''
        Set the callback functions
        '''
        self.updatePlot = updatePlot

    def plotIDs(self):
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

        self.fit_callback=None
        self.clearAll_callback=None
        self.writePar_callback=None
        self.writeTim_callback=None
        self.saveFig_callback=None

        self.initPlkActions()
    
    def initPlkActions(self):
        self.fitbutton = tk.Button(self, text='Fit', command=self.fit)
        self.fitbutton.grid(row=0, column=0)

        button = tk.Button(self, text='Reset', command=self.reset)
        button.grid(row=0, column=1)

        button = tk.Button(self, text='Write par', command=self.writePar)
        button.grid(row=0, column=2)

        button = tk.Button(self, text='Write tim', command=self.writeTim)
        button.grid(row=0, column=3)
        
        button = tk.Button(self, text='Save fig', command=self.saveFig)
        button.grid(row=0, column=4)
    
    def setCallbacks(self, fit, reset, writePar, writeTim, saveFig):
        """
        Callback functions
        """
        self.fit_callback = fit
        self.reset_callback = reset
        self.writePar_callback = writePar
        self.writeTim_callback = writeTim
        self.saveFig_callback = saveFig
    
    def setFitButtonText(self, text):
        self.fitbutton.config(text=text)

    def fit(self):
        if self.fit_callback is not None:
            self.fit_callback()

    def writePar(self):
        if self.writePar_callback is not None:
            self.writePar_callback()
        print("Write Par clicked")

    def writeTim(self):
        if self.writeTim_callback is not None:
            self.writeTim_callback()
        print("Write Tim clicked")

    def reset(self):
        if self.reset_callback is not None:
            self.reset_callback()
        print("Reset clicked")

    def saveFig(self):
        if self.saveFig_callback is not None:
            self.saveFig_callback()
        print("Save fig clicked")

class PlkWidget(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master)

        self.initPlk()
        self.initPlkLayout()

        self.update_callbacks = None

        self.psr = None

    def initPlk(self):
        self.fitboxesWidget = PlkFitBoxesWidget(master=self)
        self.xyChoiceWidget = PlkXYChoiceWidget(master=self)
        self.actionsWidget = PlkActionsWidget(master=self)

        self.plkDpi = 100
        self.plkFig = mpl.figure.Figure(dpi=self.plkDpi)
        self.plkCanvas = FigureCanvasTkAgg(self.plkFig, self)
        self.plkCanvas.mpl_connect('button_press_event', self.canvasClickEvent)
        self.plkCanvas.mpl_connect('key_press_event', self.canvasKeyEvent)

        self.plkAxes = self.plkFig.add_subplot(111)

        self.drawSomething()

    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel('MJD')
        self.plkAxes.set_ylabel('Residual ($\mu$s)')
        self.plkCanvas.draw()

    def initPlkLayout(self):
        self.xyChoiceWidget.grid(row=1, column=0, sticky='nw')
        self.plkCanvas.get_tk_widget().grid(row=1, column=1, sticky='nesw')
        self.actionsWidget.grid(row=2, column=0, columnspan=2, sticky='W')

        self.grid_columnconfigure(1, weight=10)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=10)
        self.grid_rowconfigure(2, weight=1)

    def update(self):
        if self.psr is not None:
            print('Updating')
            self.psr.update_resids()
            self.actionsWidget.setFitButtonText('Fit')
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            self.xyChoiceWidget.setChoice()
            self.updatePlot()

    def setPulsar(self, psr, updates):
        self.psr = psr
        self.update_callbacks = updates

        self.fitboxesWidget.setCallbacks(self.fitboxChecked)
        self.xyChoiceWidget.setCallbacks(self.updatePlot)
        self.actionsWidget.setCallbacks(self.fit, self.reset,
            self.writePar, self.writeTim, self.saveFig)

        self.fitboxesWidget.grid(row=0, column=0, columnspan=2, sticky='W')

        self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
        self.xyChoiceWidget.setChoice()
        self.updatePlot()

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
        self.call_updates()

    def fit(self):
        """
        We need to re-do the fit for this pulsar
        """
        if not self.psr is None:
            self.psr.fit()
            self.actionsWidget.setFitButtonText('Re-fit')
            xid, yid = self.xyChoiceWidget.plotIDs()
            self.xyChoiceWidget.setChoice(xid=xid, yid='post-fit')
            self.updatePlot()
        self.call_updates()

    def reset(self):
        '''
        Reset all changes for this pulsar
        '''
        self.psr.reset_TOAs()
        self.actionsWidget.setFitButtonText('Fit')
        self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
        self.xyChoiceWidget.setChoice()
        self.updatePlot()
        self.call_updates()

    def writePar(self):
        '''
        Write the fit parfile to a file
        '''
        filename = tkFileDialog.asksaveasfilename(title='Choose output par file')
        try:
            fout = open(filename, 'w')
            if self.psr.fitted:
                fout.write(self.psr.postfit_model.as_parfile())
                print('Saved post-fit parfile to %s' % filename)
            else:
                fout.write(self.psr.prefit_model.as_parfile())
                print('Pulsar has not been fitted! Saving pre-fit parfile to %s' % filename)
            fout.close()
        except:
            print('Could not save parfile to filename:\t%s' % filename)

    def writeTim(self):
        '''
        Write the current timfile to a file
        '''
        filename = tkFileDialog.asksaveasfilename(title='Choose output tim file')
        try:
            print('Chose output file %s' % filename)
            self.psr.toas.write_TOA_file(filename)
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

    def updatePlot(self):
        """
        Update the plot/figure
        """
        self.plkAxes.clear()
        self.plkAxes.grid(True)

        if self.psr is not None:
            # Get a mask for the plotting points
            #msk = self.psr.mask('plot')

            # Get the IDs of the X and Y axis
            xid, yid = self.xyChoiceWidget.plotIDs()

            # Retrieve the data
            x, xerr, xlabel = self.psr_data_from_label(xid)
            y, yerr, ylabel = self.psr_data_from_label(yid)

            if x is not None and y is not None:
                self.xvals, self.xlabel = x, xlabel
                self.yvals, self.ylabel = y, ylabel
                self.yerrs = yerr

                self.plotResiduals()

                if xid in ['mjd', 'year', 'rounded MJD']:
                    self.plotPhaseJumps(self.psr.phasejumps())
            else:
                raise ValueError("Nothing to plot!")

        self.plkCanvas.draw()

    def plotResiduals(self):
        """
        Update the plot, given all the plotting info
        """
        xave = 0.5 * (np.max(self.xvals) + np.min(self.xvals))
        xmin = xave - 1.05 * (xave - np.min(self.xvals))
        xmax = xave + 1.05 * (np.max(self.xvals) - xave)
        if self.yerrs is None:
            yave = 0.5 * (np.max(self.yvals) + np.min(self.yvals))
            ymin = yave - 1.05 * (yave - np.min(self.yvals))
            ymax = yave + 1.05 * (np.max(self.yvals) - yave)
            self.plkAxes.scatter(self.xvals, self.yvals, marker='.', color='blue')
        else:
            yave = 0.5 * (np.max(self.yvals+self.yerrs) + np.min(self.yvals-self.yerrs))
            ymin = yave - 1.05 * (yave - np.min(self.yvals-self.yerrs))
            ymax = yave + 1.05 * (np.max(self.yvals+self.yerrs) - yave)
            self.plkAxes.errorbar(self.xvals.reshape([-1, 1]), self.yvals.reshape([-1, 1]), \
                                  yerr=self.yerrs.reshape([-1, 1]), fmt='.', color='blue')

        self.plkAxes.axis([xmin.value, xmax.value, ymin.value, ymax.value])
        self.plkAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        if type(self.xlabel) == list:
            self.plkAxes.set_xlabel(self.xlabel[0])
        else:
            self.plkAxes.set_xlabel(self.xlabel)
        if type(self.ylabel) == list:
            self.plkAxes.set_ylabel(self.ylabel[0])
        else:
            self.plkAxes.set_ylabel(self.ylabel)
        self.plkAxes.set_title(self.psr.name, y=1.03)
    
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
    
    def psr_data_from_label(self, label):
        '''
        Given a label, get the corresponding data from the pulsar
        
        @param label: The label for the data we want
        @return:    data, error, plotlabel
        '''
        data, error, plotlabel = None, None, None
        if label == 'pre-fit':
            data = self.psr.prefit_resids.time_resids.to(u.us)
            error = self.psr.toas.get_errors().to(u.us)
            plotlabel=[r'Pre-fit residual ($\mu$s)', 'Pre-fit residual (phase)']
        elif label == 'post-fit':
            if self.psr.fitted:
                data = self.psr.postfit_resids.time_resids.to(u.us)
            else:
                print('Pulsar has not been fitted yet! Giving pre-fit residuals')
                data = self.psr.prefit_resids.time_resids.to(u.us)
            error = self.psr.toas.get_errors().to(u.us)
            plotlabel=[r'Post-fit residual ($\mu$s)', 'Post-fit residual (phase)']
        elif label == 'mjd':
            data = self.psr.toas.get_mjds()
            error = self.psr.toas.get_errors()
            plotlabel = r'MJD'
        elif label == 'orbital phase':
            data = self.psr.orbitalphase()
            error = None
            plotlabel = 'Orbital Phase'
        elif label == 'serial':
            data = np.arange(self.psr.toas.ntoas) * u.m / u.m
            error = None
            plotlabel = 'TOA number'
        elif label == 'day of year':
            data = self.psr.dayofyear()
            error = None
            plotlabel = 'Day of the year'
        elif label == 'year':
            data = self.psr.year()
            error = None
            plotlabel = 'Year'
        elif label == 'frequency':
            data = self.psr.toas.get_freqs()
            error = None
            plotlabel = r"Observing frequency (MHz)"
        elif label == 'TOA error':
            data = self.psr.toas.get_errors().to(u.us)
            error = None
            plotlabel = "TOA uncertainty ($\mu$s)"
        elif label == 'elevation':
            print('WARNING: parameter {0} not yet implemented'.format(label))
        elif label == 'rounded MJD':
            data = np.floor(self.psr.toas.get_mjds() + 0.5 * u.d)
            error = self.psr.toas.get_errors().to(u.d)
            plotlabel = r'MJD'
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
       
        return data, error, plotlabel

    def setFocusToCanvas(self):
        """
        Set the focus to the plk Canvas
        """
        self.plkCanvas.setFocus()

    def coordToPoint(self, cx, cy):
        '''
        Given a set of x-y coordinates, get the TOA index closest to it
        '''
        ind = None

        if self.psr is not None:
            x = self.xvals.value
            y = self.yvals.value

            xmin, xmax, ymin, ymax = self.plkAxes.axis()
            dist = ((x-cx)/(xmax-xmin))**2.0 + ((y-cy)/(ymax-ymin))**2.0
            ind = np.argmin(dist)
            val = dist[ind]
            dist[ind] = 100000
            ind2= np.argmin(dist)
            dist[ind] = val
            print('Closest point is %d:(%s, %s) at d=%f with next closest point at d=%f' % (ind, self.xvals[ind], self.yvals[ind], dist[ind], dist[ind2]))

            if dist[ind] * 2  > dist[ind2]:
                print('Selection is unclear between two points')
                ind = None

        return ind
    
    def canvasClickEvent(self, event):
        '''
        Call this function when the figure/canvas is clicked 
        '''
        if event.xdata is not None and event.ydata is not None:
            ind = self.coordToPoint(event.xdata, event.ydata)
            if ind is not None:
                #Right click is delete
                if event.button == 3:
                    self.psr.toas.table.remove_row(ind)
                    self.psr.toas.table = self.psr.toas.table.group_by('obs')
                    self.psr.update_resids()
                    self.updatePlot()

    def canvasKeyEvent(self, event):
        '''
        A key is pressed. Handle all the shortcuts here
        '''

        fkey = event.key
        print('%s was pressed' % fkey)
        from_canvas = True

        xpos, ypos = event.xdata, event.ydata
        ukey = ord(fkey[-1])
        print(fkey)

        if ukey == ord('r'):
            #Reset the pane
            self.reset()
        if ukey == ord('x'):
            #Re-do the fit, using post-fit values of parameters
            self.fit()
