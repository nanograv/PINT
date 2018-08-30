'''
Interactive emulator of tempo2 plk
'''

from __future__ import print_function, division
import os, sys

import Tkinter as tk
import tkFileDialog
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import numpy as np
import astropy.units as u
import copy

import pint.pintkinter.pulsar as pu

plotlabels = {'pre-fit': [r'Pre-fit residual ($\mu$s)', 'Pre-fit residual (phase)'],
              'post-fit': [r'Post-fit residual ($\mu$s)', 'Post-fit residual (phase)'],
              'mjd': r'MJD',
              'orbital phase': 'Orbital Phase',
              'serial': 'TOA number',
              'day of year': 'Day of the year',
              'year': 'Year',
              'frequency': r'Observing Frequency (MHz)',
              'TOA error': r'TOA uncertainty ($\mu$s)',
              'elevation': None,
              'rounded MJD': r'MJD',
              'sidereal time': None,
              'hour angle': None,
              'para. angle': None}

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

class PlkToolbar(NavigationToolbar2TkAgg):
    '''
    A modification of the stock Matplotlib toolbar to perform the
    necessary selections/unselections on points
    '''
    toolitems = [t for t in NavigationToolbar2TkAgg.toolitems if
                 t[0] in ('Back', 'Zoom', 'Save')]
    def __init__(self, *args, **kwargs):
        self.back_callback = None
        self.draw_callback = None
        
        NavigationToolbar2TkAgg.__init__(self, *args, **kwargs)
    
    def setCallbacks(self, back_callback, draw_callback):
        self.back_callback = back_callback
        self.draw_callback = draw_callback

    def back(self, *args):
        if not self.back_callback is None:
            self.back_callback()
        NavigationToolbar2TkAgg.back(self, *args)

    def release_zoom(self, event):
        NavigationToolbar2TkAgg.release_zoom(self, event)
        NavigationToolbar2TkAgg.zoom(self)

    def draw(self):
        NavigationToolbar2TkAgg.draw(self)
        #Now that it has been redrawn, execute callback
        if not self.draw_callback is None:
            self.draw_callback()

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
    
    def setCallbacks(self, fit, reset, writePar, writeTim):
        """
        Callback functions
        """
        self.fit_callback = fit
        self.reset_callback = reset
        self.writePar_callback = writePar
        self.writeTim_callback = writeTim
    
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

class PlkWidget(tk.Frame):
    helpstring = '''The following interactions are currently supported by the Plk pane in the PINTkinter GUI:

Left click:     Highlight a point

Right click:    Delete a point

r:              Reset the pane - undo all deletions, selections, etc.

f:              Perform a fit

s:              Select the highlights points

d:              Delete the highlighted points

u:              Undo the most recent selection

c:              Clear highlighter from map

h:              Print help
'''

    clickDist = 0.0005

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
        self.plkToolbar = PlkToolbar(self.plkCanvas, tk.Frame(self))

        self.plkAxes = self.plkFig.add_subplot(111)
        self.plkAx2x = self.plkAxes.twinx()
        self.plkAx2y = self.plkAxes.twiny()

        self.drawSomething()

    def drawSomething(self):
        self.plkAxes.clear()
        self.plkAxes.grid(True)
        self.plkAxes.set_xlabel('MJD')
        self.plkAxes.set_ylabel('Residual ($\mu$s)')
        self.plkCanvas.draw()

    def initPlkLayout(self):
        self.plkToolbar.master.grid(row=1, column=1, sticky='nesw')
        self.xyChoiceWidget.grid(row=2, column=0, sticky='nw')
        self.plkCanvas.get_tk_widget().grid(row=2, column=1, sticky='nesw')
        self.actionsWidget.grid(row=3, column=0, columnspan=2, sticky='W')

        self.grid_columnconfigure(1, weight=10)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=10)
        self.grid_rowconfigure(3, weight=1)

    def update(self):
        if self.psr is not None:
            self.psr.update_resids()
            self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
            self.actionsWidget.setFitButtonText('Fit')
            self.fitboxesWidget.addFitCheckBoxes(self.psr.prefit_model)
            self.xyChoiceWidget.setChoice()
            self.updatePlot()

    def setPulsar(self, psr, updates):
        self.psr = psr
        self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
        self.update_callbacks = updates

        self.fitboxesWidget.setCallbacks(self.fitboxChecked)
        self.xyChoiceWidget.setCallbacks(self.updatePlot)
        self.plkToolbar.setCallbacks(self.unselect, self.zoom_select)
        self.actionsWidget.setCallbacks(self.fit, self.reset,
            self.writePar, self.writeTim)

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
        if self.psr.fitted:
            getattr(self.psr.postfit_model, parchanged).frozen = not newstate
        self.call_updates()

    def unselect(self):
        '''
        Undo a selection (but not deletes)
        '''
        self.psr.toas.unselect()
        self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
        self.psr.update_resids()
        self.updatePlot()
        self.call_updates()

    def zoom_select(self):
        '''
        Apply a TOAs selection to points within the current view
        '''
        xmin, xmax = self.plkAxes.get_xlim()
        ymin, ymax = self.plkAxes.get_ylim()

        self.selected = (self.xvals.value > xmin) & (self.xvals.value < xmax)
        self.selected &= (self.yvals.value > ymin) & (self.yvals.value < ymax)
        self.updatePlot()

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
        Reset all plot changes for this pulsar
        '''
        self.psr.reset_TOAs()
        self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
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

    def updatePlot(self):
        """
        Update the plot/figure
        """
        self.plkAxes.clear()
        self.plkAx2x.clear()
        self.plkAx2y.clear()
        self.plkAxes.grid(True)

        if self.psr is not None:
            # Get a mask for the plotting points
            #msk = self.psr.mask('plot')

            # Get the IDs of the X and Y axis
            self.xid, self.yid = self.xyChoiceWidget.plotIDs()

            # Retrieve the data
            x, xerr = self.psr_data_from_label(self.xid)
            y, yerr = self.psr_data_from_label(self.yid)

            if x is not None and y is not None:
                self.xvals = x
                self.yvals = y
                self.yerrs = yerr

                self.plotResiduals()

                if self.xid in ['mjd', 'year', 'rounded MJD']:
                    self.plotPhaseJumps(self.psr.phasejumps())
            else:
                raise ValueError("Nothing to plot!")

        self.plkFig.tight_layout()
        self.plkCanvas.draw()

    def plotErrorbar(self, selected, color):
        '''
        For some reason, errorbar breaks completely when the plotting array is
        of length 2. So this workaround is needed
        '''
        if selected.sum() != 2:
            self.plkAxes.errorbar(self.xvals[selected].reshape([-1, 1]), 
                                  self.yvals[selected].reshape([-1, 1]),
                                  yerr=self.yerrs[selected].reshape([-1, 1]),
                                  fmt='.', color=color)
        else:
            self.plkAxes.errorbar(self.xvals[selected][0].reshape([-1, 1]), 
                                  self.yvals[selected][0].reshape([-1, 1]), 
                                  yerr=self.yerrs[selected][0].reshape([-1, 1]),
                                  fmt='.', color=color) 
            self.plkAxes.errorbar(self.xvals[selected][1].reshape([-1, 1]), 
                                  self.yvals[selected][1].reshape([-1, 1]), 
                                  yerr=self.yerrs[selected][1].reshape([-1, 1]),
                                  fmt='.', color=color)  

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
            self.plkAxes.scatter(self.xvals[~self.selected], self.yvals[~self.selected],   
                marker='.', color='blue')
            self.plkAxes.scatter(self.xvals[self.selected], self.yvals[self.selected],
                marker='.', color='orange')
        else:
            yave = 0.5 * (np.max(self.yvals+self.yerrs) + np.min(self.yvals-self.yerrs))
            ymin = yave - 1.05 * (yave - np.min(self.yvals-self.yerrs))
            ymax = yave + 1.05 * (np.max(self.yvals+self.yerrs) - yave)
            self.plotErrorbar(~self.selected, color='blue')
            self.plotErrorbar(self.selected, color='orange')

        self.plkAxes.axis([xmin.value, xmax.value, ymin.value, ymax.value])
        self.plkAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.plkAx2y.set_visible(False)
        self.plkAx2x.set_visible(False)

        if self.xid in ['pre-fit', 'post-fit']:
            self.plkAxes.set_xlabel(plotlabels[self.xid][0])
            m = self.psr.prefit_model if self.xid == 'pre-fit' or not self.psr.fitted \
                                      else self.psr.postfit_model
            if hasattr(m, 'F0'):
                self.plkAx2y.set_visible(True)
                self.plkAx2y.set_xlabel(plotlabels[self.xid][1])
                f0 = m.F0.quantity.to(u.Hz)
                self.plkAx2y.set_xlim((xmin.to(u.s) * f0).value, (xmax.to(u.s) * f0).value)
                self.plkAx2y.xaxis.set_major_locator(mpl.ticker.FixedLocator(
                    self.plkAxes.get_xticks() * f0.to(u.MHz).value))
        else:
            self.plkAxes.set_xlabel(plotlabels[self.xid])

        if self.yid in ['pre-fit', 'post-fit']:
            self.plkAxes.set_ylabel(plotlabels[self.yid][0])
            m = self.psr.prefit_model if self.yid == 'pre-fit' or not self.psr.fitted \
                                      else self.psr.postfit_model
            if hasattr(m, 'F0'):
                self.plkAx2x.set_visible(True)
                self.plkAx2x.set_ylabel(plotlabels[self.yid][1])
                f0 = m.F0.quantity.to(u.Hz)
                self.plkAx2x.set_ylim((ymin.to(u.s) * f0).value, (ymax.to(u.s) * f0).value)
                self.plkAx2x.yaxis.set_major_locator(mpl.ticker.FixedLocator(
                    self.plkAxes.get_yticks() * f0.to(u.MHz).value))
        else:
            self.plkAxes.set_ylabel(plotlabels[self.yid])
        
        self.plkAxes.set_title(self.psr.name, y=1.1)
    
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
        @return:    data, error
        '''
        data, error = None, None
        if label == 'pre-fit':
            data = self.psr.prefit_resids.time_resids.to(u.us)
            error = self.psr.toas.get_errors().to(u.us)
        elif label == 'post-fit':
            if self.psr.fitted:
                data = self.psr.postfit_resids.time_resids.to(u.us)
            else:
                print('Pulsar has not been fitted yet! Giving pre-fit residuals')
                data = self.psr.prefit_resids.time_resids.to(u.us)
            error = self.psr.toas.get_errors().to(u.us)
        elif label == 'mjd':
            data = self.psr.toas.get_mjds()
            error = self.psr.toas.get_errors()
        elif label == 'orbital phase':
            data = self.psr.orbitalphase()
            error = None
        elif label == 'serial':
            data = np.arange(self.psr.toas.ntoas) * u.m / u.m
            error = None
        elif label == 'day of year':
            data = self.psr.dayofyear()
            error = None
        elif label == 'year':
            data = self.psr.year()
            error = None
        elif label == 'frequency':
            data = self.psr.toas.get_freqs()
            error = None
        elif label == 'TOA error':
            data = self.psr.toas.get_errors().to(u.us)
            error = None
        elif label == 'elevation':
            print('WARNING: parameter {0} not yet implemented'.format(label))
        elif label == 'rounded MJD':
            data = np.floor(self.psr.toas.get_mjds() + 0.5 * u.d)
            error = self.psr.toas.get_errors().to(u.d)
        elif label == 'sidereal time':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'hour angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
        elif label == 'para. angle':
            print("WARNING: parameter {0} not yet implemented".format(label))
       
        return data, error

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
            #print('Closest point is %d:(%s, %s) at d=%f with next closest point at d=%f' % (ind, self.xvals[ind], self.yvals[ind], dist[ind], dist[ind2]))
            
            if dist[ind] > PlkWidget.clickDist:
                print('Not close enough to a point')
                ind = None
        
        return ind
    
    def canvasClickEvent(self, event):
        '''
        Call this function when the figure/canvas is clicked 
        '''
        if event.xdata is not None and event.ydata is not None:
            ind = self.coordToPoint(event.xdata, event.ydata)
            if ind is not None:
                if event.button == 1:
                    #Left click is select
                    self.selected[ind] = not self.selected[ind]
                    self.updatePlot()
                elif event.button == 3:
                    #Right click is delete
                    self.psr.toas.table.remove_row(ind)
                    self.psr.toas.table = self.psr.toas.table.group_by('obs')
                    if hasattr(self.psr.toas, 'table_selects'):
                        for i in range(len(self.psr.toas.table_selects)):
                            self.psr.toas.table_selects[i].remove_row(ind)
                            self.psr.toas.table_selects[i] = \
                                self.psr.toas.table_selects[i].group_by('obs')
                    self.selected = np.delete(self.selected, ind)
                    self.psr.update_resids()
                    self.updatePlot()
                    self.call_updates()

    def canvasKeyEvent(self, event):
        '''
        A key is pressed. Handle all the shortcuts here
        '''

        fkey = event.key
        from_canvas = True

        xpos, ypos = event.xdata, event.ydata
        ukey = ord(fkey[-1])

        if ukey == ord('r'):
            #Reset the pane
            self.reset()
        elif ukey == ord('f'):
            #Re-do the fit, using post-fit values of parameters
            self.fit()
        elif ukey == ord('d'):
            #Delete the selected points
            self.psr.toas.table = self.psr.toas.table[~self.selected].group_by('obs')
            if hasattr(self.psr.toas, 'table_selects'):
                for i in range(len(self.psr.toas.table_selects)):
                    self.psr.toas.table_selects[i] = \
                        self.psr.toas.table_selects[i][~self.selected].group_by('obs')
            self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
            self.psr.update_resids()
            self.updatePlot()
            self.call_updates()
        elif ukey == ord('s'):
            #Apply the selection to TOAs object
            self.psr.toas.select(self.selected)
            self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
            self.psr.update_resids()
            self.updatePlot()
            self.call_updates()
        elif ukey == ord('u'):
            self.unselect()
        elif ukey == ord('c'):
            self.selected = np.zeros(self.psr.toas.ntoas, dtype=bool)
            self.updatePlot()
        elif ukey == ord('h'):
            print(PlkWidget.helpstring)
