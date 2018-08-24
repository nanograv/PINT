#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
'''
PINTkinter: Tkinter interactive interface for PINT pulsar timing tool
'''

from __future__ import absolute_import, print_function, division
import os, sys
import numpy as np
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import code
import argparse

import pint.pintkinter.pulsar as pu
from pint.pintkinter.plk import PlkWidget
from pint.pintkinter.paredit import ParWidget

from astropy import log
log.setLevel('WARNING')

banner = """
      +----------------------------------------------+
      |              PINT                            |
      |              ====              ,~~~~.        |
      |      Modern Pulsar Timing      i====i_       |
      |                                |cccc|_)      |
      |     Brought to you by the      |cccc|        |
      |     NANOGrav collaboration     `-==-'        |
      |                                              |
      +----------------------------------------------+

"""

class PINTkinter(object):
    '''
    Main PINTkinter window
    '''
    def __init__(self, master, parfile=None, timfile=None, **kwargs):
        self.master = master
        self.master.title('Tkinter interface to PINT')

        self.mainFrame = tk.Frame(master=self.master)
        self.mainFrame.grid(row=0, column=0, sticky='nesw')
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.maxcols = 2

        self.createWidgets()
        if parfile is not None and timfile is not None:
            self.openPulsar(parfile=parfile, timfile=timfile)
        
        self.initUI()
        self.updateLayout()

    def initUI(self):
        #Create top level menus
        self.menuBar = tk.Menu(self.mainFrame.winfo_toplevel())
        self.mainFrame.winfo_toplevel()['menu'] = self.menuBar

        self.fileMenu = tk.Menu(self.menuBar)
        self.fileMenu.add_command(label='Open par/tim', command=self.openParTim)
        self.fileMenu.add_command(label='Exit', 
                                 command=self.mainFrame.winfo_toplevel().destroy)
        self.menuBar.add_cascade(label='File', menu=self.fileMenu)

        self.viewMenu = tk.Menu(self.menuBar)
        self.viewMenu.add_checkbutton(label='Plk', command=self.updateLayout,
                                      variable=self.active['plk'])
        self.viewMenu.add_checkbutton(label='Edit Par', command=self.updateLayout,
                                      variable=self.active['par'])
        self.menuBar.add_cascade(label='View', menu=self.viewMenu)

        self.helpMenu = tk.Menu(self.menuBar)
        self.helpMenu.add_command(label='About', command=self.about)
        self.menuBar.add_cascade(label='Help', menu=self.helpMenu)

    def createWidgets(self):
        self.widgets = {'plk': PlkWidget(master=self.mainFrame),
                        'par': ParWidget(master=self.mainFrame)}
        self.active = {'plk': tk.IntVar(), 'par': tk.IntVar()}
        self.active['plk'].set(1)

    def updateLayout(self):
        for widget in self.mainFrame.winfo_children():
            widget.grid_forget()
        
        visible = 0
        for key in self.active.keys():
            if self.active[key].get():
                row = int(visible / self.maxcols)
                col = visible % self.maxcols
                self.widgets[key].grid(row=row, column=col, sticky='nesw')
                self.mainFrame.grid_rowconfigure(row, weight=1)
                self.mainFrame.grid_columnconfigure(col, weight=1)
                visible += 1


    def openPulsar(self, parfile, timfile):
        self.psr = pu.Pulsar(parfile, timfile)
        self.widgets['plk'].setPulsar(self.psr, updates=[self.widgets['par'].update])
        self.widgets['par'].setPulsar(self.psr, updates=[self.widgets['plk'].update])

    def openParTim(self):
        parfile = tkFileDialog.askopenfilename(title='Open par file')
        timfile = tkFileDialog.askopenfilename(title='Open tim file')
        self.openPulsar(parfile, timfile)
    
    def about(self):
        tkMessageBox.showinfo(title='About PINTkinter', 
                              message='A Tkinter based graphical interface to PINT')
        

def main(argv=None):
    parser = argparse.ArgumentParser(description='Tkinter interface for PINT pulsar timing tool')
    parser.add_argument('-p', '--parfile', help='parfile to use')
    parser.add_argument('-t', '--timfile', help='timfile to use')

    args = parser.parse_args(argv)

    root = tk.Tk()
    app = PINTkinter(root, parfile=args.parfile, timfile=args.timfile)
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.mainloop()

if __name__=='__main__':
    main()
