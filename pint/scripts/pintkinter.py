#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
'''
PINTkinter: Tkinter interactive interface for PINT pulsar timing tool
'''

from __future__ import absolute_import, print_function, division
import os, sys
import numpy as np
import Tkinter as tk
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

        self.initUI()

        self.createWidgets()
        self.requestOpenPlk(parfile=parfile, timfile=timfile)
        
        self.active = {'plk': True, 'par': True}
        self.updateLayout()

    def initUI(self):
        self.mainFrame = tk.Frame(master=self.master)
        self.mainFrame.grid(row=0, column=0, sticky='nesw')
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

    def createWidgets(self):
        self.plkWidget = PlkWidget(master=self.mainFrame)
        self.parWidget = ParWidget(master=self.mainFrame)

    def updateLayout(self):
        for widget in self.mainFrame.winfo_children():
            widget.grid_forget()
        ii = 0
        if self.active['plk']:
            self.plkWidget.grid(row=0, column=ii, sticky='nesw')
            ii += 1
        if self.active['par']:
            self.parWidget.grid(row=0, column=ii, sticky='nesw')
            ii += 1
        for i in range(ii):
            self.mainFrame.grid_columnconfigure(ii, weight=1)
        self.mainFrame.grid_rowconfigure(0, weight=1)

    def requestOpenPlk(self, parfile, timfile):
        self.openPlkPulsar(parfile, timfile)

    def openPlkPulsar(self, parfile, timfile):
        psr = pu.Pulsar(parfile, timfile)
        self.plkWidget.setPulsar(psr, update=self.parWidget.update)
        self.parWidget.setPulsar(psr, update=self.plkWidget.update)

def main(argv=None):
    parser = argparse.ArgumentParser(description='Tkinter interface for PINT pulsar timing tool')
    parser.add_argument('parfile', help='parfile to use')
    parser.add_argument('timfile', help='timfile to use')

    args = parser.parse_args(argv)

    root = tk.Tk()
    app = PINTkinter(root, parfile=args.parfile, timfile=args.timfile)
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.mainloop()

if __name__=='__main__':
    main()
