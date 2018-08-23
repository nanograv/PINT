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

        self.createPlkWidget()
        self.createParEditWidget()

        self.requestOpenPlk(parfile=parfile, timfile=timfile)
        
        self.active = {'plk': True}
        self.updateLayout()

    def initUI(self):
        self.mainFrame = tk.Frame(master=self.master)
        self.mainFrame.grid(row=0, column=0)

    def createPlkWidget(self):
        self.plkWidget = PlkWidget(master=self.mainFrame)

    def createParEditWidget(self):
        pass

    def updateLayout(self):
        ii = 0
        for widget in self.mainFrame.winfo_children():
            widget.grid_forget()
        if self.active['plk']:
            self.plkWidget.grid(row=0, column=ii)
            ii += 1

    def requestOpenPlk(self, parfile, timfile):
        self.openPlkPulsar(parfile, timfile)

    def openPlkPulsar(self, parfile, timfile):
        psr = pu.Pulsar(parfile, timfile)
        self.plkWidget.setPulsar(psr)

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
