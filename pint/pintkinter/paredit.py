import Tkinter as tk
import tkFileDialog
import tempfile
import copy
import os

import pint.models

class ParChoiceWidget(tk.Frame):
    '''
    Lets the user select between the pre-fit and post-fit model for the
    loaded pulsar
    '''
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        
        self.choose_callback = None
        self.initLayout()

    def initLayout(self):
        self.choice = tk.StringVar()
        self.prefit = tk.Radiobutton(self, text='Pre-Fit', command=self.choose,
                                     variable=self.choice, value='prefit')
        self.prefit.select()
        self.prefit.grid(row=0, column=0)

        self.postfit = tk.Radiobutton(self, text='Post-Fit', command=self.choose,
                                      variable=self.choice, value='postfit')
        self.postfit.grid(row=0, column=1)

    def setCallbacks(self, choose):
        self.choose_callback = choose

    def choose(self):
        self.choose_callback()

class ParActionsWidget(tk.Frame):
    '''
    Allows the user to reset the model, apply changes, or save to a parfile
    '''
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.reset_callback = None
        self.apply_callback = None
        self.write_callback = None

        self.initLayout()

    def initLayout(self):
        button = tk.Button(self, text='Reset Changes', command=self.resetChanges)
        button.grid(row=0, column=0)

        button = tk.Button(self, text='Apply Changes', command=self.applyChanges)
        button.grid(row=0, column=1)

        button = tk.Button(self, text='Write Par', command=self.writePar)
        button.grid(row=0, column=2)
    
    def setCallbacks(self, resetChanges, applyChanges, writePar):
        self.reset_callback = resetChanges
        self.apply_callback = applyChanges
        self.write_callback = writePar

    def resetChanges(self):
        if self.reset_callback is not None:
            self.reset_callback()
        print('Reset clicked')

    def applyChanges(self):
        if self.apply_callback is not None:
            self.apply_callback()
        print('Apply clicked')

    def writePar(self):
        if self.write_callback is not None:
            self.write_callback()
        print('Write clicked')

class ParWidget(tk.Frame):
    '''
    A widget that allows editing and saving of a pulsar parfile
    '''
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.psr = None
        self.update_callback = None
        self.initLayout()
    
    def initLayout(self):
        self.choiceWidget = ParChoiceWidget(master=self)
        self.choiceWidget.grid(row=0, column=0, sticky='nw')

        self.editor = tk.Text(self)
        self.editor.grid(row=1, column=0, sticky='nesw')

        scrollbar = tk.Scrollbar(self, command=self.editor.yview)
        scrollbar.grid(row=1, column=1, sticky='nesw')
        self.editor['yscrollcommand'] = scrollbar.set

        self.actionsWidget = ParActionsWidget(master=self)
        self.actionsWidget.grid(row=2, column=0, sticky='w')

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=10)
        self.grid_rowconfigure(2, weight=1)

    def update(self):
        self.set_model()

    def setPulsar(self, psr, update):
        self.psr = psr

        self.choiceWidget.setCallbacks(self.set_model)
        self.actionsWidget.setCallbacks(self.set_model,
                                        self.applyChanges,
                                        self.writePar)
        self.set_model()
        self.update_callback = update

    def set_model(self):
        choice = self.choiceWidget.choice.get()
        self.editor.delete('1.0', tk.END)
        if choice == 'postfit':
            if self.psr.fitted:
                self.editor.insert('1.0', self.psr.postfit_model.as_parfile())
            else:   
                print('There is no postfit model yet!')
                self.editor.insert('1.0', self.psr.prefit_model.as_parfile())
        elif choice == 'prefit':
            self.editor.insert('1.0', self.psr.prefit_model.as_parfile())

    def updateModel(self, model):
        pfilename = tempfile.mkstemp()[1]
        pfile = open(pfilename, 'w')
        pfile.write(self.editor.get('1.0', 'end-1c'))
        pfile.close()
        model = pint.models.get_model(pfilename)
        os.remove(pfilename)
        
    def applyChanges(self):
        self.updateModel(self.psr.prefit_model)
        if self.update_callback is not None:
            self.update_callback()

    def writePar(self):
        filename = tkFileDialog.asksaveasfilename(title='Choose output par file')
        try:
            fout = open(filename, 'w')
            fout.write(self.editor.get('1.0', 'end-1c'))
            fout.close()
            print('Saved parfile to %s' % filename)
        except:
            print('Could not save parfile to filename:\t%s' % filename)
