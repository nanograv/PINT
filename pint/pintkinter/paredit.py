import Tkinter as tk
import tkFileDialog
import copy

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

class ParTableWidget(tk.Frame):
    '''
    Contains the actual grid of parfile values
    '''
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.initLayout()
    
    def initLayout(self):
        label = tk.Label(self, text='Selected')
        label.grid(row=0, column=0)

        label = tk.Label(self, text='Parameter')
        label.grid(row=0, column=1)
        
        label = tk.Label(self, text='Value')
        label.grid(row=0, column=2)
        
        label = tk.Label(self, text='Fit?')
        label.grid(row=0, column=3)
        
        label = tk.Label(self, text='Uncertainty')
        label.grid(row=0, column=4)

    def loadTable(self, model):
        '''
        Load the model parfile values into the arrays stored here
        '''
        print('Loading model')
        self.selects = []
        self.params = []
        self.values = []
        self.fits = []
        self.uncs = []
        
        for ii, par in enumerate(model.params):
            self.selects.append(tk.IntVar())

            self.params.append(tk.StringVar())
            self.params[ii].set(par)

            self.values.append(tk.StringVar())
            val = getattr(model, par).value
            self.values[ii].set(str(val) if not val is None else '')

            self.fits.append(tk.IntVar())
            self.fits[ii].set(0 if getattr(model, par).frozen else 1)

            self.uncs.append(tk.StringVar())
            unc = getattr(model, par).uncertainty_value
            self.uncs[ii].set(str(unc) if not unc is None else '')

        self.renderTable()

    def renderTable(self):
        '''
        Add the grid of editable boxes for the parfile
        '''
        #Delete old boxes
        for widget in self.winfo_children():
            if not isinstance(widget, tk.Label):
                widget.destroy()
        
        for ii in range(len(self.params)):
            cb = tk.Checkbutton(self, variable=self.selects[ii])
            cb.grid(row=ii+1, column=0)

            entry = tk.Entry(self, textvariable=self.params[ii])
            entry.grid(row=ii+1, column=1)
            
            entry = tk.Entry(self, textvariable=self.values[ii])
            entry.grid(row=ii+1, column=2)
            
            cb = tk.Checkbutton(self, variable=self.fits[ii])
            cb.grid(row=ii+1, column=3)

            entry = tk.Entry(self, textvariable=self.uncs[ii])
            entry.grid(row=ii+1, column=4)

    def addRow(self):
        indices = []
        for ii, sel in enumerate(self.selects):
            if sel.get():
                indices.append(ii)
        if len(indices) > 0:
            for ii in reversed(indices):
                self.selects[ii].set(0)
                self.selects.insert(ii, tk.IntVar())
                self.params.insert(ii, tk.StringVar())
                self.values.insert(ii, tk.StringVar())
                self.fits.insert(ii, tk.IntVar())
                self.uncs.insert(ii, tk.StringVar())
        else:
            self.selects.append(tk.IntVar())
            self.params.append(tk.StringVar())
            self.values.append(tk.StringVar())
            self.fits.append(tk.IntVar())
            self.uncs.append(tk.StringVar())

        print('Add Row clicked')
        self.renderTable()

    def deleteRow(self):
        for ii, sel in reversed(list(enumerate(self.selects))):
            if sel.get():
                self.params.pop(ii)
                self.values.pop(ii)
                self.fits.pop(ii)
                self.uncs.pop(ii)
                self.selects.pop(ii)

        print('Delete Row clicked')
        self.renderTable()

class ParEditWidget(tk.Frame):
    '''
    Lets the user edit selected values for the pulsar
    '''
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        self.initLayout()

    def initLayout(self):
        self.table = ParTableWidget(master=self)
        self.table.grid(row=0, column=0, columnspan=2, sticky='nesw')

        button = tk.Button(self, text='Add Row', command=self.table.addRow)
        button.grid(row=1, column=0)

        button = tk.Button(self, text='Delete Row', command=self.table.deleteRow)
        button.grid(row=1, column=1)

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

        self.editWidget = ParEditWidget(master=self)
        self.editWidget.grid(row=1, column=0, sticky='nesw')

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
        self.editWidget.table.loadTable(self.psr.prefit_model)
        self.actionsWidget.setCallbacks(self.set_model,
                                        self.applyChanges,
                                        self.writePar)
        self.update_callback = update

    def set_model(self):
        choice = self.choiceWidget.choice.get()
        if choice == 'postfit':
            if self.psr.fitted:
                self.editWidget.table.loadTable(self.psr.postfit_model)
            else:   
                print('There is no postfit model yet!')
                self.choiceWidget.prefit.select()
        elif choice == 'prefit':
            self.editWidget.table.loadTable(self.psr.prefit_model)

    def updateModel(self, model):
        for p, v, f, u in zip(self.editWidget.table.params,
                              self.editWidget.table.values,
                              self.editWidget.table.fits,
                              self.editWidget.table.uncs):
            par = p.get()
            #Edit existing row
            if hasattr(model, par):
                val = v.get()
                if not val == '':
                    try:
                        val = float(val)
                    except:
                        if val in ['True', 'y', 'Y']:
                            val = True
                        elif val in ['False', 'n', 'N']:
                            val = False
                        else:
                            print('%s cannot resolve value %s' % (par, val))
                            continue
                    getattr(model, par).value = val
                getattr(model, par).frozen = not bool(f.get())
                unc = u.get()
                if not unc == '':
                    try:
                        unc = float(unc)
                        getattr(model, par).uncertainty_value = unc
                    except:
                        print('%s uncertainty value %s could not be set' % (par, unc))

    def applyChanges(self):
        self.updateModel(self.psr.prefit_model)
        if self.update_callback is not None:
            self.update_callback()

    def writePar(self):
        model = copy.deepcopy(self.psr.prefit_model)
        self.updateModel(model)
        #Save as filename
        filename = tkFileDialog.asksaveasfilename(title='Choose output par file')
        try:
            fout = open(filename, 'w')
            fout.write(model.as_parfile())
            fout.close()
            print('Saved parfile to %s' % filename)
        except:
            print('Could not save parfile to filename:\t%s' % filename)
