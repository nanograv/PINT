'''A wrapper around pulsar functions for pintkinter to use.

This object will be shared between widgets in the main frame
and will contain the pre/post fit model, toas,
pre/post fit residuals, and other useful information.
'''
from __future__ import print_function
from __future__ import division
import os, sys

# Numpy etc.
import numpy as np
import astropy.units as u
from astropy.time import Time
from enum import Enum
import copy

#Pint imports
import pint.models
import pint.toa
import pint.fitter
import pint.residuals
import pint.random_models


plot_labels = ['pre-fit', 'post-fit', 'mjd', 'year', 'orbital phase', 'serial', \
    'day of year', 'frequency', 'TOA error', 'rounded MJD']

# Some parameters we do not want to add a fitting checkbox for:
nofitboxpars = ['PSR', 'START', 'FINISH', 'POSEPOCH', 'PEPOCH', 'DMEPOCH', \
    'EPHVER', 'TZRMJD', 'TZRFRQ', 'TRES', 'PLANET_SHAPIRO']

class Fitters(Enum):
    POWELL = 0
    WLS = 1
    GLS = 2

class Pulsar(object):
    '''
    Wrapper class for a pulsar. Contains the toas, model, residuals, and fitter
    '''

    def __init__(self, parfile=None, timfile=None, ephem=None):
        super(Pulsar, self).__init__()

        print('STARTING LOADING OF PULSAR %s' % str(parfile))

        if parfile is not None and timfile is not None:
            self.parfile = parfile
            self.timfile = timfile
        else:
            raise ValueError("No valid pulsar to load")

        self.prefit_model = pint.models.get_model(self.parfile)
        
        if ephem is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=ephem, planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=ephem, planets=True)
            self.prefit_model.EPHEM.value = ephem
        elif getattr(self.prefit_model, 'EPHEM').value is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
        else:
            self.toas = pint.toa.get_TOAs(self.timfile,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile,planets=True)
        
        self.prefit_model.jump_flags_to_params(self.fulltoas)
        self.toas = copy.deepcopy(self.fulltoas)
        print("prefit_model.as_parfile():")
        print(self.prefit_model.as_parfile())
        
        self.toas.print_summary()
        
        self.prefit_resids = pint.residuals.Residuals(self.toas, self.prefit_model)
        print("RMS PINT residuals are %.3f us\n" % \
              self.prefit_resids.time_resids.std().to(u.us).value)
        self.fitter = Fitters.WLS
        self.fitted = False
        self.track_added = False
        
    @property
    def name(self):
        return getattr(self.prefit_model, 'PSR').value

    def __getitem__(self, key):
        try:
            return getattr(self.prefit_model, key)
        except AttributeError:
            print('Parameter %s was not found in pulsar model %s' % (key, self.name))
            return None

    def __contains__(self, key):
        return key in self.prefit_model.params

    def reset_model(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False
        self.update_resids()

    def reset_TOAs(self):
        print('in reset_toas')
        print('toas before',self.toas)
        if getattr(self.prefit_model, 'EPHEM').value is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
        else:
            self.toas = pint.toa.get_TOAs(self.timfile,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile,planets=True)
        print('toas after',self.toas)
        if self.track_added:
            self.prefit_model.TRACK.value = ''
            if self.fitted:
                self.postfit_model.TRACK.value = ''
            self.track_added = False
        self.update_resids()

    def resetAll(self):
        self.prefit_model = pint.models.get_model(self.parfile)
        self.postfit_model = None
        self.postfit_resids = None
        self.fitted = False

        if getattr(self.prefit_model, 'EPHEM').value is not None:
            self.toas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
        else:
            self.toas = pint.toa.get_TOAs(self.timfile,planets=True)
            self.fulltoas = pint.toa.get_TOAs(self.timfile, ephem=self.prefit_model.EPHEM.value,planets=True)
            
        self.update_resids()

    def update_resids(self):
        self.prefit_resids = pint.residuals.resids(self.fulltoas, self.prefit_model)
        if self.fitted:
            self.postfit_resids = pint.residuals.resids(self.fulltoas, self.postfit_model)

    def orbitalphase(self):
        '''
        For a binary pulsar, calculate the orbital phase. Otherwise, return
        an array of zeros
        '''
        if 'PB' not in self:
            print("WARNING: This is not a binary pulsar")
            return np.zeros(len(self.toas))

        toas = self.toas.get_mjds()

        if 'T0' in self and not self['T0'].quantity is None:
            tpb = (toas.value - self['T0'].value) / self['PB'].value
        elif 'TASC' in self and not self['TASC'].quantity is None:
            tpb = (toas.value - self['TASC'].value) / self['PB'].value
        else:
            print("ERROR: Neither T0 nor TASC set")
            return np.zeros(len(toas))

        phase = np.modf(tpb)[0]
        phase[phase < 0] += 1
        return phase * u.cycle

    def dayofyear(self):
        '''
        Return the day of the year for all the TOAs of this pulsar
        '''
        t = Time(self.toas.get_mjds(), format='mjd')
        year = Time(np.floor(t.decimalyear), format='decimalyear')
        return (t.mjd - year.mjd) * u.day

    def year(self):
        '''
        Return the decimal year for all the TOAs of this pulsar
        '''
        t = Time(self.toas.get_mjds(), format='mjd')
        return (t.decimalyear) * u.year

    def write_fit_summary(self):
        '''
        Summarize fitting results
        '''
        if self.fitted:
            chi2 = self.postfit_resids.chi2
            wrms = np.sqrt(chi2 / self.toas.ntoas)
            print('Post-Fit Chi2:\t\t%.8g us^2' % chi2)
            print('Post-Fit Weighted RMS:\t%.8g us' % wrms)
            print('%19s  %24s\t%24s\t%16s  %16s  %16s' %
                  ('Parameter', 'Pre-Fit', 'Post-Fit', 'Uncertainty', 'Difference', 'Diff/Unc'))
            print('-' * 132)
            fitparams = [p for p in self.prefit_model.params
                         if not getattr(self.prefit_model, p).frozen]
            for key in fitparams:
                line = '%8s ' % key
                pre = getattr(self.prefit_model, key)
                post = getattr(self.postfit_model, key)
                line += '%10s  ' % ('' if post.units is None else str(post.units))
                if post.quantity is not None:
                    line += '%24s\t' % pre.print_quantity(pre.quantity)
                    line += '%24s\t' % post.print_quantity(post.quantity)
                    try:
                        line += '%16.8g  ' % post.uncertainty.value
                    except:
                        line += '%18s' % ''
                    try:
                        diff = post.value - pre.value
                        line += '%16.8g  ' % diff
                        if pre.uncertainty is not None:
                            line += '%16.8g' % (diff / pre.uncertainty.value)
                    except:
                        pass
                print(line)
        else:
            print('Pulsar has not been fitted yet!')

    def add_phase_wrap(self, selected, phase):
        """
        Add a phase wrap to selected points in the TOAs object
        Turn on pulse number tracking in the model, if it isn't already
        """
        #Check if pulse numbers are in table already
        if 'pn' not in self.fulltoas.table.colnames or 'pn' not in self.toas.table.colnames:
            self.fulltoas.compute_pulse_numbers(self.prefit_model)
            self.toas.compute_pulse_numbers(self.prefit_model)
        if 'delta_pulse_numbers' not in self.fulltoas.table.colnames or 'delta_pulse_numbers' not in self.toas.table.colnames:
            self.fulltoas.table['delta_pulse_numbers'] = np.zeros(len(self.fulltoas.get_mjds()))
            self.toas.table['delta_pulse_numbers'] = np.zeros(len(self.toas.get_mjds()))
        
        self.fulltoas.table['delta_pulse_numbers'][selected] += phase
        self.toas.table['delta_pulse_numbers'] += phase
                        
        #Turn on pulse number tracking
        if self.prefit_model.TRACK.value != '-2':
            self.track_added = True
            self.prefit_model.TRACK.value = '-2'
            if self.fitted:
                self.postfit_model.TRACK.value = '-2'

        self.update_resids()
        
    def add_jump(self, selected):
        """
        jump the toas selected (add a new parameter to the model) or unjump them (remove the jump parameter) if already jumped
        """        
        print('flags before', self.fulltoas.table['flags'][selected])
        if "PhaseJump" not in self.prefit_model.components:
            print("PhaseJump component added")
            a = pint.models.jump.PhaseJump()
            a.setup()
            self.prefit_model.add_component(a)
            self.prefit_model.remove_param("JUMP1")
            param = pint.models.parameter.maskParameter(name = 'JUMP', index=1, key='jump', key_value = 1, value = 0.0, units = 'second')
            self.prefit_model.add_param_from_top(param, "PhaseJump")
            getattr(self.prefit_model, param.name).frozen = False
            if self.fitted:
                self.postfit_model.add_component(a)
            for dict1, dict2 in zip(self.fulltoas.table['flags'][selected], self.toas.table['flags']):
                dict1['jump'] = 1
                dict2['jump'] = 1
            print('phase_deriv_funcs',self.prefit_model.phase_deriv_funcs)
            return param.name
        #if gets here, has at least one jump param already
        #if doesnt overlap or cancel, add the param
        jump_nums = [int(dict['jump']) if 'jump' in dict.keys() else np.nan for dict in self.fulltoas.table['flags']]
        print(jump_nums)
        print(np.arange(1, np.nanmax(jump_nums)+1))
        for num in np.arange(1, np.nanmax(jump_nums)+1):
                num = int(num)
                jump_select = [num == jump_num for jump_num in jump_nums]
                if np.array_equal(jump_select, selected):
                    #if current jump exactly matches selected, remove it
                    self.prefit_model.remove_param('JUMP'+str(num))
                    print('flags before', self.fulltoas.table['flags'])
                    for dict1, dict2 in zip(self.fulltoas.table['flags'][selected], self.toas.table['flags']):
                        if 'jump' in dict1.keys() and dict1['jump'] == num:
                            del dict1['jump']#somehow deletes from both
                    print('flags after removal', self.fulltoas.table['flags'])
                    nums_subset = np.arange(num+1,np.nanmax(jump_nums)+1)
                    for n in nums_subset:
                        n = int(n)
                        for dict in self.fulltoas.table['flags']:
                            if 'jump' in dict.keys() and dict['jump'] == n:
                                print(n-1)
                                print('flags with old jumps', self.fulltoas.table['flags'])
                                dict['jump'] = n-1
                                print('flags with jumps replaced', self.fulltoas.table['flags'])
                        param = pint.models.parameter.maskParameter(name = 'JUMP', index=int(n-1), key='jump', 
                        key_value = int(n-1), value = getattr(self.prefit_model, 'JUMP'+str(n)).value, units = 'second')
                        self.prefit_model.add_param_from_top(param, 'PhaseJump')
                        getattr(self.prefit_model, param.name).frozen = getattr(self.prefit_model, 'JUMP'+str(n)).frozen
                        self.prefit_model.remove_param('JUMP'+str(n))
                    print(self.fulltoas.table['flags'])
                    print(nums_subset)
                    print(self.prefit_model.params)
                    if "JUMP1" not in self.prefit_model.params:
                        comp_list = getattr(self.prefit_model, 'PhaseComponent_list')
                        print(comp_list)
                        for item in comp_list:
                            if isinstance(item, pint.models.jump.PhaseJump):
                                comp_list.remove(item)
                                break
                        print(comp_list)
                        #This still doesn't work great because will reset all parameters if have done a few fits, need better way
                        self.prefit_model.setup_components(comp_list)
                        if self.fitted:
                            self.postfit_model.setup_components(comp_list)
                    else:
                        self.prefit_model.components["PhaseJump"].setup()
                    self.update_resids()
                    print("removed param", 'JUMP'+str(num))
                    print(self.prefit_model.params)
                    return jump_select
                elif True in [a and b for a,b in zip(jump_select, selected)]:
                    #if current jump overlaps selected, raise and error and end
                    print('Cannot jump toas that have already been jumped. Check for overlap.')
                    return None
        #if here, then doesn't overlap or match anything
        for dict1, dict2 in zip(self.fulltoas.table['flags'][selected], self.toas.table['flags']):
            dict1['jump'] = int(np.nanmax(jump_nums))+1
            dict2['jump'] = int(np.nanmax(jump_nums))+1
        param = pint.models.parameter.maskParameter(name = 'JUMP', index=int(np.nanmax(jump_nums))+1,\
            key='jump', key_value = int(np.nanmax(jump_nums))+1, value = 0.0, units = 'second', aliases=['JUMP'])
        self.prefit_model.add_param_from_top(param, "PhaseJump")
        getattr(self.prefit_model, param.name).frozen = False
        self.prefit_model.components["PhaseJump"].setup()
        if self.fitted:
            self.postfit_model.add_param_from_top(param, "PhaseJump")
            getattr(self.postfit_model, param.name).frozen = False
            self.postfit_model.components["PhaseJump"].setup()
        self.update_resids()
        j1t = self.prefit_model.JUMP1.toa_selector
        j2t = self.prefit_model.JUMP2.toa_selector
        print('phase_deriv_funcs',self.prefit_model.phase_deriv_funcs)
        return param.name
                
        #mjds = self.fulltoas.table['mjd_float'][selected]
        #minmjd = min(mjds)
        #maxmjd = max(mjds)
        #if "PhaseJump" not in self.prefit_model.components:
        #    print("PhaseJump component added")
        #    a = pint.models.jump.PhaseJump()
        #    a.setup()
        #    self.prefit_model.add_component(a)
        #    self.prefit_model.remove_param("JUMP1")
        #    param = pint.models.parameter.maskParameter(name = 'JUMP', index=1, key='mjd', key_value = [minmjd, maxmjd], frozen = False, value = 0.0, units = 'second')
        #    self.prefit_model.add_param_from_top(param, "PhaseJump")
        #    getattr(self.prefit_model, param.name).frozen = False
        #    if self.fitted:
        #        self.postfit_model.add_component(a)
        #    return param.name
        """
        ranges = []
        for param in self.prefit_model.params:
            if param.startswith("JUMP"):
                ranges.append(getattr(self.prefit_model,param).key_value+[getattr(self.prefit_model,param)])
        nums = []
        for r in ranges:
            nums.append(int(r[2].name[4:]))
            if minmjd == r[0] and maxmjd == r[1]:
                self.prefit_model.remove_param(r[2].name)
                ranges_subset = ranges[ranges.index(r):]
                c = True
                for rr in ranges_subset:
                    if c:#skip first loop
                        c = False
                        continue
                    param = pint.models.parameter.maskParameter(name = 'JUMP', index=int(rr[2].name[4:])-1, key='mjd', key_value = [rr[0], rr[1]], 
                    frozen = rr[2].frozen, value = rr[2].value, units = 'second')
                    self.prefit_model.add_param_from_top(param, 'PhaseJump')
                    getattr(self.prefit_model, param.name).frozen = rr[2].frozen
                    self.prefit_model.remove_param(rr[2].name)
                if "JUMP1" not in self.prefit_model.params:
                    comp_list = getattr(self.prefit_model, 'PhaseComponent_list')
                    print(comp_list)
                    for item in comp_list:
                        if isinstance(item, pint.models.jump.PhaseJump):
                            comp_list.remove(item)
                            break
                    print(comp_list)
                    self.prefit_model.setup_components(comp_list)
                    if self.fitted:
                        self.postfit_model.setup_components(comp_list)
                else:
                    self.prefit_model.components["PhaseJump"].setup()
                print("removed param", r[2].name)
                print(self.prefit_model.params)
                return [r[0],r[1]]#return the removed jump's range for updating jumped
            elif (r[0] <= minmjd and minmjd <= r[1]) or (r[0] <= maxmjd and maxmjd <= r[1]):
                print('jump range overlapping with',r[0], r[1], r[2].name)
                print('jump trying to implement', minmjd, maxmjd, max(nums)+1)
                print("Cannot JUMP toas that have already been jumped, check for overlap.")
                return None#end the function call
        #if doesn't overlap or cancel, add it to the model
        if nums == []:
            param = pint.models.parameter.maskParameter(name = 'JUMP', index=1, key='mjd', key_value = [minmjd, maxmjd], frozen = False, value = 0.0, units = 'second')
            self.prefit_model.add_param_from_top(param, "PhaseJump")
            getattr(self.prefit_model, param.name).frozen = False
            return param.name
        
        param = pint.models.parameter.maskParameter(name = 'JUMP', index=max(nums)+1, key='mjd', key_value = [minmjd, maxmjd], frozen = False, value = 0.0, units = 'second')
        self.prefit_model.add_param_from_top(param, "PhaseJump")
        getattr(self.prefit_model, param.name).frozen = False
        self.prefit_model.components["PhaseJump"].setup()
        return param.name
    """
    def fit(self, selected, iters=1):
        '''
        Run a fit using the specified fitter
        '''
        print(selected)
        print(self.toas.table['flags'])
        #print(self.toas.table['jump_section'])
        if not any(selected):
            print('none selected = all selected')
            selected = ~selected
        """JUMP check, put in fitter?"""
        """
        if 'PhaseJump' in self.prefit_model.components:
            #if attemped fit (selected) 
            #B) excludes a jump, turn that jump off
            #A) contains only jumps, don't do the fit and return an error
            #C) partially contains a jump, redifne that jump only with the overlap
            fit_jumps = []
            for param in self.prefit_model.params:
                if getattr(self.prefit_model, param).frozen == False and param.startswith('JUMP'):
                    fit_jumps.append(int(param[4:]))
            print(fit_jumps)
            jumps = [True if 'jump' in dict.keys() and dict['jump'] in fit_jumps else False for dict in self.toas.table['flags']]
            if all(jumps):
                print('toas being fit must not all be jumped. Remove or uncheck at least one jump in the selected toas before fitting.')
                return None
            sel_jump_nums = [dict['jump'] if 'jump' in dict.keys() else np.nan for dict in self.toas.table['flags']]
            full_jump_nums = [dict['jump'] if 'jump' in dict.keys() else np.nan for dict in self.fulltoas.table['flags']]
            for num in np.arange(1, np.nanmax(full_jump_nums)+1):
                num = int(num)
                if num not in sel_jump_nums:
                    getattr(self.prefit_model, 'JUMP'+str(num)).frozen = True
                    continue
                jump_select = [num == jump_num for jump_num in full_jump_nums]
                overlap = [a and b for a,b in zip(jump_select, selected)]
                #remove the jump flags for that num
                for dict in self.fulltoas.table['flags']:
                    if 'jump' in dict.keys() and dict['jump'] == num:
                        del dict['jump']
                #re-add the jump using overlap as 'selected'
                for dict in self.fulltoas.table['flags'][overlap]:
                    dict['jump'] = num
        """
#            mjds = self.toas.table['mjd_float']
#            mjds_copy = list(copy.deepcopy(mjds))
#            minmjd = min(mjds)
#            maxmjd = max(mjds)
#            
#            prefit_save = copy.deepcopy(self.prefit_model)
#            for param in self.prefit_model.params:
#                if param.startswith("JUMP") and getattr(self.prefit_model, param).frozen == False:
#                    minmax = getattr(self.prefit_model,param).key_value
#                    #checks if selected toas are all jumped and returns error if they all are
#                    if minmax[0] in mjds_copy:
                        #if min jump range in selected
#                        if minmax[1] in mjds_copy:
#                            #if max jump range in selected
#                            mjds_copy[mjds_copy.index(minmax[0]):mjds_copy.index(minmax[1])+1] = []
#                        else:
#                            #if min jump range in selected but not max jump range
#                            setattr(getattr(self.prefit_model, param), 'key_value', [minmax[0],maxmjd])
#                            mjds_copy[mjds_copy.index(minmax[0]):] = []
#                    elif minmax[1] in mjds_copy:
#                        #if max jump range in selected but not min jump range
#                        setattr(getattr(self.prefit_model, param), 'key_value', [minmjd, minmax[1]])
#                        mjds_copy[:mjds_copy.index(minmax[1])+1] = []
#                    elif minmax[0] < minmjd and minmax[1] > maxmjd:
#                        #if selected entirely within the jump range
#                        #dont bother resizing the jump range because its going to reset anyways
#                        mjds_copy = []
#                    else:
#                        #if being fit for but jump entirely outside range, uncheck it
##                        print("outside range")
#                        setattr(getattr(self.prefit_model, param), 'frozen', True)
#                    if mjds_copy == []:
#                        self.prefit_model = prefit_save
#                        print("toas being fit must not all be jumped. Remove or uncheck at least one jump in the selected toas before fitting.")
#                        return None
        print('toa flags right before fit', self.toas.table['flags'])
        print('prefit_model right before fit', self.prefit_model.as_parfile())
        if self.fitted:
            self.prefit_model = self.postfit_model
            self.prefit_resids = self.postfit_resids
            
        if 'delta_pulse_numbers' in self.toas.table.keys():
            print(self.toas.table['delta_pulse_numbers'])
        else:
            print('no dpn for toas yet')
        if self.fitter == Fitters.POWELL:
            fitter = pint.fitter.PowellFitter(self.toas, self.prefit_model)
        elif self.fitter == Fitters.WLS:
            fitter = pint.fitter.WlsFitter(self.toas, self.prefit_model)
        elif self.fitter == Fitters.GLS:
            fitter = pint.fitter.GLSFitter(self.toas, self.prefit_model)
        chi2 = self.prefit_resids.chi2
        wrms = np.sqrt(chi2 / self.toas.ntoas)
        print('Pre-Fit Chi2:\t\t%.8g us^2' % chi2)
        print('Pre-Fit Weighted RMS:\t%.8g us' % wrms)

        fitter.fit_toas(maxiter=1)
        self.postfit_model = fitter.model
        self.postfit_resids = pint.residuals.Residuals(self.fulltoas, self.postfit_model, set_pulse_nums = True)
        self.fitted = True
        self.write_fit_summary()
        

        q = list(self.fulltoas.get_mjds())
        index = q.index([i for i in self.fulltoas.get_mjds() if i > self.toas.get_mjds().min()][0])
        rs_mean = pint.residuals.resids(self.fulltoas, fitter.model, set_pulse_nums=True).phase_resids[index:index+len(self.toas.get_mjds())].mean()
        if len(fitter.get_fitparams()) < 3:
            redge = ledge = 3
            npoints = 400
        else:
            redge = ledge = 3
            npoints = 200
        f_toas, rs = pint.random_models.random(fitter, rs_mean=rs_mean, redge_multiplier=redge, ledge_multiplier=ledge, iter=10, npoints=npoints)
        self.random_resids = rs
        self.fake_toas = f_toas
            
