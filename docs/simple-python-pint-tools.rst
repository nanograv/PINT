
How to: Simple python PINT tools
================================

(This was originally posted by Alex McEwen to the PINT wiki.)

Below are several tools I use for new pulsar timing, including cleaning TOAs, adding wraps, and fitting parameters. Please send suggestions/comments/questions to aemcewen@uwm.edu.

Load various packages as well as some little convenience functions::

    import numpy as np
    import matplotlib.pyplot as plt
    import pint.residuals as res
    import copy
    from pint.models import BinaryELL1, BinaryDD, PhaseJump, parameter, get_model
    from pint.simulation import make_fake_toas_uniform as mft
    from astropy import units as u, constants as c
    def dot(l1,l2):
        return np.array([v1 and v2 for v1,v2 in zip(l1,l2)])
    def inv(l):
        return np.array([not i for i in l])

Zapping TOAs on given MJDs, before/after some MJD, or within a window of days::

    def mask_toas(toas,before=None,after=None,on=None,window=None):
        cnd=np.array([True for t in toas.get_mjds()])
        if before is not None:
            cnd = dot(cnd,toas.get_mjds().value > before)
        if after is not None:
            cnd = dot(cnd,toas.get_mjds().value < after)
        if on is not None:
            on=np.array(on)
            for i,m in enumerate(on):
                m=m*u.day
                if type(m) is int:
                    cnd = dot(cnd,inv(np.abs(toas.get_mjds()-m).astype(int) == np.abs((toas.get_mjds()-m)).min().astype(int)))
                else:
                    cnd = dot(cnd,inv(np.abs(toas.get_mjds()-m) == np.abs((toas.get_mjds()-m)).min()))
        if window is not None:
            if len(window)!=2:
                raise ValueError("window must be a 2 element list/array")
            window = window*u.day
            lower = window[0]
            upper = window[1]
            cnd = dot(cnd,toas.get_mjds() < lower)+dot(cnd,toas.get_mjds() > upper)
        print(f'{sum(cnd)}/{len(cnd)} TOAs selected')
        return toas[cnd]

Add in integer phase wraps on a given MJD::

    def add_wraps(toas,mjd,sign,nwrap=1):
        cnd = toas.table['mjd'] > Time(mjd,scale='utc',format='mjd')
        if sign == '-':
            toas.table['pulse_number'][cnd] -= nwrap
        elif sign == '+':
            toas.table['pulse_number'][cnd] += nwrap
        else:
            raise TypeError('sign must be "+" or "-"')

Plot residuals in phase::

    def plot_fit(toas,model,track_mode="use_pulse_numbers",title=None,xlim=None,ylim=None):
        rs=res.Residuals(toas,model,track_mode=track_mode)
        fig, ax = plt.subplots(figsize=(12,10))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.errorbar(rs.toas.get_mjds().value,rs.calc_phase_resids(), \
                     yerr=(rs.toas.get_errors()*model.F0.quantity).decompose().value,fmt='x')
        ax.tick_params(labelsize=15)

        if title is None:
            ax.set_title('%s Residuals, %s toas' %(model.PSR.value,len(toas.get_mjds())),fontsize=18)
        else:
            ax.set_title(title,fontsize=18)
        ax.set_xlabel('MJD',fontsize=15)
        ax.set_ylabel(f'Residuals [phase, P0 = {((1/model.F0.quantity).to(u.ms)).value:2.0f} ms]',fontsize=15)
        ax.grid()
        return fig, ax

Model phase uncertainty over a range of MJDs::

    def calculate_phase_uncertainties(model, MJDmin, MJDmax, Nmodels=100, params = 'all', error=1*u.us):
        mjds = np.arange(MJDmin,MJDmax)
        Nmjd = len(mjds)
        phases_i = np.zeros((Nmodels,Nmjd))
        phases_f = np.zeros((Nmodels, Nmjd))
        tnew = mft(MJDmin,MJDmax,Nmjd,model=model, error=error)
        pnew = {}
        if params == 'all':
            params = model.free_params
        for p in params:
            pnew[p] = getattr(model,p).quantity + np.random.normal(size=Nmodels) * getattr(model,p).uncertainty
        for imodel in range(Nmodels):
            m2 = copy.deepcopy(model)
            for p in params:
                getattr(m2,p).quantity=pnew[p][imodel]
            phase = m2.phase(tnew, abs_phase=True)
            phases_i[imodel] = phase.int
            phases_f[imodel] = phase.frac
        phases = phases_i+ phases_f
        phases0 = model.phase(tnew, abs_phase = True)
        dphase = phases - (phases0.int + phases0.frac)
        return tnew, dphase

Plot the phase uncertainty from calculate_phase_uncertainties()::

    def plot_phase_unc(model,start,end,params='all'):
        if params == 'all':
            print("calculating phase uncertainty due to all parameters")
            plab = 'All params'
            t, dp = calculate_phase_uncertainties(model, start, end)
        else:
            if type(params) is list:
                print("calculating phase uncertainty due to params "+str(params))
                plab = str(params)
                t, dp = calculate_phase_uncertainties(model, start, end, params = params)
            else:
                raise TypeError('"params" should be either list or "all"')

        plt.gcf().set_size_inches(12,10)
        plt.plot(t.get_mjds(),dp.std(axis=0),'.',label=plab)
        dt = t.get_mjds() - model.PEPOCH.value*u.d
        plt.plot(t.get_mjds(), np.sqrt((model.F0.uncertainty * dt)**2 + (0.5*model.F1.uncertainty*dt**2)**2).decompose(),label='Analytic')
        plt.xlabel('MJD')
        plt.ylabel('Phase Uncertainty (cycles)')
        plt.legend()

Less common tools
-----------------

Plot frequency against residuals::

    rs=res.Residuals(newtoas,f.model)
    fig,ax = plt.subplots(figsize=(12,10))
    ax.tick_params(labelsize=15)
    ax.set_ylabel('Frequency [MHz]',fontsize=18)
    ax.set_xlabel('Phase residuals',fontsize=18)

    y = newtoas.get_freqs().to('MHz').value
    x = rs.calc_phase_resids()

    ax.errorbar(x,y,xerr=newtoas.get_errors().to('s').value*f.model.F0.value,elinewidth=2,lw=0,marker='+')

Plot residuals in orbital phase::

    x = f.model.orbital_phase(newtoas.get_mjds()).value
    rs=res.Residuals(newtoas,f.model)
    y = rs.calc_phase_resids()

    fig, ax = plt.subplots(figsize=(12,10))
    ax.tick_params(labelsize=15)
    ax.set_xlabel('Orbital Phase',fontsize=18)
    ax.set_ylabel('Phase Residuals',fontsize=18)
    ax.grid()
    for mjd in np.unique(newtoas.get_mjds().astype(int)):
        cnd = dot(newtoas.get_mjds().astype(int) == mjd,newtoas.get_errors().astype(int) <= 125*u.us)
        ax.errorbar(x[cnd],y[cnd],yerr=(newtoas.get_errors().to('s')*f.model.F0.quantity).value[cnd],elinewidth=2,lw=0,marker='+',label=mjd.value)


    ax.legend(fontsize=15)

Removing/adding binary components::

    if 'BinaryELL1' in model.components:
        model.remove_component('BinaryELL1')

    cmp = BinaryELL1()
    cmp.PB.value = 10
    cmp.EPS1.value = 1e-5
    cmp.EPS2.value = 1e-5
    cmp.TASC.value = 59200
    cmp.A1.value = 10

    model.add_component(cmp,setup=True)


    cmp = BinaryDD()
    cmp.PB.value = 10
    cmp.ECC.value = 0.8
    cmp.T0.value = 59251.
    cmp.OM.value = 269.
    cmp.A1.value = 136.9

    model.add_component(cmp,setup=True)

Add spin-down component of a given order::

    n = 2
    model.components['Spindown'].add_param(
        parameter.prefixParameter(
            name='F'+str(n),
            value=0.0,
            units=u.Hz/u.s**n,
            frozen=False,
            parameter_type="float",
            longdouble=True
        ),
        setup=True
    )
    model.components['Spindown'].validate()
