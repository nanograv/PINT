
How to: Common timing workflow
==============================

(This was originally written by Alex McEwen on the PINT wiki.)

When I'm working on new pulsar solutions, my work flow usually follows
something like this. This uses some of the basic tools that are included in the
:ref:`user questions <user-questions>`. Please send
suggestions/comments/questions to aemcewen@uwm.edu.


1. load pulsar model/TOAs via get_TOAs() and get_model()::

     model = get_model('parfile.par')
     toas = get_TOAs('toas.tim', model=model)

2. make a copy of the TOAs that I will edit (for easy resets)::

     newtoas = toas

3. use plot_fit(newtoas), identify bad toas/missing wraps and mask those data. in this example, i am zapping all TOAs before MJD 59000 and also on 59054.321. also, i'm adding a wrap on 59166::

     newtoas = mask_toas(newtoas,before=59000,on=[59054.321])
     newtoas.compute_pulse_numbers(model)
     add_wraps(newtoas,59166,'-')
     plot_fit(newtoas,model)

4. once i have the TOAs i want, i fit the data, look at the residuals, and see how the model changed::

     f=WLSFitter(newtoas,model,track_mode='use_pulse_numbers')
     f.model.free_params=['F0']
     f.fit_toas()
     plot_fit(newtoas,f.model)
     f.print_summary()
     f.model.compare(model,verbosity='check')

5. when the model appears to be a good fit, i update the model and add new observations::

     model=f.model
     newtoas=mask_toas(toas,before=58000,on=59054.321)
     plot_fit(newtoas,model)

6. iterate these last two steps until the fit breaks/i run out of data.
