# # Licensed under the  BSD 3-clause license - see LICENSE
#
#
# """ f_tests.py defines a set of functions for examing the timing model
#     parameters using F test.
# """
#
#
# import numpy as np
# from scipy.special import fdtr
# from copy import deepcopy
#
# from .toa import get_TOAs
# from .models import get_model
#
#
# def Ftest(chi2_1, dof_1, chi2_2, dof_2):
#     """Compute an F-test to see if a model with extra parameters is significant
#        compared to a simpler model.
#
#        Parameters
#        ----------
#        chi2_1: float
#            Original model's non-reduced chi^2.
#        dof_1: float
#            Original model's degree of freedom.
#        chi2_2: float
#            New model's non-reduced chi^2.
#        dof_2: float
#            New model's degree of freedom.
#
#        Return
#        ------
#        The probability that the improvement in chi2 is due to random chance
#        (i.e. a low probability means that the new fit is quantitatively better,
#        while a value near 1 means that the new model should likely be rejected).
#        If the new model has a higher chi^2 than the original model, returns
#        value is -1.
#
#        Note
#        ----
#        The probability is computed exactly like Sherpa's F-test routine
#        (in Ciao) and is also described in the Wikipedia article on the
#        F-test:  http://en.wikipedia.org/wiki/F-test.
#     """
#     delta_chi2 = chi2_1 - chi2_2
#     if delta_chi2 > 0:
#       delta_dof = dof_1 - dof_2
#       new_redchi2 = chi2_2 / dof_2
#       F = (delta_chi2 / delta_dof) / new_redchi2
#       ft = 1.0 - fdtr(delta_dof, dof_2, F)
#     else:
#       ft = -1
#     return ft
#
#
# def test_params(input_model, toas, params, action, fitter='WlsFitter'):
#     component = est_model
#
#     if action == 'remove':
#         test_model
#
#
# p,parlines,remove=False, group=None, label=None,
#         allow_errors=False):
#     partmp = parfile + ".tmp"
#     outpar = params['PSR'] + ".par"
#     f = open(partmp,'w')
#     if remove:
#         newparlines = list(parlines) # makes a copy
#         for k in p:
#             newparlines = [l for l in newparlines if not l.startswith(k)]
#         f.writelines(newparlines)
#     else:
#         f.writelines(parlines)
#         for k in p:
#             val = 0.0
#             if k=='M2': val=0.25
#             if k=='SINI': val=0.8
#             l = "%s %f 1\n" % (k,val)
#             f.write(l)
#     f.close()
#     if opt.tempo2:
#         (rms,chi2,ndof) = run_tempo2(partmp,timfile,use_npulse=npulsefile)
#     else:
#         (rms,chi2,ndof) = run_tempo(partmp,timfile,use_npulse=npulsefile,
#                 allow_errors=allow_errors)
#     try:
#         os.unlink(partmp)
#         os.unlink(outpar)
#     except:
#         pass
#     try:
#         if remove:
#             ft = Ftest(chi2, ndof, base_chi2, base_ndof)
#             if group is None: group = 'existing'
#         else:
#             ft = Ftest(base_chi2, base_ndof, chi2, ndof)
#             if group is None: group = 'additional'
#     except ZeroDivisionError:
#         ft = 0.0
#     if not label:
#         label = p
#     report_ptest(label, rms, chi2, ndof, ft, group=group)
#     return (rms,chi2,ndof)
