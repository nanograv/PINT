# Licensed under the  BSD 3-clause license - see LICENSE


""" f_tests.py defines a set of functions for examing the timing model
    parameters using F test.
"""


import numpy as np
from scipy.special import fdtr


def Ftest(chi2_1, dof_1, chi2_2, dof_2):
    """Compute an F-test to see if a model with extra parameters is significant
       compared to a simpler model.

       Parameters
       ----------
       chi2_1: float
           Original model's non-reduced chi^2.
       dof_1: float
           Original model's degree of freedom.
       chi2_2: float
           New model's non-reduced chi^2.
       dof_2: float
           New model's degree of freedom.

       Return
       ------
       The probability that the improvement in chi2 is due to random chance
       (i.e. a low probability means that the new fit is quantitatively better,
       while a value near 1 means that the new model should likely be rejected).
       If the new model has a higher chi^2 than the original model, returns
       value is -1.

       Note
       ----
       The probability is computed exactly like Sherpa's F-test routine
       (in Ciao) and is also described in the Wikipedia article on the
       F-test:  http://en.wikipedia.org/wiki/F-test.
    """
    delta_chi2 = chi2_1 - chi2_2
    if delta_chi2 > 0:
      delta_dof = dof_1 - dof_2
      new_redchi2 = chi2_2 / dof_2
      F = float((delta_chi2 / delta_dof) / new_redchi2)
      ft = 1.0 - fdtr(delta_dof, dof_2, F)
    else:
      ft = -1
    return ft
