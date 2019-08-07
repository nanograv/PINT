# # Licensed under the  BSD 3-clause license - see LICENSE
#
#
# """ timing_setup.py defines a set of functions for setting up PINT for timing.
# """
#
# from .toa import get_TOAs
# from .models import get_model
#
#
#
# def check_components(model, required_components=['astrometry', 'spindown']):
#     """Check if a timing model has sufficent components for high precision
#     pulsar timing.
#
#        Parameter
#        ---------
#        model: `pint.TimingModel` object
#            The timing model need to exam.
#     """
#     for req_comp in required_components:
#         if req_comp not in model.categories:
#             raise ValueError('A timing model requires a component in the'
#                              ' category of {}'.format(req_comp))
#
#     # Set up the timing model again
#     model.setup()
#     return
#
#
# def init_params():
#     pass
