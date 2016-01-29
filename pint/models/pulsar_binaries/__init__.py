# __init__.py for PINT models/binarys directory
"""This module contains implementations of pulsar timing independent binary models.
"""
# setup environment
# Load the PINT environment variable to get the top level directory
pintdir = os.getenv("PINT")
if pintdir is None:
    filedir = os.path.split(os.path.realpath(__file__))[0]
    pintdir = os.path.abspath(os.path.join(filedir, ".."))
    del filedir
