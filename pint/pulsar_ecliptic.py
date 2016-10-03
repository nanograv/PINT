from __future__ import print_function
import numpy as np
from astropy.coordinates import frame_transform_graph, DynamicMatrixTransform
from astropy.coordinates.angles import rotation_matrix
import astropy.coordinates as coord
import astropy.units as u
from .config import datapath
import os

# Load obliquity data
# Assume the data file is in the ./datafile directory
def load_obliquity_file(filename):
    obliquity_data = {}
    for l in open(filename).readlines():
        l = l.strip()
        if l.startswith('Obliquity of the ecliptic'):
            continue
        if l.startswith('#'):
            continue
        if l == '':
            continue
        line = l.split()
        obliquity_data[line[0]] = float(line[1])* u.arcsecond
    return obliquity_data

OBL = load_obliquity_file(datapath('ecliptic.dat'))


class PulsarEcliptic(coord.BaseCoordinateFrame):
    """A Pulsar Ecliptic coordinate system is defined by rotating ICRS coordinate
    about x-axis by obliquity angle. Historical, This coordinate is used by
    tempo/tempo2 for a better fitting error treatment.
    The obliquity angle values respect to time are given in the file named "ecliptic.dat"
    in the pint/datafile directory.
    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : `Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    """
    default_representation = coord.SphericalRepresentation
    obliquity = OBL['DEFAULT']
    # def __init__(self, obliquity=OBL['DEFAULT'], *argu, **kwargs)
    #     super(PulsarEcliptic, self).__init__(*argu, **kwargs):
    #     self.obliquity = obliquity


def _ecliptic_rotation_matrix_pulsar(obl):
    """Here we only do the obliquity angle rotation. Astropy will add the
    precession-nutation correction.
    """
    return rotation_matrix(obl, 'x')

@frame_transform_graph.transform(coord.DynamicMatrixTransform, coord.ICRS, PulsarEcliptic)
def icrs_to_pulsarecliptic(from_coo, to_frame):
    return _ecliptic_rotation_matrix_pulsar(to_frame.obliquity)


@frame_transform_graph.transform(coord.DynamicMatrixTransform, PulsarEcliptic, coord.ICRS)
def pulsarecliptic_to_icrs(from_coo, to_frame):
    return icrs_to_pulsarecliptic(to_frame, from_coo).T
