from __future__ import absolute_import, division, print_function

import sys

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import QuantityAttribute, frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix

from pint.config import datapath
from pint.utils import interesting_lines, lines_of

__all__ = ["OBL", "PulsarEcliptic"]

astropy_version = sys.modules["astropy"].__version__

# Load obliquity data
# Assume the data file is in the ./datafile directory
def load_obliquity_file(filename):
    obliquity_data = {}
    for l in interesting_lines(lines_of(filename), comments="#"):
        if l.startswith("Obliquity of the ecliptic"):
            continue
        line = l.split()
        obliquity_data[line[0]] = float(line[1]) * u.arcsecond
    return obliquity_data


OBL = load_obliquity_file(datapath("ecliptic.dat"))


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
    # NOTE: The feature below needs astropy verison 2.0. Disable it right now
    default_differential = coord.SphericalCosLatDifferential
    obliquity = QuantityAttribute(default=OBL["DEFAULT"], unit=u.arcsec)

    def __init__(self, *args, **kwargs):
        # Allow using 'pm_lat' and 'pm_lon_coslat' keywords under astropy 2.
        # This matches the behavior of the built-in frames in astropy 2,
        # and the behavior of custom frames in astropy 3+.
        if int(astropy_version.split(".")[0]) <= 2:
            try:
                kwargs["d_lon_coslat"] = kwargs["pm_lon_coslat"]
                del kwargs["pm_lon_coslat"]
            except KeyError:
                pass
            try:
                kwargs["d_lat"] = kwargs["pm_lat"]
                del kwargs["pm_lat"]
            except KeyError:
                pass

        if "ecl" in kwargs:
            try:
                kwargs["obliquity"] = OBL[kwargs["ecl"]]
            except KeyError:
                raise ValueError(
                    "No obliquity " + kwargs["ecl"] + " provided. "
                    "Check your pint/datafile/ecliptic.dat file."
                )
            del kwargs["ecl"]

        super(PulsarEcliptic, self).__init__(*args, **kwargs)


def _ecliptic_rotation_matrix_pulsar(obl):
    """Here we only do the obliquity angle rotation. Astropy will add the
    precession-nutation correction.
    """
    return rotation_matrix(obl, "x")


@frame_transform_graph.transform(
    coord.DynamicMatrixTransform, coord.ICRS, PulsarEcliptic
)
def icrs_to_pulsarecliptic(from_coo, to_frame):
    return _ecliptic_rotation_matrix_pulsar(to_frame.obliquity)


@frame_transform_graph.transform(
    coord.DynamicMatrixTransform, PulsarEcliptic, coord.ICRS
)
def pulsarecliptic_to_icrs(from_coo, to_frame):
    return icrs_to_pulsarecliptic(to_frame, from_coo).T
