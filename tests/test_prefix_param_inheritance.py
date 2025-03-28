import io

from pint.models import get_model

input_par = """PSRJ                           J0523-7125
EPHEM                               DE405
CLK                               TT(TAI)
UNITS                                 TDB
START              55415.8045121523831364
FINISH             59695.2673406681377430
TIMEEPH                              FB90
T2CMETHOD                        IAU2000B
DILATEFREQ                              N
DMDATA                                  N
NTOA                                   87
CHI2                   404.35757416343705
RAJ                      5:23:48.66000000
DECJ                   -71:25:52.58000000
PMRA                                  0.0
PMDEC                                 0.0
PX                                    0.0
POSEPOCH           59369.0000000000000000
F0                  3.1001291305547288772 1 2.7544353718657238425e-11
F1              -2.4892219423278130317e-15 1 2.8277388169449218064e-19
PEPOCH             59609.0000000000000000
"""


def test_prefixparaminheritance_stayfrozen():
    # start with the case that has frozen parameters.  make sure they remain so
    m = get_model(io.StringIO(input_par + "\nF2 0\nF3 0"))
    assert m.F2.frozen
    assert m.F3.frozen


def test_prefixparaminheritance_unfrozen():
    # start with the case that has the new parameters unfrozen
    m = get_model(io.StringIO(input_par + "\nF2 0 1\nF3 0 1"))
    assert not m.F2.frozen
    assert not m.F3.frozen
