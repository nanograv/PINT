import contextlib
import io
from astropy import units as u
import pint.logging
from pint.models import get_model
import pint.fitter
import pint.simulation

pint.logging.setup("ERROR")


def test_sini_fit():
    s = """PSRJ                           J1853+1303
    EPHEM                               DE440
    CLK                               TT(TAI)
    UNITS                                 TDB
    START              55731.1937186050359860
    FINISH             59051.1334073910116910
    TIMEEPH                              FB90
    T2CMETHOD                        IAU2000B
    DILATEFREQ                              N
    DMDATA                                  N
    NTOA                                 4570
    CHI2R                              0.9873 0 4455.0
    TRES                                1.297
    ELONG                 286.257303932061347 1 0.00000000931838521883
    ELAT                   35.743349195626379 1 0.00000001342269042838
    PMELONG               -1.9106838589844408 1 0.011774688289719207
    PMELAT                -2.7055634767966588 1 0.02499026250312041
    PX                     0.6166118941596491 1 0.13495439191146907
    ECL                              IERS2010
    POSEPOCH           57391.0000000000000000
    F0                  244.39137771284777388 1 4.741681125242900587e-13
    F1              -5.2068158193285555695e-16 1 1.657252200776189767e-20
    PEPOCH             57391.0000000000000000
    CORRECT_TROPOSPHERE                         Y
    PLANET_SHAPIRO                          Y
    NE_SW                                 0.0
    SWM                                   0.0
    DM                  30.570200097819536894
    DM1                                   0.0
    DMEPOCH            57391.0000000000000000
    BINARY                                 DD
    PB                  115.65378643873621578 1 3.2368969907e-09
    PBDOT                                 0.0
    A1                     40.769522249658145 1 1.53854961349644e-06
    XDOT                1.564794972862985e-14 1 8.617009015429654e-16
    ECC                2.3701427108198803e-05 1 3.467883031e-09
    EDOT                                  0.0
    T0                 57400.7367807004642730 1 0.00999271094974152688
    OM                  346.60040313716600618 1 0.03110468777178034688
    OMDOT                                 0.0
    M2                    0.18373369620702712 1 0.19097347492091507
    SINI                                  0.2 1 0.14457246899831822
    A0                                    0.0
    B0                                    0.0
    GAMMA                                 0.0
    DR                                    0.0
    DTH                                   0.0
    TZRMJD             57390.6587742196275694
    TZRSITE                           arecibo
    TZRFRQ                            426.875
    """
    m = get_model(io.StringIO(s))
    t = pint.simulation.make_fake_toas_uniform(
        55731, 59051, 100, m, obs="gbt", error=1 * u.us, add_noise=True
    )
    f = pint.fitter.Fitter.auto(t, m)
    f.fit_toas()
