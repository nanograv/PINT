import astropy.units as u
import astropy.time
import pint.simulation
from pint.models.model_builder import get_model, get_model_and_toas
from pint.toa import get_TOAs
import pint.residuals
import io
import numpy as np
import tempfile
import os
import pint.config
from pint.fitter import GLSFitter
from pinttestdata import datadir
import pytest


def roundtrip(toas, model):
    with tempfile.NamedTemporaryFile("wt") as f:
        toas.write_TOA_file(f)
        f.flush()
        toas2 = get_TOAs(f.name, model=model)
    return toas2


NGC6440E = """
PSR              1748-2021E
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
EPHEM               DE421
UNITS               TDB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
DILATEFREQ          N
TZRMJD  53801.38605118223
TZRFRQ            1949.609
TZRSITE                  1
"""


@pytest.mark.parametrize(
    "clock, planet",
    [
        ("UTC(NIST)", "Y"),
        ("UTC(NIST)", "N"),
        ("TT(BIPM2021)", "Y"),
        ("TT(BIPM2021)", "N"),
        ("TT(TAI)", "Y"),
        ("TT(TAI)", "N"),
    ],
)
def test_roundtrip(clock, planet):
    # test for roundtrip accuracy at high precision
    partext = f"{NGC6440E}\nCLK {clock}\nPLANET_SHAPIRO {planet}"
    model = get_model(io.StringIO(partext))

    toas = pint.simulation.make_fake_toas_uniform(
        startMJD=56000,
        endMJD=56400,
        ntoas=100,
        model=model,
        obs="gbt",
        error=1 * u.microsecond,
        freq=1400 * u.MHz,
    )
    res = pint.residuals.Residuals(toas, model)
    toas2 = roundtrip(toas, model)
    res2 = pint.residuals.Residuals(toas2, model)
    assert np.allclose(res.time_resids, res2.time_resids)


def test_noise_addition():
    # basic model, no EFAC or EQUAD
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        57001, 58000, 200, model=model, error=1 * u.us, add_noise=True
    )
    r = pint.residuals.Residuals(toas, model)

    assert np.allclose(toas.get_errors(), 1 * u.us)
    # need a generous rtol because of the small statistics
    assert np.isclose(r.calc_time_resids().std(), 1 * u.us, rtol=0.2)


def test_multiple_freqs():
    # basic model, no EFAC or EQUAD
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        57001,
        58000,
        200,
        model=model,
        error=1 * u.us,
        add_noise=False,
        freq=np.array([1400, 400]) * u.MHz,
    )
    assert (toas.table["freq"][::2] == 1400 * u.MHz).all()
    assert (toas.table["freq"][1::2] == 400 * u.MHz).all()


def test_noise_addition_EFAC():
    # add in EFAC
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        EFAC mjd 57000 58000 2
        """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        57001, 58000, 200, model=model, error=1 * u.us, add_noise=True
    )
    r = pint.residuals.Residuals(toas, model)
    # need a generous rtol because of the small statistics
    assert np.isclose(r.calc_time_resids().std(), 2 * 1 * u.us, rtol=0.2)


def test_noise_addition_EQUAD():
    # add in EFAC
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        EQUAD mjd 57000 58000 5
        """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        57001, 58000, 200, model=model, error=1 * u.us, add_noise=True
    )
    r = pint.residuals.Residuals(toas, model)
    # need a generous rtol because of the small statistics
    assert np.isclose(
        r.calc_time_resids().std(), np.sqrt((1 * u.us) ** 2 + (5 * u.us) ** 2), rtol=0.2
    )


def test_zima():
    parfile = pint.config.examplefile("NGC6440E.par")
    outfile = tempfile.NamedTemporaryFile(suffix="tim")

    error = 1 * u.us

    m = get_model(parfile)

    zima_command = f"zima --freq 1400 --error {error.to_value(u.us)} --ntoa 100 --startMJD 58000 --duration 1000 --obs GBT --addnoise {parfile} {outfile.name}"

    os.system(zima_command)

    t = get_TOAs(outfile.name)
    r = pint.residuals.Residuals(t, m)
    # need a generous rtol because of the small statistics
    assert np.isclose(r.calc_time_resids().std(), 1 * u.us, rtol=0.5)


@pytest.mark.parametrize(
    "MJDs",
    [
        np.linspace(57001, 58000, 200, dtype=np.longdouble) * u.d,
        np.linspace(57001, 58000, 200, dtype=np.longdouble),
        astropy.time.Time(
            np.linspace(57001, 58000, 200, dtype=np.longdouble), format="mjd"
        ),
    ],
)
def test_fake_fromMJDs(MJDs):
    # basic model, no EFAC or EQUAD
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        """
        )
    )
    toas = pint.simulation.make_fake_toas_fromMJDs(
        MJDs, model=model, error=1 * u.us, add_noise=True
    )
    r = pint.residuals.Residuals(toas, model)

    # need a generous rtol because of the small statistics
    assert np.isclose(r.calc_time_resids().std(), 1 * u.us, rtol=0.2)


@pytest.mark.parametrize(
    "t1,t2",
    [
        (57001, 58000),
        (57001 * u.d, 58000 * u.d),
        (
            astropy.time.Time(57001, format="mjd"),
            astropy.time.Time(58000, format="mjd"),
        ),
    ],
)
def test_fake_uniform(t1, t2):
    # basic model, no EFAC or EQUAD
    model = get_model(
        io.StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        t1, t2, 50, model=model, error=1 * u.us, add_noise=True
    )
    r = pint.residuals.Residuals(toas, model)

    # need a generous rtol because of the small statistics
    assert np.isclose(r.calc_time_resids().std(), 1 * u.us, rtol=0.2)


def test_fake_from_timfile():
    # FIXME: this file is unnecessarily huge
    m, t = get_model_and_toas(
        pint.config.examplefile("B1855+09_NANOGrav_9yv1.gls.par"),
        pint.config.examplefile("B1855+09_NANOGrav_9yv1.tim"),
    )

    # refit the data to get rid of a trend
    f = GLSFitter(t, m)
    f.fit_toas()

    r = pint.residuals.Residuals(t, f.model)
    t_sim = pint.simulation.make_fake_toas(t, f.model, add_noise=True)
    r_sim = pint.residuals.Residuals(t_sim, f.model)
    # need a generous rtol because of the small statistics
    assert np.isclose(
        r.calc_time_resids().std(), r_sim.calc_time_resids().std(), rtol=2
    )


def test_fake_highF1():
    m = get_model(os.path.join(datadir, "ngc300nicer.par"))
    m.F1.quantity *= 10
    MJDs = np.linspace(58300, 58400, 100, dtype=np.longdouble) * u.d
    t = pint.simulation.make_fake_toas_fromMJDs(
        MJDs, model=m, add_noise=True, error=1 * u.us
    )
    r = pint.residuals.Residuals(t, m)
    assert np.isclose(r.calc_time_resids(calctype="taylor").std(), 1 * u.us, rtol=0.2)


def test_fake_DMfit():
    """fit only for DM with fake TOAs
    compare the variance of that result against the uncertainties
    """
    parfile = pint.config.examplefile("NGC6440E.par")
    timfile = pint.config.examplefile("NGC6440E.tim")
    m, t = get_model_and_toas(parfile, timfile)

    # pick only data for a DM fit
    t = t[-8:]
    m.RAJ.frozen = True
    m.DECJ.frozen = True
    m.F0.frozen = True
    m.F1.frozen = True
    f = GLSFitter(t, m)
    f.fit_toas()

    N = 30

    DMs = np.zeros(N) * u.pc / u.cm**3
    for iter in range(N):
        t_fake = pint.simulation.make_fake_toas(t, m, add_noise=True)
        f_fake = GLSFitter(t_fake, m)
        f_fake.fit_toas()
        DMs[iter] = f_fake.model.DM.quantity.astype(np.float64)

    assert np.isclose(DMs.std(), f.model.DM.uncertainty, rtol=0.5)
    assert np.abs(DMs.mean() - f.model.DM.quantity) < 5 * f.model.DM.uncertainty


def test_fake_wb_toas():
    parfile = pint.config.examplefile("NGC6440E.par")
    m = get_model(parfile)

    tfu = pint.simulation.make_fake_toas_uniform(
        startMJD=50000,
        endMJD=51000,
        ntoas=100,
        model=m,
        wideband=True,
        add_noise=True,
    )
    assert len(tfu) == 100
    assert all("pp_dm" in f and "pp_dme" in f for f in tfu.get_flags())
