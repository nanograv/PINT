import astropy.units as u
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
        r.calc_time_resids().std(),
        np.sqrt((1 * u.us) ** 2 + (5 * u.us) ** 2),
        rtol=0.2,
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
    assert np.isclose(r.calc_time_resids().std(), 1 * u.us, rtol=0.2,)


def test_fake_from_timfile():
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
        r.calc_time_resids().std(), r_sim.calc_time_resids().std(), rtol=2,
    )
