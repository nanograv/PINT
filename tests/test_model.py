import contextlib
import io
import os
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import log
from pinttestdata import datadir

import pint.models as tm
from pint import toa
from pint.residuals import Residuals


def test_simple_parfile():
    """Test a simple parfile without frequency derivatives, proper motion,
    PEPOCH and POSEPOCH."""
    model = tm.get_model(
        io.StringIO(
            """
            PSR J1234+5678
            ELAT 0
            ELONG 0
            F0 1
            DM 10
            """
        )
    )

    assert model.ELAT.value == model.ELONG.value == 0
    assert model.PEPOCH.quantity is None and model.POSEPOCH.quantity is None


def test_model():
    log.setLevel("ERROR")
    # for nice output info, set the following instead
    # log.setLevel('INFO')
    os.chdir(datadir)

    parfile = "J1744-1134.basic.par"
    t1_parfile = "J1744-1134.t1.par"
    timfile = "J1744-1134.Rcvr1_2.GASP.8y.x.tim"

    m = tm.get_model(parfile)
    log.info("model.as_parfile():\n%s" % m.as_parfile())
    try:
        planets = m.PLANET_SHAPIRO.value
    except AttributeError:
        planets = False

    t0 = time.time()
    t = toa.get_TOAs(timfile, planets=planets, include_bipm=False, usepickle=False)
    time_toa = time.time() - t0
    if log.level < 25:
        t.print_summary()
    log.info("Read/corrected TOAs in %.3f sec" % time_toa)

    mjds = t.get_mjds()
    errs = t.get_errors()

    log.info("Computing residuals...")
    t0 = time.time()
    resids_us = Residuals(t, m, use_weighted_mean=False).time_resids.to(u.us)
    time_phase = time.time() - t0
    log.info("Computed phases and residuals in %.3f sec" % time_phase)

    # resids in (approximate) us:
    log.info("RMS PINT residuals are %.3f us" % resids_us.std().value)

    # Get some general2 stuff
    log.info("Running TEMPO2...")
    # Old tempo2 values output
    # tempo2_vals = tempo2_utils.general2(parfile, timfile,
    #                                     ['tt2tb', 'roemer', 'post_phase',
    #                                      'shapiro', 'shapiroJ'])
    tempo2_vals = np.genfromtxt(
        f"{parfile}.tempo2_test", names=True, comments="#", dtype=np.longdouble
    )

    t2_resids = tempo2_vals["post_phase"] / float(m.F0.value) * 1e6 * u.us
    diff_t2 = (resids_us - t2_resids).to(u.ns)
    diff_t2 -= diff_t2.mean()
    log.info(
        "Max resid diff between PINT and T2: %.2f ns" % np.fabs(diff_t2).max().value
    )
    log.info("Std resid diff between PINT and T2: %.2f ns" % diff_t2.std().value)

    assert np.fabs(diff_t2).max() < 10.0 * u.ns

    # run tempo1 also, if the tempo_utils module is available
    did_tempo1 = False
    with contextlib.suppress(Exception):
        t1_result = np.genfromtxt(
            f"{t1_parfile}.tempo_test", names=True, comments="#", dtype=np.longdouble
        )

        t1_resids = t1_result["residuals_phase"] / float(m.F0.value) * 1e6 * u.us
        did_tempo1 = True
        diff_t1 = (resids_us - t1_resids).to(u.ns)
        diff_t1 -= diff_t1.mean()
        log.info(
            "Max resid diff between PINT and T1: %.2f ns" % np.fabs(diff_t1).max().value
        )
        log.info("Std resid diff between PINT and T1: %.2f ns" % diff_t1.std().value)
        diff_t2_t1 = (t2_resids - t1_resids).to(u.ns)
        diff_t2_t1 -= diff_t2_t1.mean()
        log.info(
            "Max resid diff between T1 and T2: %.2f ns"
            % np.fabs(diff_t2_t1).max().value
        )
        log.info("Std resid diff between T1 and T2: %.2f ns" % diff_t2_t1.std().value)
    if did_tempo1 and not planets:
        assert np.fabs(diff_t1).max() < 32.0 * u.ns

    def do_plot():
        plt.clf()
        plt.subplot(211)
        plt.hold(False)
        plt.errorbar(
            mjds.value, resids_us.value, errs.to(u.us).value, fmt=None, label="PINT"
        )
        plt.title("J1744-1134 GBT/GASP timing")
        plt.xlabel("MJD")
        plt.ylabel("Residual (us)")
        plt.legend()
        plt.grid()
        plt.subplot(212)
        plt.plot(mjds, diff_t2, label="PINT - T2")
        plt.hold(True)
        if did_tempo1:
            plt.plot(mjds, diff_t1, label="PINT - T1")
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("Residual diff (ns)")
        plt.legend()

    if log.level < 25:
        do_plot()
        plt.show()
