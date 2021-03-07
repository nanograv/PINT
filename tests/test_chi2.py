from io import StringIO

import astropy.time
import astropy.units as u
import numpy as np
import scipy.stats

import pint.models
import pint.toa


def test_ecorr_chi2_probabilistic():
    # This test should fail for 2% of seeds, but if it fails more broadly something is wrong
    s = np.random.RandomState(0)

    model = pint.models.get_model(
        StringIO(
            """
            PSR J1234+5678
            EPHEM DE440
            ELAT 0
            ELONG 0
            PEPOCH 57000
            POSEPOCH 57000
            F0 1
            DM 10
            ECORR tel ao 10
            """
        )
    )
    model.free_params = []
    t = pint.toa.merge_TOAs(
        [
            pint.toa.make_fake_toas(56000, 57000, 21, model, freq=f, obs="ao")
            for f in [1000, 1400, 1700, 2000]
        ]
    )
    t.compute_pulse_numbers(model)
    for i in range(10):
        r = pint.residuals.Residuals(t, model, track_mode="use_pulse_numbers")
        if abs(r.time_resids).max() < 1 * u.ns:
            break
        t.adjust_TOAs(astropy.time.TimeDelta(-r.time_resids))
    else:
        raise ValueError(
            "Unable to make fake residuals - left over errors are {}".format(
                abs(r.time_resids).max()
            )
        )
    delta = np.zeros(len(t)) * u.us
    groups = np.array(t.get_groups())
    for g in np.unique(groups):
        delta[groups == g] = model.ECORR1.quantity * s.standard_normal()
    delta += t.table["error"] * s.standard_normal(len(delta))
    t.adjust_TOAs(astropy.time.TimeDelta(delta))

    resids = pint.residuals.Residuals(t, model)

    assert (
        scipy.stats.chi2(resids.dof).ppf(0.01)
        < resids.chi2
        < scipy.stats.chi2(resids.dof).ppf(0.99)
    )
