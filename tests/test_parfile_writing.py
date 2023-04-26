"""Various tests to assess the performance of parfile writing."""
import numbers
import os
from io import StringIO
import pytest

import astropy.units as u
import numpy as np

import pint.models.model_builder as mb
import pint.models.parameter as mp
import pint.toa as toa
from pint.residuals import Residuals
from pinttestdata import datadir


def test_parfile_write(tmp_path):
    parfileB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par")
    timB1855 = os.path.join(datadir, "B1855+09_NANOGrav_9yv1.tim")
    toasB1855 = toa.get_TOAs(timB1855, ephem="DE421", planets=False, include_bipm=False)
    modelB1855 = mb.get_model(parfileB1855)
    out_parfile = tmp_path / "test_parfile_write.par"
    # change parameter value
    for p in modelB1855.params:
        par = getattr(modelB1855, p)
        # Change value for 20%
        if isinstance(par.value, numbers.Number) and not isinstance(
            par, mp.MJDParameter
        ):
            ov = par.value
            par.value = ov * 0.8
    res = Residuals(toasB1855, modelB1855, use_weighted_mean=False).time_resids.to(u.s)
    with open(out_parfile, "w") as f:
        f.write(modelB1855.as_parfile())
    read_model = mb.get_model(out_parfile)
    read_res = Residuals(toasB1855, read_model, use_weighted_mean=False).time_resids.to(
        u.s
    )
    assert np.all(
        np.abs(read_res.value - res.value) < 1e-15
    ), "Output parfile did not produce same residuals."
    for pp in modelB1855.params:
        par_ori = getattr(modelB1855, pp)
        par_read = getattr(read_model, pp)
        if par_ori.uncertainty_value is not None:
            unc_diff = par_ori.uncertainty_value - par_read.uncertainty_value
            assert np.abs(unc_diff) < 1e-15, (
                pp
                + "uncertainty does not keep the precision. at"
                + str(np.abs(unc_diff))
            )


@pytest.mark.parametrize(
    "name,alias", [("ECORR1", "TNECORR"), ("A1DOT", "XDOT"), ("PSR", "PSRJ")]
)
def test_write_custom_aliases(name, alias):
    p = mb.get_model(
        StringIO(
            """
        PSR J1234+5678
        PEPOCH 57000
        F0 1
        BINARY BT
        A1 1
        PB 1
        T0 57000
        OM 0
        A1DOT 1e-10
        ECORR tel arecibo 1
        """
        )
    )
    getattr(p, name).use_alias = alias
    assert alias in p.as_parfile()


@pytest.mark.parametrize(
    "name,alias",
    [
        pytest.param(
            "ECORR", "TNECORR", marks=pytest.mark.xfail(reason="Alias not recognized")
        ),
        ("A1DOT", "XDOT"),
        ("PSR", "PSRJ"),
    ],
)
def test_retain_aliases(name, alias):
    p = mb.get_model(
        StringIO(
            """
        PSR J1234+5678
        PEPOCH 57000
        F0 1
        BINARY BT
        A1 1
        PB 1
        T0 57000
        OM 0
        A1DOT 1e-10
        ECORR tel arecibo 1
        """.replace(
                name, alias
            )
        )
    )
    assert alias in p.as_parfile()
