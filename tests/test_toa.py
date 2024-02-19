import pytest
import io
import os
import re
import pytest
import numpy as np

import astropy.units as u
from astropy.time import Time
from datetime import datetime
from pinttestdata import datadir

from pint.models import get_model
from pint.observatory import get_observatory
from pint.toa import TOA, TOAs
import pint.toa
from pint.simulation import make_fake_toas_uniform


class TestTOA:
    """Test of TOA class"""

    def setup_method(self):
        self.MJD = 57000

    def test_units(self):
        with pytest.raises(u.UnitConversionError):
            t = TOA(self.MJD * u.m)
        with pytest.raises(u.UnitConversionError):
            t = TOA((self.MJD * u.m, 0))
        t = TOA((self.MJD * u.day).to(u.s))
        with pytest.raises(u.UnitConversionError):
            t = TOA((self.MJD * u.day, 0))
        t = TOA((self.MJD * u.day, 0 * u.day))
        with pytest.raises(u.UnitConversionError):
            t = TOA(self.MJD, error=1 * u.m)
        t = TOA(self.MJD, freq=100 * u.kHz)
        with pytest.raises(u.UnitConversionError):
            t = TOA(self.MJD, freq=100 * u.s)

    def test_precision_mjd(self):
        t = TOA(self.MJD)
        assert t.mjd.precision == 9

    def test_precision_time(self):
        t = TOA(Time("2008-08-19", format="iso", precision=1))
        assert t.mjd.precision == 9

    def test_typo(self):
        TOA(self.MJD, errror="1")
        with pytest.raises(TypeError):
            TOA(self.MJD, errror=1, flags={})

    def test_toa_object(self):
        toatime = Time(datetime.now())
        toaerr = u.Quantity(1e-6, "s")
        freq = u.Quantity(1400, "MHz")
        obs = "ao"

        # scale should be None when MJD is a Time
        with pytest.raises(ValueError):
            toa = TOA(MJD=toatime, error=toaerr, freq=freq, obs=obs, scale="utc")

        # flags should be stored without their leading -
        with pytest.raises(ValueError):
            toa = TOA(
                MJD=toatime, error=toaerr, freq=freq, obs=obs, flags={"-foo": "foo1"}
            )

        # Invalid flag
        with pytest.raises(ValueError):
            toa = TOA(
                MJD=toatime, error=toaerr, freq=freq, obs=obs, flags={"$": "foo1"}
            )
        with pytest.raises(ValueError):
            toa = TOA(MJD=toatime, error=toaerr, freq=freq, obs=obs, flags={"foo": 1})

        toa = TOA(MJD=toatime, error=toaerr, freq=freq, obs=obs, foo="foo1")
        assert "foo1" in str(toa)
        assert "bla" in toa.as_line(name="bla")
        assert len(toa.flags) > 0

        # Missing name
        with pytest.raises(ValueError):
            toa.as_line()

        toa = TOA(MJD=toatime, error=toaerr, freq=freq, obs=obs, foo="foo1", name="bla")
        assert "bla" in toa.as_line()


class TestTOAs:
    """Test of TOAs class"""

    def setup_method(self):
        self.freq = 1440.012345678 * u.MHz
        self.obs = "gbt"
        self.MJD = 57000
        self.error = 3.0

    def test_make_toas(self):
        t = TOA(self.MJD, freq=self.freq, obs=self.obs, error=self.error)
        t_list = [t, t]
        assert t_list[0].mjd.precision == 9
        assert t_list[1].mjd.precision == 9
        assert t_list[0].mjd.location is not None
        assert t_list[1].mjd.location is not None
        # Check information in the TOAs table
        toas = TOAs(toalist=t_list)
        assert toas.table[1]["freq"] == self.freq.to_value(u.MHz)
        assert toas.table["freq"].unit == self.freq.unit
        assert toas.table[1]["obs"] == self.obs
        assert toas.table[1]["error"] == self.error
        assert toas.table["error"].unit == u.us
        assert toas.table["mjd"][0].precision == 9
        assert toas.table["mjd"][0].location is not None

    def test_multiple_observatories_stay_attached(self):
        obs1 = "gbt"
        obs2 = "ao"
        obs3 = "barycenter"
        site1 = get_observatory(obs1)
        site2 = get_observatory(obs2)
        site3 = get_observatory(obs3)
        t1 = TOA(self.MJD, freq=self.freq, obs=obs1, error=self.error)
        t2 = TOA(self.MJD + 1.0, freq=self.freq, obs=obs2, error=self.error)
        t3 = TOA(self.MJD + 1.0, freq=self.freq, obs=obs3, error=self.error)

        toas = TOAs(toalist=[t1, t2, t3])

        assert toas.table["obs"][0] == site1.name
        assert toas.table["mjd"][0] == t1.mjd
        assert toas.table["obs"][1] == site2.name
        assert toas.table["mjd"][1] == t2.mjd
        assert toas.table["obs"][2] == site3.name
        assert toas.table["mjd"][2] == t3.mjd

        # obs in time object
        assert toas.table["mjd"][0].location == site1.earth_location_itrf()
        assert toas.table["mjd"][1].location == site2.earth_location_itrf()
        assert toas.table["mjd"][2].location == site3.earth_location_itrf()


def test_toa_summary():
    m = get_model(os.path.join(datadir, "NGC6440E.par"))
    toas = make_fake_toas_uniform(57000, 58000, 10, m, obs="ao", freq=2000 * u.MHz)
    s = toas.get_summary()

    assert re.search(r"Number of commands: *0", s)
    assert re.search(r"Number of observatories: *1 *\['arecibo'\]", s)
    assert re.search(r"MJD span: *57000\.000 to 58000\.000", s)
    assert re.search(
        r"Date span: *20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ to 20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+",
        s,
    )
    assert re.search(r"arecibo TOAs *\(10\):", s)
    assert re.search(r" *Min freq: *2000.000 MHz", s)
    assert re.search(r" *Max freq: *2000.000 MHz", s)
    assert re.search(r" *Min error: *1 us", s)
    assert re.search(r" *Max error: *1 us", s)
    assert re.search(r" *Median error: *1 us", s)


def test_merge_toas():
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
    toas = make_fake_toas_uniform(
        57000, 58000, 20, model=model, error=1 * u.us, add_noise=False
    )
    toas2 = make_fake_toas_uniform(
        59000, 60000, 20, model=model, error=1 * u.us, add_noise=False
    )
    toas_out = pint.toa.merge_TOAs([toas, toas2])
    toas_outb = toas + toas2
    assert np.all(toas_out.table == toas_outb.table)


def test_mix_nb_wb():
    with pytest.raises(ValueError):
        t1 = pint.toa.get_TOAs(
            io.StringIO(
                """
                fake.ff 1430.000000 53393.561383615118386   0.178  ao  -fe L-wide -be ASP -pp_dm 10.0
                fake.ff 1430.000000 53394.561383615118386   0.178  ao  -fe L-wide -be ASP
                """
            )
        )
