import os

import astropy.units as u
import contextlib
import numpy as np
import pytest
from astropy.time import Time

from pint.observatory import NoClockCorrections
from pint.observatory.clock_file import ClockFile
from pint.observatory.topo_obs import find_clock_file


@pytest.fixture
def sandbox(tmp_path):
    class Sandbox:
        pass

    o = Sandbox()
    e = os.environ.copy()
    with contextlib.suppress(KeyError):
        del os.environ["PINT_CLOCK_OVERRIDE"]
    with contextlib.suppress(KeyError):
        del os.environ["TEMPO"]
    with contextlib.suppress(KeyError):
        del os.environ["TEMPO2"]
    o.override_dir = tmp_path / "override"
    o.override_dir.mkdir()
    o.repo_dir = tmp_path / "repo"
    o.repo_dir.mkdir()
    o.clocks = [
        ClockFile(mjd=np.array([50000 + i, 59000]), clock=np.zeros(2) * u.us)
        for i in range(10)
    ]
    o.name = "fake.clk"
    o.hdrline = "# FAKE1 FAKE2"

    o.override_clock = o.clocks[0]
    o.override_clock_file = o.override_dir / o.name
    with o.override_clock_file.open("wt") as f:
        o.override_clock.write_tempo2_clock_file(f, hdrline=o.hdrline)

    o.repo_clock = o.clocks[1]
    o.repo_clock_file = o.repo_dir / "T2runtime" / "clock" / o.name
    o.repo_clock_file.parent.mkdir(parents=True)
    with o.repo_clock_file.open("wt") as f:
        o.repo_clock.write_tempo2_clock_file(f, hdrline=o.hdrline)
    (o.repo_dir / "index.txt").write_text(
        """
        # File                                   Update (days)   Invalid if older than
        T2runtime/clock/fake.clk                 7.0   ---
        """
    )
    o.repo_uri = f"{o.repo_dir.as_uri()}/"

    o.t2_dir = tmp_path / "t2"
    o.t2_clock = o.clocks[2]
    o.t2_clock_file = o.t2_dir / "clock" / o.name
    o.t2_clock_file.parent.mkdir(parents=True)
    with o.t2_clock_file.open("wt") as f:
        o.t2_clock.write_tempo2_clock_file(f, hdrline=o.hdrline)

    try:
        yield o
    finally:
        os.environ = e


def test_can_find_in_repo(sandbox, temp_cache):
    c = find_clock_file(sandbox.name, format="tempo2", url_base=sandbox.repo_uri)
    assert c.time.mjd[0] == sandbox.repo_clock.time.mjd[0]


def test_pint_env_overrides(sandbox, temp_cache):
    os.environ["PINT_CLOCK_OVERRIDE"] = str(sandbox.override_dir)
    c = find_clock_file(sandbox.name, format="tempo2", url_base=sandbox.repo_uri)
    assert c.time.mjd[0] == sandbox.override_clock.time.mjd[0]
    # FIXME: how to test that a warning is emitted?


def test_obeys_clock_dir_tempo2(sandbox, temp_cache):
    os.environ["TEMPO2"] = str(sandbox.t2_dir)
    c = find_clock_file(
        sandbox.name, format="tempo2", clock_dir="tempo2", url_base=sandbox.repo_uri
    )
    assert c.time.mjd[0] == sandbox.t2_clock.time.mjd[0]


def test_obeys_clock_dir_specific(sandbox, temp_cache):
    c = find_clock_file(
        sandbox.name,
        format="tempo2",
        clock_dir=sandbox.override_dir,
        url_base=sandbox.repo_uri,
    )
    assert c.time.mjd[0] == sandbox.override_clock.time.mjd[0]


# FIXME: how to test fallback to the runtime dir?
@pytest.mark.parametrize(
    "name, format",
    [
        ("tai2tt_bipm2020.clk", "tempo2"),
        ("tai2tt_bipm2019.clk", "tempo2"),
        ("gps2utc.clk", "tempo2"),
        ("time_gbt.dat", "tempo"),
    ],
)
def test_can_find_known_files(name, format):
    c = find_clock_file(name, format=format)
    c.evaluate(Time(59000, format="pulsar_mjd"), limits="error")


def test_tempo2_file_nonexistent_raises(sandbox):
    os.environ["TEMPO2"] = str(sandbox.override_dir)
    with pytest.raises(NoClockCorrections):
        find_clock_file("nonexistent.clk", format="tempo2", clock_dir="tempo2")


def test_tempo_file_nonexistent_raises(sandbox):
    os.environ["TEMPO"] = str(sandbox.override_dir)
    with pytest.raises(NoClockCorrections):
        find_clock_file("nonexistent.dat", format="tempo", clock_dir="tempo")


def test_nonexistent_in_repo_raises(sandbox, temp_cache):
    with pytest.raises(NoClockCorrections):
        find_clock_file("nonexistent.clk", format="tempo2", url_base=sandbox.repo_uri)


def test_nonexistent_specific_path_raises(sandbox, tmp_path):
    with pytest.raises(NoClockCorrections):
        find_clock_file(
            "nonexistent.clk",
            clock_dir=tmp_path,
            format="tempo2",
            url_base=sandbox.repo_uri,
        )
