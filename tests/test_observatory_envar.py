import contextlib
import os
import pytest
import importlib

import pint.observatory.topo_obs
import pint.observatory.special_locations
import pint.observatory


@pytest.fixture
def sandbox(tmp_path):
    class Sandbox:
        pass

    o = Sandbox()
    e = os.environ.copy()

    with contextlib.suppress(KeyError):
        del os.environ["PINT_OBS_OVERRIDE"]
    reg = pint.observatory.Observatory._registry.copy()
    o.override_dir = tmp_path / "override"
    o.override_dir.mkdir()

    # just like the original GBT, but ITRF Y is positive here, and negative in the real one
    wronggbt = r"""
    {
        "gbt": {
        "tempo_code": "1",
        "itoa_code": "GB",
        "clock_file": "time_gbt.dat",
        "itrf_xyz": [
            882589.289,
            4924872.368,
            3943729.418
        ],
        "origin": "The Robert C. Byrd Green Bank Telescope.\nThis data was obtained by Joe Swiggum from Ryan Lynch in 2021 September.\n"
        }
    }
    """

    o.name = "gbt.json"
    o.override_file = o.override_dir / o.name
    with o.override_file.open("wt") as f:
        f.write(wronggbt)

    try:
        yield o
    finally:
        os.environ = e
        pint.observatory.Observatory._registry = reg


def test_override_gbt(sandbox):
    pint.observatory.Observatory.clear_registry()
    os.environ["PINT_OBS_OVERRIDE"] = str(sandbox.override_file)
    importlib.reload(pint.observatory.topo_obs)
    importlib.reload(pint.observatory.special_locations)

    newgbt = pint.observatory.Observatory.get("gbt")
    assert newgbt.location.y > 0


def test_override_gbt_loadfunction(sandbox):
    os.environ["PINT_OBS_OVERRIDE"] = str(sandbox.override_file)
    pint.observatory.topo_obs.load_observatories_from_usual_locations(clear=True)
    pint.observatory.special_locations.load_special_locations()

    newgbt = pint.observatory.Observatory.get("gbt")
    assert newgbt.location.y > 0


def test_is_gbt_ok(sandbox):
    pint.observatory.Observatory.clear_registry()

    importlib.reload(pint.observatory.topo_obs)
    importlib.reload(pint.observatory.special_locations)

    newgbt = pint.observatory.Observatory.get("gbt")
    assert newgbt.location.y < 0


def test_is_gbt_ok_loadfunction(sandbox):
    pint.observatory.topo_obs.load_observatories_from_usual_locations(clear=True)
    pint.observatory.special_locations.load_special_locations()

    newgbt = pint.observatory.Observatory.get("gbt")
    assert newgbt.location.y < 0


def test_is_ssb_ok(sandbox):
    pint.observatory.Observatory.clear_registry()

    importlib.reload(pint.observatory.topo_obs)
    importlib.reload(pint.observatory.special_locations)

    ssb = pint.observatory.Observatory.get("ssb")
