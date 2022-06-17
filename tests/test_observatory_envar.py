import os

import pytest


@pytest.fixture
def sandbox(tmp_path):
    class Sandbox:
        pass

    o = Sandbox()
    e = os.environ.copy()
    try:
        del os.environ["PINT_OBS_OVERRIDE"]
    except KeyError:
        pass

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


def test_override_gbt(sandbox):
    os.environ["PINT_OBS_OVERRIDE"] = str(sandbox.override_file)
    import pint.observatory.observatories
    from pint.observatory import get_observatory

    newgbt = get_observatory("gbt")
    assert newgbt._loc_itrf.y > 0
