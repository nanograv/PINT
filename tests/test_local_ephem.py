from pinttestdata import datadir
import pint.logging
import pint.simulation
from pint.models.model_builder import get_model, get_model_and_toas
from pint.toa import get_TOAs
import pint.solar_system_ephemerides
import io
import os
from astropy import units as u
import astropy.utils.data
import astropy.utils.state

pint.logging.setup("DEBUG")


def test_local_file():
    # make sure this file does not exist locally
    astropy.utils.data.clear_download_cache(
        "https://data.nanograv.org/static/data/ephem/de118.bsp"
    )

    out = pint.solar_system_ephemerides.load_kernel("de118", path=datadir)
    model = get_model(
        io.StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            EPHEM DE118
            """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        57001, 58000, 200, model=model, error=1 * u.us, add_noise=True
    )
    assert isinstance(out, str)


def test_downloaded_file():
    # make sure this file does not exist locally until we download it
    astropy.utils.data.clear_download_cache(
        "https://data.nanograv.org/static/data/ephem/de118.bsp"
    )
    pint.solar_system_ephemerides.clear_loaded_ephem()
    out = pint.solar_system_ephemerides.load_kernel("de118")
    model = get_model(
        io.StringIO(
            """
            PSRJ J1234+5678
            ELAT 0
            ELONG 0
            DM 10
            F0 1
            PEPOCH 58000
            EPHEM DE118
            """
        )
    )
    toas = pint.simulation.make_fake_toas_uniform(
        57001, 58000, 200, model=model, error=1 * u.us, add_noise=True
    )
    assert isinstance(out, bool)
