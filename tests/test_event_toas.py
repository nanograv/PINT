import os
import pytest
import numpy as np

from astropy import units as u

from pint.event_toas import read_mission_info_from_heasoft, create_mission_config
from pint.event_toas import get_fits_TOAs, get_NICER_TOAs, _default_uncertainty
from pinttestdata import datadir


def test_xselect_mdb_is_found_headas(monkeypatch, tmp_path):
    """Test event file reading."""
    path = tmp_path / "bin"
    path.mkdir()
    f = path / "xselect.mdb"
    f.write_text("MAXI:submkey       NONE\nMAXI:instkey       INSTRUME")

    monkeypatch.setenv("HEADAS", tmp_path)

    info = read_mission_info_from_heasoft()
    assert "nustar" not in info


def test_create_mission_config_headas(monkeypatch, tmp_path):
    """Test event file reading."""
    path = tmp_path / "bin"
    path.mkdir()
    f = path / "xselect.mdb"
    f.write_text("!\nINTEGRAL:events       EVENTS\nINTEGRAL:ecol     PHA")
    f.write_text("!\nINTEGRAL:SPI:ecol     PHA")

    monkeypatch.setenv("HEADAS", tmp_path)

    info = create_mission_config()
    assert "xte" in info
    assert info["xte"]["fits_extension"] == 1


def test_create_mission_config_headas_missing_file_no_fail(monkeypatch, tmp_path):
    """Test event file reading."""
    monkeypatch.setenv("HEADAS", tmp_path)
    info = create_mission_config()
    assert "xte" in info
    assert info["xte"]["fits_extension"] == 1


def test_load_events_wrongext_raises():
    eventfile_nicer_topo = os.path.join(datadir, "sgr1830kgfilt.evt")
    msg = "At the moment, only data in the first FITS extension"
    with pytest.raises(ValueError) as excinfo:
        # Not sure how to test that the warning is raised, with Astropy's log system
        # Anyway, here I'm testing another error
        get_fits_TOAs(eventfile_nicer_topo, mission="xdsgse", extension=2)
    assert msg in str(excinfo.value)


def test_load_events_wrongext_text_raises():
    eventfile_nicer_topo = os.path.join(datadir, "sgr1830kgfilt.evt")
    msg = "First table in FITS file"
    with pytest.raises(RuntimeError) as excinfo:
        # Not sure how to test that the warning is raised, with Astropy's log system
        # Anyway, here I'm testing another error
        get_fits_TOAs(eventfile_nicer_topo, mission="xdsgse", extension="dafasdfa")
    assert msg in str(excinfo.value)


def test_for_toa_errors_default():
    eventfile_nicer = datadir / "ngc300nicer_bary.evt"

    ts = get_NICER_TOAs(
        eventfile_nicer,
    )
    assert np.all(ts.get_errors() == _default_uncertainty["NICER"])


@pytest.mark.parametrize("errors", [2, 2 * u.us])
def test_for_toa_errors_manual(errors):
    eventfile_nicer = datadir / "ngc300nicer_bary.evt"

    ts = get_NICER_TOAs(
        eventfile_nicer,
        errors=errors,
    )
    assert np.all(ts.get_errors() == 2 * u.us)
