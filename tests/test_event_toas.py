import os
import pytest

from pint.event_toas import read_mission_info_from_heasoft, create_mission_config
from pint.event_toas import get_fits_TOAs
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
