from pint.event_toas import read_mission_info_from_heasoft


def test_xselect_mdb_is_found_headas(monkeypatch, tmp_path):
    """Test event file reading."""
    path = tmp_path / 'bin'
    path.mkdir()
    f = path / 'xselect.mdb'
    f.write_text("MAXI:submkey       NONE\nMAXI:instkey       INSTRUME")

    monkeypatch.setenv("HEADAS", tmp_path)

    info = read_mission_info_from_heasoft()
    assert "NUSTAR" not in info
