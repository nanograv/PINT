from pint.extern._version import get_versions


def test_version():
    ver = get_versions()
    assert isinstance(ver, dict)
