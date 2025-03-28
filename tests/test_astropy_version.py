import pytest
import importlib
import astropy
import pint


@pytest.fixture
def sandbox():
    class Sandbox:
        pass

    o = Sandbox()
    import astropy

    version = astropy.version.major

    try:
        yield o
    finally:
        astropy.version.major = version


def test_astropy_version_raisesexception(sandbox):
    # check this as a test
    astropy.version.major = 3
    with pytest.raises(ValueError):
        importlib.reload(pint)


def test_astropy_version(sandbox):
    importlib.reload(pint)

    assert astropy.version.major >= 4
