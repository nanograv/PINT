import numpy as np
from astropy import units as u, constants as c
from astropy.time import Time
import os
from pinttestdata import datadir
import pytest

from pint.models import get_model


def test_fb():
    # with FB terms
    m = get_model(os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par"))
    assert np.isclose(m.pb()[0].to_value(u.d), (1 / m.FB0.quantity).to_value(u.d))


@pytest.mark.parametrize(
    "t",
    [
        Time(55555, format="pulsar_mjd", scale="tdb", precision=9),
        55555 * u.d,
        55555.0,
        55555,
        "55555",
        np.array([55555, 55556]),
    ],
)
def test_fb_input(t):
    # with FB terms
    m = get_model(os.path.join(datadir, "J0023+0923_NANOGrav_11yv0.gls.par"))
    pb, pberr = m.pb(t)


def test_pb():
    m = get_model(os.path.join(datadir, "J0437-4715.par"))
    assert np.isclose(m.pb()[0].to_value(u.d), m.PB.quantity.to_value(u.d))


@pytest.mark.parametrize(
    "t",
    [
        Time(55555, format="pulsar_mjd", scale="tdb", precision=9),
        55555 * u.d,
        55555.0,
        55555,
        "55555",
        np.array([55555, 55556]),
    ],
)
def test_pb(t):
    m = get_model(os.path.join(datadir, "J0437-4715.par"))
    pb, pberr = m.pb(t)


def test_every_binary_component_declares_an_epoch_it_has():
    """``pb()`` reads ``binary_epoch_name``; each family must mean it.

    This replaces a ``binary_model_name.startswith("ELL1")`` test, which had to
    be edited centrally for every new family and silently asked a TASC-based
    ``BinaryDDR`` for a ``T0`` it does not have. Declaring the epoch per family
    only helps if the declaration is true, so check it for every registered
    component -- including the suffixed outer-orbit ones, via the component's
    own suffix-aware accessor rather than ``hasattr``.
    """
    from pint.models.pulsar_binary import PulsarBinary
    from pint.models.timing_model import Component

    components = {
        name: cls
        for name, cls in Component.component_types.items()
        if isinstance(cls, type) and issubclass(cls, PulsarBinary)
    }
    assert components, "no binary components are registered"

    undeclared = []
    for name, cls in sorted(components.items()):
        instance = cls()
        epoch = instance.binary_epoch_name
        assert epoch in ("T0", "TASC"), (name, epoch)
        if not instance._hasbp(epoch):
            undeclared.append(f"{name} declares {epoch}")
    assert (
        not undeclared
    ), f"binary components declaring an epoch they do not carry: {undeclared}"
