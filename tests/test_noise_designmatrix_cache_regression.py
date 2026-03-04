import numpy as np

from pint.config import examplefile
from pint.models.timing_model import Component
from pint.models import get_model_and_toas


def _model_and_toas():
    parfile = examplefile("B1855+09_NANOGrav_9yv1.gls.par")
    timfile = examplefile("B1855+09_NANOGrav_9yv1.tim")
    return get_model_and_toas(parfile, timfile)


def _multi_basis_model_and_toas():
    model, toas = _model_and_toas()
    all_components = Component.component_types

    model.add_component(all_components["PLDMNoise"](), validate=False)
    model["TNDMAMP"].quantity = -13
    model["TNDMGAM"].quantity = 1.2
    model["TNDMC"].value = 8

    model.add_component(all_components["PLSWNoise"](), validate=False)
    model["TNSWAMP"].quantity = -12
    model["TNSWGAM"].quantity = -2.0
    model["TNSWC"].value = 8

    model.add_component(all_components["PLChromNoise"](), validate=False)
    model["TNCHROMAMP"].quantity = -14
    model["TNCHROMGAM"].quantity = 1.2
    model["TNCHROMC"].value = 8

    model.add_component(all_components["ChromaticCM"](), validate=False)
    model["TNCHROMIDX"].value = 4

    model.validate()
    return model, toas


def test_noise_designmatrix_cache_hit_avoids_recomputing_basis(monkeypatch):
    model, toas = _model_and_toas()
    component = model.components["PLRedNoise"]

    calls = {"n": 0}
    original = component.get_noise_basis

    def wrapped_get_noise_basis(current_toas):
        calls["n"] += 1
        return original(current_toas)

    monkeypatch.setattr(component, "get_noise_basis", wrapped_get_noise_basis)

    first = model.noise_model_designmatrix(toas)
    second = model.noise_model_designmatrix(toas)

    assert first is second
    assert calls["n"] == 1


def test_noise_designmatrix_cache_invalidates_on_noise_parameter_change():
    model, toas = _model_and_toas()

    first = model.noise_model_designmatrix(toas)

    old_value = int(model.TNREDC.value) if model.TNREDC.value is not None else 30
    model.TNREDC.value = old_value + 1
    model.setup()
    model.validate()

    second = model.noise_model_designmatrix(toas)
    third = model.noise_model_designmatrix(toas)

    assert second is third
    assert second.shape[1] == first.shape[1] + 2
    assert not np.array_equal(first, second)


def test_noise_designmatrix_cache_multi_basis_hit_and_invalidation(monkeypatch):
    model, toas = _multi_basis_model_and_toas()

    component_names = ["PLRedNoise", "PLDMNoise", "PLSWNoise", "PLChromNoise"]
    calls = {name: 0 for name in component_names}

    originals = {
        name: model.components[name].get_noise_basis for name in component_names
    }

    for name in component_names:
        component = model.components[name]
        original = originals[name]

        def make_wrapper(component_name, original_func):
            def wrapped_get_noise_basis(current_toas):
                calls[component_name] += 1
                return original_func(current_toas)

            return wrapped_get_noise_basis

        monkeypatch.setattr(component, "get_noise_basis", make_wrapper(name, original))

    first = model.noise_model_designmatrix(toas)
    second = model.noise_model_designmatrix(toas)

    assert first is second
    for name in component_names:
        assert calls[name] == 1

    model.TNCHROMC.value = int(model.TNCHROMC.value) + 1
    model.setup()
    model.validate()

    third = model.noise_model_designmatrix(toas)
    fourth = model.noise_model_designmatrix(toas)

    assert third is fourth
    assert calls["PLRedNoise"] == 2
    assert calls["PLDMNoise"] == 2
    assert calls["PLSWNoise"] == 2
    assert calls["PLChromNoise"] == 2
    assert third.shape[1] == first.shape[1] + 2


def test_noise_basis_weight_cache_hit_avoids_recomputing_weights(monkeypatch):
    model, toas = _model_and_toas()
    component = model.components["PLRedNoise"]

    calls = {"n": 0}
    original = component.get_noise_weights

    def wrapped_get_noise_weights(current_toas):
        calls["n"] += 1
        return original(current_toas)

    monkeypatch.setattr(component, "get_noise_weights", wrapped_get_noise_weights)

    first = model.noise_model_basis_weight(toas)
    second = model.noise_model_basis_weight(toas)

    assert first is second
    assert calls["n"] == 1


def test_noise_basis_weight_cache_invalidates_on_noise_parameter_change():
    model, toas = _model_and_toas()

    first = model.noise_model_basis_weight(toas)

    old_value = int(model.TNREDC.value) if model.TNREDC.value is not None else 30
    model.TNREDC.value = old_value + 1
    model.setup()
    model.validate()

    second = model.noise_model_basis_weight(toas)
    third = model.noise_model_basis_weight(toas)

    assert second is third
    assert second.shape[0] == first.shape[0] + 2
    assert not np.array_equal(first, second)
