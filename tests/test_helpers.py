from vitabel.utils.helpers import match_object


def test_match_object():
    class Container:
        pass

    container = Container()
    container.a = 1
    container.b = 2
    container.c = 3
    container.metadata = {"name": "test", "version": "0.1"}

    assert match_object(container)
    assert match_object(container, a=1)
    assert match_object(container, a=1, b=2, c=3)
    assert not match_object(container, x=42)
    assert not match_object(container, a=1, c=3, x=42)
    assert not match_object(container, a=1, c=42)
    assert match_object(container, metadata={"name": "test"})
    assert match_object(container, metadata={"name": "test", "version": "0.1"})
    assert not match_object(container, metadata={"name": "test", "version": "0.2"})
    assert not match_object(container, metadata={"name": "test", "x": 42})


def test_load_circulation_model():
    import importlib.resources

    import numpy as np

    from vitabel.utils.helpers import _load_circulation_model

    asset_dir = importlib.resources.files("vitabel") / "assets"
    assert (asset_dir / "circ-classification_model.joblib").is_file()
    assert (asset_dir / "circ-classification_scaler.joblib").is_file()

    model, scaler = _load_circulation_model()
    assert _load_circulation_model()[0] is model  # cached
    n_features = scaler.n_features_in_
    features = scaler.transform(np.zeros((2, n_features)))
    assert features.shape == (2, n_features)
    assert model.predict(features).shape == (2,)
    assert model.predict_proba(features).shape == (2, 2)
