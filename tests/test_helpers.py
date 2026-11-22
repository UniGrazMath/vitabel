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


def testload_circulation_model():
    import importlib.resources

    import numpy as np

    from vitabel.analysis.circulation.model import load_circulation_model

    asset_dir = importlib.resources.files("vitabel") / "assets"
    assert (asset_dir / "circulation_model.npz").is_file()

    model, scaler = load_circulation_model()
    assert load_circulation_model()[0] is model  # cached
    n_features = scaler.n_features_in_
    features = scaler.transform(np.zeros((2, n_features)))
    assert features.shape == (2, n_features)
    assert model.predict(features).shape == (2,)
    assert model.predict_proba(features).shape == (2, 2)


def test_circulation_model_matches_scikit_learn(vitabel_test_data_dir):
    """The NumPy implementation of the circulation model gives the results of
    the original scikit-learn objects (``SVC``, ``RobustScaler``), stored for
    20 fixed inputs (10 like the training data, 10 off-distribution) by
    scikit-learn 1.9.0 from the pickled model trained with 1.5.1."""
    import json

    import numpy as np

    from vitabel.analysis.circulation.model import load_circulation_model

    reference = json.loads(
        (vitabel_test_data_dir / "circulation_model_reference.json").read_text()
    )
    model, scaler = load_circulation_model()

    features = scaler.transform(reference["inputs"])
    np.testing.assert_allclose(features, reference["scaled"], rtol=1e-12)
    np.testing.assert_allclose(
        model.decision_function(features),
        reference["decision_function"],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_array_equal(model.predict(features), reference["predict"])
    np.testing.assert_allclose(
        model.predict_proba(features), reference["predict_proba"], rtol=0, atol=1e-8
    )
    # the inputs cover the whole range of probabilities, including p ~ 0.5
    probabilities = np.asarray(reference["predict_proba"])[:, 1]
    assert probabilities.min() < 0.01 and probabilities.max() > 0.99
    assert np.any(np.abs(probabilities - 0.5) < 0.1)


def test_circulation_model_probabilities_are_stable():
    """No overflow for large decision values; probabilities sum to 1."""
    import numpy as np

    from vitabel.analysis.circulation.model import load_circulation_model

    model, scaler = load_circulation_model()
    rng = np.random.default_rng(0)
    features = rng.normal(0, 50, (200, scaler.n_features_in_))
    with np.errstate(all="raise"):
        probabilities = model.predict_proba(features)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
