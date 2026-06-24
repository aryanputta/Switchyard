from pathlib import Path

from switchyard.features import extract_features
from switchyard.learned_router import LearnedRouter, train
from switchyard.routes import ALL_ROUTES, Route


def _training_set():
    # a separable signal: model numbers -> LEXICAL, negations -> RERANK
    samples = []
    for _ in range(40):
        samples.append((extract_features("sony wh-1000xm5"), Route.LEXICAL))
        samples.append((extract_features("logitech mx-3s mouse"), Route.LEXICAL))
        samples.append((extract_features("headphones without microphone"), Route.RERANK))
        samples.append((extract_features("shoes under $50 no laces"), Route.RERANK))
    return samples


def test_router_learns_separable_signal():
    model = train(_training_set())
    assert model.predict(extract_features("bose nc-700 headset")) == Route.LEXICAL
    assert model.predict(extract_features("jacket without hood")) == Route.RERANK


def test_predict_returns_valid_route():
    model = train(_training_set())
    route = model.predict(extract_features("green tea bags"))
    assert route in ALL_ROUTES


def test_save_and_load_roundtrip(tmp_path: Path):
    model = train(_training_set())
    path = tmp_path / "router.json"
    model.save(path)
    loaded = LearnedRouter.load(path)
    q = extract_features("sony wh-1000xm5")
    assert loaded.predict(q) == model.predict(q)


def test_empty_training_set_is_safe():
    model = train([])
    assert model.predict(extract_features("anything")) in ALL_ROUTES
