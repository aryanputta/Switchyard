from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from switchyard.features import QueryFeatures
from switchyard.routes import ALL_ROUTES, Route

"""Learned route classifier.

The deployable router predicts the best route from query features alone, with
no access to labels. This module trains a multinomial logistic regression from
scratch with gradient descent. It has zero external dependencies, so the gate,
training, and serving all run anywhere, and the trained weights serialize to a
small JSON the Go service can load.

LightGBM is the documented production swap-in (see ARCHITECTURE Phase 2): same
feature vector, same per-query-route training log, richer non-linear boundary.
The interface here (train -> predict -> save/load) does not change.
"""

FEATURE_ORDER = [
    "token_count",
    "char_count",
    "has_question",
    "has_model_number",
    "has_money",
    "has_negation",
    "has_compatibility",
    "has_color",
]


def _vec(features: QueryFeatures) -> list[float]:
    d = features.as_dict()
    # standardize the two count features so gradient descent is well conditioned
    d = dict(d)
    d["token_count"] = d["token_count"] / 10.0
    d["char_count"] = d["char_count"] / 60.0
    return [1.0] + [d[name] for name in FEATURE_ORDER]


def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]


@dataclass
class LearnedRouter:
    weights: list[list[float]]  # [route][feature], includes bias at index 0
    routes: list[str]

    def predict(self, features: QueryFeatures) -> Route:
        x = _vec(features)
        scores = [sum(w * xi for w, xi in zip(row, x)) for row in self.weights]
        best = max(range(len(scores)), key=lambda i: scores[i])
        return Route(self.routes[best])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"routes": self.routes, "weights": self.weights}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> "LearnedRouter":
        payload = json.loads(Path(path).read_text())
        return cls(weights=payload["weights"], routes=payload["routes"])


def train(
    samples: list[tuple[QueryFeatures, Route]],
    *,
    epochs: int = 300,
    lr: float = 0.3,
    l2: float = 1e-3,
    seed: int = 7,
) -> LearnedRouter:
    """Train the multinomial logistic router on (features, oracle-route) pairs.
    The oracle route is the per-query utility-maximizing route from the route
    log, so the classifier learns to imitate the oracle from features alone."""
    routes = [r.value for r in ALL_ROUTES]
    route_index = {r: i for i, r in enumerate(routes)}
    n_features = len(FEATURE_ORDER) + 1
    weights = [[0.0] * n_features for _ in routes]

    xs = [_vec(f) for f, _ in samples]
    ys = [route_index[r.value] for _, r in samples]
    if not xs:
        return LearnedRouter(weights=weights, routes=routes)

    for _ in range(epochs):
        grads = [[0.0] * n_features for _ in routes]
        for x, y in zip(xs, ys):
            scores = [sum(w * xi for w, xi in zip(row, x)) for row in weights]
            probs = _softmax(scores)
            for k in range(len(routes)):
                err = probs[k] - (1.0 if k == y else 0.0)
                for j in range(n_features):
                    grads[k][j] += err * x[j]
        m = len(xs)
        for k in range(len(routes)):
            for j in range(n_features):
                g = grads[k][j] / m + l2 * weights[k][j]
                weights[k][j] -= lr * g

    return LearnedRouter(weights=weights, routes=routes)
