"""Pre-trained classification model for the detection of circulation (ROSC).

The model is a support vector classifier (RBF kernel, two classes, with
probability estimates) on features standardized by a robust scaler. It was
trained with scikit-learn; its fitted parameters are stored as plain arrays
(``assets/circulation_model.npz``, see ``scripts/export_circulation_model.py``)
and evaluated with NumPy, giving the same results as the scikit-learn objects
(up to floating point rounding).
"""

from __future__ import annotations

import functools
import importlib.resources
import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("vitabel")

# libsvm clips the pairwise probabilities to [MIN_PROB, 1 - MIN_PROB]
_MIN_PROB = 1e-7


class RobustScaler:
    """Feature scaling ``(x - center) / scale``, as
    ``sklearn.preprocessing.RobustScaler.transform``."""

    def __init__(self, center: npt.NDArray, scale: npt.NDArray):
        self.center = center
        self.scale = scale

    @property
    def n_features_in_(self) -> int:
        return len(self.center)

    def transform(self, X: npt.ArrayLike) -> npt.NDArray:
        return (np.asarray(X, dtype=float) - self.center) / self.scale


class SupportVectorClassifier:
    """A fitted binary support vector classifier with RBF kernel; the methods
    give the same results as those of ``sklearn.svm.SVC``."""

    def __init__(
        self,
        support_vectors: npt.NDArray,
        dual_coef: npt.NDArray,
        intercept: float,
        gamma: float,
        prob_a: float,
        prob_b: float,
        classes: npt.NDArray,
    ):
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = float(intercept)
        self.gamma = float(gamma)
        self.prob_a = float(prob_a)
        self.prob_b = float(prob_b)
        self.classes = classes
        self._support_norms = np.einsum("ij,ij->i", support_vectors, support_vectors)

    def decision_function(self, X: npt.ArrayLike) -> npt.NDArray:
        """Signed distance to the separating hyperplane (positive: second
        class)."""
        X = np.asarray(X, dtype=float)
        squared_distances = (
            np.einsum("ij,ij->i", X, X)[:, None]
            + self._support_norms[None, :]
            - 2 * X @ self.support_vectors.T
        )
        kernel = np.exp(-self.gamma * np.maximum(squared_distances, 0))
        return kernel @ self.dual_coef + self.intercept

    def predict(self, X: npt.ArrayLike) -> npt.NDArray:
        """The predicted class of every sample."""
        return np.where(self.decision_function(X) > 0, self.classes[1], self.classes[0])

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray:
        """Probabilities of the two classes (columns in the order of
        ``classes``), as computed by libsvm: Platt scaling of the decision
        function, followed by libsvm's (iterative) coupling of the pairwise
        probabilities."""
        # libsvm's decision value has the opposite sign of scikit-learn's
        f = -self.decision_function(X) * self.prob_a + self.prob_b
        with np.errstate(over="ignore"):
            # 1 / (1 + exp(f)), evaluated without overflow
            r = np.where(f >= 0, np.exp(-f) / (1 + np.exp(-f)), 1 / (1 + np.exp(f)))
        r = np.clip(r, _MIN_PROB, 1 - _MIN_PROB)
        return np.column_stack(_couple_binary_probabilities(r))


def _couple_binary_probabilities(
    r: npt.NDArray, max_iter: int = 100
) -> tuple[npt.NDArray, npt.NDArray]:
    """libsvm's ``multiclass_probability`` for two classes, vectorised over
    the samples. ``r`` is the pairwise probability of the first class; the
    iteration stops for each sample at libsvm's tolerance, so the results
    agree with libsvm to rounding."""
    q00, q11, q01 = (1 - r) ** 2, r**2, -(1 - r) * r
    p0, p1 = np.full_like(r, 0.5), np.full_like(r, 0.5)
    active = np.ones(r.shape, dtype=bool)
    tolerance = 0.005 / 2
    for _ in range(max_iter):
        qp0 = q00 * p0 + q01 * p1
        qp1 = q01 * p0 + q11 * p1
        pqp = p0 * qp0 + p1 * qp1
        active &= np.maximum(np.abs(qp0 - pqp), np.abs(qp1 - pqp)) >= tolerance
        if not active.any():
            break
        for t in (0, 1):
            q_tt, qp_t = (q00, qp0) if t == 0 else (q11, qp1)
            diff = np.where(active, (pqp - qp_t) / q_tt, 0.0)
            pqp = (pqp + diff * (diff * q_tt + 2 * qp_t)) / (1 + diff) ** 2
            q_t0, q_t1 = (q00, q01) if t == 0 else (q01, q11)
            qp0 = (qp0 + diff * q_t0) / (1 + diff)
            qp1 = (qp1 + diff * q_t1) / (1 + diff)
            if t == 0:
                p0 = p0 + diff
            else:
                p1 = p1 + diff
            p0, p1 = p0 / (1 + diff), p1 / (1 + diff)
    return p0, p1


@functools.cache
def load_circulation_model() -> tuple[SupportVectorClassifier, RobustScaler]:
    """Load the (pre-trained) classification model and the corresponding
    scaler used by :func:`.predict_circulation` from the package assets.

    Returns
    -------
    A tuple ``(model, scaler)``.
    """
    asset = importlib.resources.files("vitabel") / "assets" / "circulation_model.npz"
    with asset.open("rb") as fd, np.load(fd) as parameters:
        model = SupportVectorClassifier(
            support_vectors=parameters["support_vectors"],
            dual_coef=parameters["dual_coef"],
            intercept=parameters["intercept"],
            gamma=parameters["gamma"],
            prob_a=parameters["prob_a"],
            prob_b=parameters["prob_b"],
            classes=parameters["classes"],
        )
        scaler = RobustScaler(
            center=parameters["scaler_center"], scale=parameters["scaler_scale"]
        )
    return model, scaler
