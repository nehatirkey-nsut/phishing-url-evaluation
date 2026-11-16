"""SMOTE, and a wrapper that applies it only inside `fit`.

Why a wrapper?  scikit-learn pipelines cannot hold a step that changes the
number of rows.  If SMOTE is applied to the whole dataset before splitting
(as the reference study describes), synthetic points built from a test row's
neighbours end up in the training data, and the test score becomes optimistic.
`SMOTEClassifier` oversamples only the data passed to `fit`, so inside any
cross-validation loop it only ever sees training folds.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.metaestimators import available_if


def smote(X, y, k_neighbors=5, random_state=None):
    """Synthetic Minority Over-sampling TEchnique (Chawla et al., 2002).

    For each synthetic sample: pick a random minority row, pick one of its k
    nearest minority neighbours, and place a new point at a random position on
    the line segment between them.  Returns a balanced (X, y).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        return X, y
    minority = classes[np.argmin(counts)]
    n_new = int(counts.max() - counts.min())
    X_min = X[y == minority]
    k = min(k_neighbors, len(X_min) - 1)
    if k < 1:
        return X, y
    rng = np.random.default_rng(random_state)
    _, neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min)
    base = rng.integers(0, len(X_min), n_new)
    partner = neighbours[base, rng.integers(1, k + 1, n_new)]  # column 0 is the row itself
    gap = rng.random((n_new, 1))
    X_new = X_min[base] + gap * (X_min[partner] - X_min[base])
    return np.vstack([X, X_new]), np.concatenate([y, np.full(n_new, minority)])


def _has(method):
    return lambda self: hasattr(self.estimator, method)


class SMOTEClassifier(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on SMOTE-balanced training data; predict as usual.

    Parameters
    ----------
    estimator : scikit-learn classifier or pipeline
    k_neighbors : int, neighbours used by SMOTE
    enabled : bool, set False to switch resampling off (useful in grids)
    random_state : int or None
    """

    def __init__(self, estimator=None, k_neighbors=5, enabled=True, random_state=None):
        self.estimator = estimator
        self.k_neighbors = k_neighbors
        self.enabled = enabled
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.enabled:
            X, y = smote(X, y, self.k_neighbors, self.random_state)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))

    @available_if(_has("predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

    @available_if(_has("decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(np.asarray(X, dtype=float))
