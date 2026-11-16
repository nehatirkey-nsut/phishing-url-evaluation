"""The eight classifiers of the reference study and their tuning grids.

Grids are deliberately small so that nested cross-validation finishes on a
laptop; they cover the ranges reported in Table 10 of Abdul Samad et al. (2023).
"""
import warnings

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .resampling import SMOTEClassifier

try:  # XGBoost is optional so the kit still runs where it is not installed.
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def available(names):
    """Drop XGB (with a warning) when xgboost is not installed."""
    if XGBClassifier is None and "XGB" in names:
        warnings.warn("xgboost is not installed; skipping XGB. Run: pip install xgboost")
        return [n for n in names if n != "XGB"]
    return list(names)


def make_classifier(name):
    rs = RANDOM_STATE
    if name == "LogR":
        return LogisticRegression(solver="liblinear", max_iter=2000, random_state=rs)
    if name == "SVM":
        return SVC(kernel="rbf", random_state=rs)
    if name == "NB":
        return GaussianNB()
    if name == "KNN":
        return KNeighborsClassifier(n_jobs=1)
    if name == "DT":
        return DecisionTreeClassifier(random_state=rs)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=rs)
    if name == "GB":
        return GradientBoostingClassifier(random_state=rs)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss",
                             n_jobs=1, random_state=rs)
    raise ValueError(f"unknown classifier {name}")


# Hyper-parameter grids (keys are the classifier's own parameter names).
GRIDS = {
    "LogR": {"C": [0.01, 0.1, 1, 10, 100]},
    "SVM": {"C": [1, 10], "gamma": ["scale", 0.1]},
    "NB": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "metric": ["minkowski", "manhattan"]},
    "DT": {"criterion": ["gini", "entropy"], "max_depth": [None, 20, 40]},
    "RF": {"max_features": ["sqrt", "log2"], "max_depth": [None, 40]},
    "GB": {"learning_rate": [0.1, 0.5], "n_estimators": [100, 300]},
    "XGB": {"max_depth": [6, 12], "learning_rate": [0.1, 0.3]},
}


def k_options(n_features, fast=False):
    """Feature counts tried by SelectKBest (always includes 'all')."""
    if fast:
        return ["all"]
    return sorted({k for k in (10, 20) if k < n_features}) + ["all"]


def leak_free_pipeline(name, smote_enabled=True):
    """scale -> [SMOTE (training rows only) -> drop constant columns -> SelectKBest -> classifier]."""
    inner = Pipeline([("drop_constant", VarianceThreshold()),
                      ("select", SelectKBest(f_classif, k="all")),
                      ("clf", make_classifier(name))])
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("model", SMOTEClassifier(inner, enabled=smote_enabled, random_state=RANDOM_STATE)),
    ])


def leak_free_grid(name, n_features, fast=False):
    grid = {f"model__estimator__clf__{k}": v for k, v in GRIDS[name].items()}
    grid["model__estimator__select__k"] = k_options(n_features, fast)
    return grid


def plain_pipeline(name):
    """SelectKBest -> classifier, for data that was already scaled and balanced."""
    return Pipeline([("drop_constant", VarianceThreshold()),
                     ("select", SelectKBest(f_classif, k="all")),
                     ("clf", make_classifier(name))])


def plain_grid(name, n_features, fast=False):
    grid = {f"clf__{k}": v for k, v in GRIDS[name].items()}
    grid["select__k"] = k_options(n_features, fast)
    return grid
