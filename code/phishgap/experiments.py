"""The three experiments.

E1  Protocol comparison (UCI, and Mendeley if present).
    Protocol A follows the reference study: scale and SMOTE-balance the whole
    dataset, then grid-search with 10-fold CV and report the best CV score.
    Protocol B is leakage-free nested CV: scaling, SMOTE and feature selection
    are fitted on training folds only, tuning uses an inner CV, and performance
    is measured on outer folds that were never resampled.
E2  Class-imbalance stress test (PhiUSIIL).  Train once, then test on sets with
    r = 1, 10, 100 legitimate URLs per phishing URL (repeated random draws).
    Observed precision is compared with the analytic prediction
    P(r) = TPR / (TPR + r * FPR) computed from the r = 1 draws.
E3  Deployment cost (PhiUSIIL).  Training time and per-URL prediction latency
    for the dataset's provided features versus URL-only lexical features.
"""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, make_scorer,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import features
from .config import E2_REPEATS, INNER_FOLDS, OUTER_FOLDS, RANDOM_STATE, RATIOS, REPORTED_ACCURACY
from .models import leak_free_grid, leak_free_pipeline, plain_grid, plain_pipeline
from .resampling import smote

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "mcc": make_scorer(matthews_corrcoef),
    "pr_auc": "average_precision",
}


def environment(out_dir):
    """Record software and hardware details for the paper's Experimental Setup section."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except ImportError:
        info["xgboost"] = "not installed"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "environment.json").write_text(json.dumps(info, indent=2))
    return info


# ------------------------------------------------------------------------- E1
def run_e1(datasets, models, fast=False, log=print):
    outer = StratifiedKFold(3 if fast else OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner = StratifiedKFold(2 if fast else INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for X, y, info in datasets:
        n_features = X.shape[1]
        # Protocol A: preprocessing on the full dataset, then tuning on the same CV that is reported.
        X_a = MinMaxScaler().fit_transform(X)
        X_a, y_a = smote(X_a, y, random_state=RANDOM_STATE)
        for name in models:
            start = time.perf_counter()
            log(f"[E1] {info['name']} | {name} | protocol A")
            gs_a = GridSearchCV(plain_pipeline(name), plain_grid(name, n_features, fast),
                                scoring=SCORING, refit="accuracy", cv=outer, n_jobs=-1)
            gs_a.fit(X_a, y_a)
            a = {m: gs_a.cv_results_[f"mean_test_{m}"][gs_a.best_index_] for m in SCORING}

            log(f"[E1] {info['name']} | {name} | protocol B (nested)")
            gs_b = GridSearchCV(leak_free_pipeline(name), leak_free_grid(name, n_features, fast),
                                scoring="accuracy", cv=inner, n_jobs=-1)
            cv_b = cross_validate(gs_b, X.to_numpy(), y, cv=outer, scoring=SCORING,
                                  return_estimator=True, n_jobs=1)
            row = {
                "dataset": info["key"], "model": name,
                "reported_accuracy": REPORTED_ACCURACY.get(info["key"], {}).get(name, np.nan),
                "A_accuracy": 100 * a["accuracy"], "A_f1": 100 * a["f1"], "A_mcc": a["mcc"],
                "A_best_params": json.dumps(gs_a.best_params_, default=str),
            }
            for m in SCORING:
                scale = 1 if m == "mcc" else 100
                row[f"B_{m}"] = scale * cv_b[f"test_{m}"].mean()
                row[f"B_{m}_sd"] = scale * cv_b[f"test_{m}"].std()
            row["B_best_params_fold1"] = json.dumps(cv_b["estimator"][0].best_params_, default=str)
            row["minutes"] = (time.perf_counter() - start) / 60
            log(f"      A acc {row['A_accuracy']:.2f}  B acc {row['B_accuracy']:.2f}  "
                f"B MCC {row['B_mcc']:.3f}  ({row['minutes']:.1f} min)")
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- E2 + E3
def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _test_draws(y, test_idx, ratios, repeats, rng):
    """Index sets with exactly r legitimate URLs per phishing URL (same draws for every model)."""
    pos = test_idx[y[test_idx] == 1]
    neg = test_idx[y[test_idx] == 0]
    draws = {}
    for r in ratios:
        n_neg = min(len(neg), r * len(pos))
        n_pos = max(n_neg // r, 1)
        draws[r] = [np.concatenate([rng.choice(pos, n_pos, replace=False),
                                    rng.choice(neg, n_neg, replace=False)]) for _ in range(repeats)]
    return draws


def run_e2_e3(X_provided, y, urls, models, n_train=20000, fast=False, log=print):
    rng = np.random.default_rng(RANDOM_STATE)
    log(f"[E2] extracting lexical features from {len(urls):,} URLs")
    start = time.perf_counter()
    X_lexical = features.extract(urls)
    lexical_us = (time.perf_counter() - start) / len(urls) * 1e6
    feature_sets = {"provided": X_provided.to_numpy(), "lexical": X_lexical.to_numpy()}

    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(all_idx, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    if n_train and len(train_idx) > n_train:
        train_idx, _ = train_test_split(train_idx, train_size=n_train, stratify=y[train_idx],
                                        random_state=RANDOM_STATE)
    repeats = 3 if fast else E2_REPEATS
    draws = _test_draws(y, test_idx, RATIOS, repeats, rng)
    position = np.full(len(y), -1)
    position[test_idx] = np.arange(len(test_idx))
    latency_rows = test_idx[: min(len(test_idx), 10000)]

    imbalance, cost = [], []
    for fs_name, X in feature_sets.items():
        for name in models:
            log(f"[E2] {fs_name} features | {name}")
            model = leak_free_pipeline(name)  # default hyper-parameters, SMOTE on training rows only
            t0 = time.perf_counter()
            model.fit(X[train_idx], y[train_idx])
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            model.predict(X[latency_rows])
            latency_us = (time.perf_counter() - t0) / len(latency_rows) * 1e6

            pred = model.predict(X[test_idx])
            score = _scores(model, X[test_idx])
            per_ratio = {}
            for r in RATIOS:
                metrics = {k: [] for k in ("accuracy", "precision", "recall", "f1", "mcc", "pr_auc", "fpr")}
                for ids in draws[r]:
                    yt, yp, ys = y[ids], pred[position[ids]], score[position[ids]]
                    metrics["accuracy"].append(accuracy_score(yt, yp))
                    metrics["precision"].append(precision_score(yt, yp, zero_division=0))
                    metrics["recall"].append(recall_score(yt, yp))
                    metrics["f1"].append(f1_score(yt, yp, zero_division=0))
                    metrics["mcc"].append(matthews_corrcoef(yt, yp))
                    metrics["pr_auc"].append(average_precision_score(yt, ys))
                    metrics["fpr"].append(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
                per_ratio[r] = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
            tpr1, fpr1 = per_ratio[1]["recall"][0], per_ratio[1]["fpr"][0]
            for r in RATIOS:
                row = {"features": fs_name, "model": name, "ratio": r}
                for k, (mean, sd) in per_ratio[r].items():
                    scale = 1 if k == "mcc" else 100
                    row[k], row[f"{k}_sd"] = scale * mean, scale * sd
                row["precision_predicted"] = 100 * tpr1 / (tpr1 + r * fpr1) if tpr1 + r * fpr1 > 0 else np.nan
                imbalance.append(row)
            cost.append({
                "features": fs_name, "model": name, "n_features": X.shape[1], "n_train": len(train_idx),
                "train_seconds": train_s, "predict_us_per_url": latency_us,
                "extract_us_per_url": lexical_us if fs_name == "lexical" else np.nan,
                "mcc_r1": per_ratio[1]["mcc"][0], "mcc_r100": per_ratio[RATIOS[-1]]["mcc"][0],
                "precision_r100": 100 * per_ratio[RATIOS[-1]]["precision"][0],
            })
            log(f"      precision r=1 {100 * per_ratio[1]['precision'][0]:.1f}%  "
                f"r={RATIOS[-1]} {100 * per_ratio[RATIOS[-1]]['precision'][0]:.1f}%  "
                f"train {train_s:.1f}s  predict {latency_us:.1f} us/URL")
    return pd.DataFrame(imbalance), pd.DataFrame(cost)


# ---------------------------------------------------------------- LaTeX tables
def latex_table(df, columns, header, caption, label, colspec):
    """Booktabs table in the Springer template style, ready to paste into sn-article.tex."""
    lines = [r"\begin{table}[htbp]", rf"\caption{{{caption}}}\label{{{label}}}%",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[c]) for c in columns) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_tables(out_dir, e1=None, e2=None, e3=None):
    tdir = Path(out_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    fmt = lambda v, d=2: "--" if pd.isna(v) else f"{v:.{d}f}"
    if e1 is not None and len(e1):
        names = {"uci": "UCI", "mendeley": "Mendeley"}
        for key, part in e1.groupby("dataset", sort=False):
            t = pd.DataFrame({
                "model": part["model"],
                "rep": part["reported_accuracy"].map(fmt),
                "a": part["A_accuracy"].map(fmt),
                "b": [f"{fmt(m)} $\\pm$ {fmt(s)}" for m, s in zip(part["B_accuracy"], part["B_accuracy_sd"])],
                "gap": (part["A_accuracy"] - part["B_accuracy"]).map(lambda v: f"${v:+.2f}$"),
                "mcc": part["B_mcc"].map(lambda v: fmt(v, 3)),
            })
            (tdir / f"table_e1_{key}.tex").write_text(latex_table(
                t, ["model", "rep", "a", "b", "gap", "mcc"],
                ["Model", "Reported", "Protocol A", "Protocol B", "A $-$ B", "MCC (B)"],
                f"E1 results on the {names.get(key, key)} dataset: accuracy (\\%) reported by the reference "
                f"study, obtained with its protocol (A) and with leakage-free nested cross-validation "
                f"(B, mean $\\pm$ s.d.\\ over five outer folds), their difference, and the MCC under protocol B",
                f"tab:e1-{key}", "@{}lcccrc@{}"))
    if e2 is not None and len(e2):
        for fs, part in e2.groupby("features", sort=False):
            t = part.copy()
            t["model_shown"] = [m if i == 0 else "" for i, m in
                                zip(t.groupby("model", sort=False).cumcount(), t["model"])]
            t["prec"] = [f"{fmt(m, 1)} $\\pm$ {fmt(s, 1)}" for m, s in zip(t["precision"], t["precision_sd"])]
            t["pred"] = t["precision_predicted"].map(lambda v: fmt(v, 1))
            t["rec"] = t["recall"].map(lambda v: fmt(v, 1))
            t["mccs"] = t["mcc"].map(lambda v: fmt(v, 3))
            t["prauc"] = t["pr_auc"].map(lambda v: fmt(v, 1))
            label = "provided features" if fs == "provided" else "URL-only lexical features"
            (tdir / f"table_e2_{fs}.tex").write_text(latex_table(
                t, ["model_shown", "ratio", "prec", "pred", "rec", "mccs", "prauc"],
                ["Model", "$r$", "Precision", "Predicted", "Recall", "MCC", "PR-AUC"],
                f"E2 results on PhiUSIIL with {label}: observed precision (\\%, mean $\\pm$ s.d.\\ over "
                f"repeated draws), precision predicted from the $r=1$ error rates, recall (\\%), MCC and "
                f"PR-AUC (\\%) at each ratio $r$ of legitimate to phishing URLs",
                f"tab:e2-{fs}", "@{}lrccccc@{}"))
    if e3 is not None and len(e3):
        t = e3.copy()
        t["fs"] = t["features"].map({"provided": "Prov.", "lexical": "Lex."})
        t["train"] = t["train_seconds"].map(lambda v: fmt(v, 1))
        t["pred"] = t["predict_us_per_url"].map(lambda v: fmt(v, 1))
        t["ext"] = t["extract_us_per_url"].map(lambda v: fmt(v, 1))
        t["m1"] = t["mcc_r1"].map(lambda v: fmt(v, 3))
        t["m100"] = t["mcc_r100"].map(lambda v: fmt(v, 3))
        n_prov = int(t.loc[t["features"] == "provided", "n_features"].max()) if (t["features"] == "provided").any() else 0
        n_lex = int(t.loc[t["features"] == "lexical", "n_features"].max()) if (t["features"] == "lexical").any() else 0
        (tdir / "table_e3.tex").write_text(latex_table(
            t, ["fs", "model", "train", "pred", "ext", "m1", "m100"],
            ["Feat.", "Model", "Train (s)", "Pred.\\ ($\\mu$s)", "Extr.\\ ($\\mu$s)", "MCC$_{r=1}$", "MCC$_{r=100}$"],
            f"E3 results on PhiUSIIL: training time, prediction and lexical-feature extraction time per URL, "
            f"and MCC at $r=1$ and $r=100$ for the provided features (Prov., {n_prov} features) and the "
            f"URL-only lexical features (Lex., {n_lex} features)",
            "tab:e3", "@{}llrrrrr@{}"))
