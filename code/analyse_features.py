"""Single-feature analysis of PhiUSIIL (supports the discussion of E2 and E3).

For every feature, a one-split decision tree (a single threshold) is evaluated with stratified
5-fold cross-validation on a stratified sample of 50,000 URLs.  A feature that separates the
classes almost perfectly on its own indicates that the dataset is far easier than real traffic.
Also reports, for the full dataset, how the raw URLs of each class differ in form (path, trailing slash, query, HTTPS).

Writes results/feature_analysis.csv and results/tables/table_features.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from phishgap import data, features
from phishgap.config import RANDOM_STATE

OUT = Path(__file__).resolve().parent / "results"


def main():
    X_prov, y, urls, _ = data.load_phiusiil()
    idx, _ = train_test_split(np.arange(len(y)), train_size=50000, stratify=y, random_state=RANDOM_STATE)
    X_prov, y_s, urls_s = X_prov.iloc[idx].reset_index(drop=True), y[idx], urls[idx]
    X_lex = features.extract(urls_s)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for set_name, X in (("provided", X_prov), ("lexical", X_lex)):
        for col in X.columns:
            acc = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
                                  X[[col]].to_numpy(), y_s, cv=cv, scoring="accuracy").mean()
            rows.append({"features": set_name, "feature": col, "single_threshold_accuracy": 100 * acc})
    df = pd.DataFrame(rows).sort_values("single_threshold_accuracy", ascending=False)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "feature_analysis.csv", index=False)

    from urllib.parse import urlsplit
    parts = [urlsplit(u if "://" in u else "http://" + u) for u in urls]
    form = pd.DataFrame({"empty_path": [q.path == "" for q in parts], "slash_only": [q.path == "/" for q in parts],
                         "has_query": [q.query != "" for q in parts], "https": [q.scheme == "https" for q in parts]})
    share = {c: (100 * form.loc[y == 0, c].mean(), 100 * form.loc[y == 1, c].mean()) for c in form}
    for c, (leg, phi) in share.items():
        print(f"{c:11s} legitimate {leg:5.1f}%   phishing {phi:5.1f}%")
    pd.DataFrame(share, index=["legitimate_%", "phishing_%"]).to_csv(OUT / "url_form.csv")
    for s in ("provided", "lexical"):
        print(df[df["features"] == s].head(6).round(2).to_string(index=False))

    top = pd.concat([df[df["features"] == "provided"].head(5), df[df["features"] == "lexical"].head(3)])
    lines = [r"\begin{table}[htbp]",
             r"\caption{Accuracy of a single-threshold classifier (one-split decision tree, 5-fold "
             r"cross-validation on 50,000 PhiUSIIL URLs) for the strongest individual features of each set}"
             r"\label{tab:features}%",
             r"\begin{tabular}{@{}llr@{}}", r"\toprule", r"Feature set & Feature & Accuracy (\%) \\", r"\midrule"]
    for _, r in top.iterrows():
        name = r["feature"].replace("_", r"\_")
        lines.append(f"{'Provided' if r['features'] == 'provided' else 'Lexical'} & \\texttt{{{name}}} & "
                     f"{r['single_threshold_accuracy']:.2f} \\\\")
    lines += [r"\botrule", r"\end{tabular}",
              rf"\footnotetext{{In the full dataset, {share['empty_path'][0]:.1f}\% of legitimate URLs have no path "
              rf"(not even a trailing slash), {share['has_query'][0]:.1f}\% have a query string and "
              rf"{share['https'][0]:.1f}\% use HTTPS; for phishing URLs the shares are {share['empty_path'][1]:.1f}\%, "
              rf"{share['has_query'][1]:.1f}\% and {share['https'][1]:.1f}\%, and {share['slash_only'][1]:.1f}\% "
              rf"end in a single slash.}}", r"\end{table}"]
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "tables" / "table_features.tex").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "feature_analysis.csv")


if __name__ == "__main__":
    main()
