"""Dataset loaders.  Every loader returns (X, y, info) with y = 1 for phishing.

UCI Phishing Websites (id 327)  https://doi.org/10.24432/C51W2X
    11,055 sites, 30 features coded -1/0/1, target 'result' (-1 = phishing).
PhiUSIIL Phishing URL (id 967)  https://archive.ics.uci.edu/dataset/967
    235,795 URLs, 54 features plus the raw URL, target 'label' (0 = phishing).
Mendeley "Phishing Dataset for Machine Learning"  https://doi.org/10.17632/h3cgnj8hft.1
    10,000 pages, 48 features, target 'CLASS_LABEL' (1 = phishing).
    Mendeley needs a manual download: put the .arff file (or a CSV export) in data/mendeley/.

Downloaded files are cached in data/ so each dataset is fetched once.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _fetch_uci(uci_id, cache_name):
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / cache_name
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit("Install ucimlrepo first: pip install -r requirements.txt") from exc
    ds = fetch_ucirepo(id=uci_id)
    df = ds.data.original if ds.data.original is not None else pd.concat(
        [ds.data.features, ds.data.targets], axis=1)
    df.to_csv(cache, index=False)
    return df


def load_uci():
    df = _fetch_uci(327, "uci_phishing_websites.csv")
    target = "result" if "result" in df.columns else "Result"
    y = (df[target] == -1).astype(int).to_numpy()
    X = df.drop(columns=[target]).astype(float)
    return X, y, {"name": "UCI Phishing Websites", "key": "uci"}


def load_mendeley():
    """Reads the Mendeley file (ARFF as published, or a CSV export) from data/mendeley/."""
    folder = DATA_DIR / "mendeley"
    files = sorted(folder.glob("*.arff")) + sorted(folder.glob("*.csv"))
    if not files:
        return None
    if files[0].suffix == ".arff":
        from scipy.io import arff
        records, _ = arff.loadarff(files[0])
        df = pd.DataFrame(records)
        df["CLASS_LABEL"] = df["CLASS_LABEL"].map(lambda v: int(v.decode() if isinstance(v, bytes) else v))
    else:
        df = pd.read_csv(files[0])
    y = df["CLASS_LABEL"].astype(int).to_numpy()
    X = df.drop(columns=[c for c in ("CLASS_LABEL", "id") if c in df.columns]).astype(float)
    return X, y, {"name": "Mendeley Phishing", "key": "mendeley"}


def load_phiusiil():
    """Returns (X_provided, y, urls, info): the dataset's numeric features and the raw URLs."""
    df = _fetch_uci(967, "phiusiil.csv")
    y = (df["label"] == 0).astype(int).to_numpy()
    urls = df["URL"].astype(str).to_numpy()
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).astype(float)
    return X, y, urls, {"name": "PhiUSIIL", "key": "phiusiil"}


# ---------------------------------------------------------------- synthetic mode
def synthetic_tabular(n=3000, n_features=30, phishing_share=0.44, seed=RANDOM_STATE):
    """Small UCI-like dataset for smoke tests only (never report these numbers)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=12,
                               weights=[1 - phishing_share], flip_y=0.03, random_state=seed)
    X = np.round(np.clip(X / 2, -1, 1))  # ternary, like the UCI coding
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y, {
        "name": "Synthetic (smoke test)", "key": "uci"}


def synthetic_urls(n=6000, phishing_share=0.43, seed=RANDOM_STATE):
    """Fake URLs with a learnable but noisy signal, for smoke tests only."""
    rng = np.random.default_rng(seed)
    words = ["home", "about", "news", "shop", "blog", "docs", "help", "contact"]
    bad = ["login", "verify", "account", "secure-update", "signin", "confirm"]
    urls, y = [], []
    for i in range(n):
        phish = rng.random() < phishing_share
        noisy = rng.random() < 0.15
        if phish != noisy:
            host = rng.choice([f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}",
                               f"secure-{rng.integers(100000)}.account-check.{rng.choice(['xyz', 'top', 'info'])}"])
            url = f"http://{host}/{rng.choice(bad)}/{rng.choice(bad)}.php?id={rng.integers(1000000)}&s={rng.integers(10000)}"
        else:
            url = f"https://www.site{rng.integers(10000)}.{rng.choice(['com', 'org', 'in'])}/{rng.choice(words)}"
        urls.append(url)
        y.append(int(phish))
    y = np.array(y)
    from .features import extract
    X_provided = extract(urls) + rng.normal(0, 0.5, (n, 23))
    X_provided.columns = [f"p{i}" for i in range(X_provided.shape[1])]
    return X_provided, y, np.array(urls), {"name": "Synthetic URLs (smoke test)", "key": "phiusiil"}
