"""URL-only lexical features.

These features are computed from the URL string alone: no page download, no
DNS query, no search-engine or ranking lookup.  That makes them fast and safe
to compute at click time, which is the deployment question tested in E3.
"""
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_HEX_IP = re.compile(r"^0x[0-9a-f]+$", re.I)
_SPECIAL = set("@?=&%~+!*,;$")
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
              "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy"}
SUSPICIOUS_WORDS = ("login", "signin", "verify", "account", "update", "secure",
                    "bank", "confirm", "password", "webscr", "billing", "wallet")

# Name and one-line description of every feature (used for the paper's table).
FEATURE_DESCRIPTIONS = {
    "url_length": "Number of characters in the URL",
    "host_length": "Number of characters in the host name",
    "path_length": "Number of characters in the path",
    "query_length": "Number of characters in the query string",
    "num_dots": "Count of '.' in the URL",
    "num_hyphens": "Count of '-' in the URL",
    "num_underscores": "Count of '_' in the URL",
    "num_slashes": "Count of '/' in the URL",
    "num_digits": "Count of digits in the URL",
    "digit_ratio": "Digits divided by URL length",
    "num_special": "Count of special characters such as @ ? = & % ~ +",
    "has_at": "1 if the URL contains '@'",
    "has_ip_host": "1 if the host is an IPv4 or hexadecimal address",
    "is_https": "1 if the scheme is https",
    "num_subdomains": "Number of labels in the host minus two",
    "has_port": "1 if an explicit port is given",
    "path_depth": "Number of non-empty path segments",
    "double_slash_in_path": "1 if '//' appears after the scheme",
    "num_params": "Number of query parameters",
    "tld_length": "Length of the top-level domain",
    "is_shortener": "1 if the host is a known URL-shortening service",
    "suspicious_words": "Count of words such as login, verify, account",
    "entropy": "Shannon entropy of the URL characters (bits)",
}
FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)


def _entropy(text):
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def url_features(url):
    """Return the lexical feature vector (list of floats) for one URL string."""
    url = str(url).strip()
    parsed = urlsplit(url if "://" in url else "http://" + url)
    host = (parsed.hostname or "").lower()
    path, query = parsed.path or "", parsed.query or ""
    lower = url.lower()
    labels = [p for p in host.split(".") if p]
    try:
        has_port = int(parsed.port is not None)
    except ValueError:  # malformed port such as "host:abc"
        has_port = 1
    n = max(len(url), 1)
    digits = sum(ch.isdigit() for ch in url)
    return [
        len(url), len(host), len(path), len(query),
        url.count("."), url.count("-"), url.count("_"), url.count("/"),
        digits, digits / n,
        sum(ch in _SPECIAL for ch in url),
        int("@" in url),
        int(bool(_IPV4.match(host) or _HEX_IP.match(host))),
        int(parsed.scheme.lower() == "https"),
        max(len(labels) - 2, 0),
        has_port,
        len([p for p in path.split("/") if p]),
        int("//" in url.split("://", 1)[-1]),
        len([p for p in query.split("&") if p]),
        len(labels[-1]) if labels else 0,
        int(host.removeprefix("www.") in SHORTENERS),
        sum(lower.count(w) for w in SUSPICIOUS_WORDS),
        _entropy(url),
    ]


def extract(urls):
    """Lexical feature matrix (DataFrame) for an iterable of URL strings."""
    rows = [url_features(u) for u in urls]
    return pd.DataFrame(np.asarray(rows, dtype=float), columns=FEATURE_NAMES)
