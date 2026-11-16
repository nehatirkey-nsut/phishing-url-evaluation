"""phishgap: leakage-free, imbalance-aware evaluation of fine-tuned phishing URL classifiers.

Modules
-------
config       experiment settings and the values reported by Abdul Samad et al. (2023)
data         dataset loaders (UCI Phishing Websites, Mendeley, PhiUSIIL) and a synthetic mode
features     URL-only lexical feature extractor (works on raw URL strings)
resampling   SMOTE implemented as a classifier wrapper, so it only ever sees training data
models       the eight classifiers and their hyper-parameter grids
experiments  E1 (protocol comparison), E2 (class-imbalance stress test), E3 (deployment cost)
plots        figure helpers
"""

__version__ = "1.0.0"
