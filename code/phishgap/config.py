"""Experiment settings.

Every number in REPORTED_ACCURACY is copied from Abdul Samad et al. (2023),
Electronics 12(7):1642, https://doi.org/10.3390/electronics12071642
(Table 16 for the UCI dataset and Table 17 for the Mendeley dataset, column
"Feature Selection", i.e. after all three tuning factors were applied).
"""

RANDOM_STATE = 42

# Classifier short names, in the order used by the reference study.
CLASSIFIERS = ["LogR", "SVM", "NB", "KNN", "DT", "RF", "GB", "XGB"]

# Legitimate URLs per phishing URL in the E2 stress test.
RATIOS = (1, 10, 100)

# Repeated random test-set draws per ratio in E2 (mean and s.d. are reported).
E2_REPEATS = 10

# Cross-validation folds for E1 (outer = performance estimate, inner = tuning).
OUTER_FOLDS = 5
INNER_FOLDS = 3

# Accuracy (%) reported by the reference study after all tuning factors.
# The reference study used Bernoulli NB on UCI and Gaussian NB on Mendeley;
# this kit uses Gaussian NB on both (stated in the README and the paper).
REPORTED_ACCURACY = {
    "uci": {"LogR": 92.33, "SVM": 97.04, "NB": 91.27, "KNN": 97.04,
            "DT": 96.69, "RF": 97.44, "GB": 97.47, "XGB": 97.26},
    "mendeley": {"LogR": 93.88, "SVM": 96.79, "NB": 86.99, "KNN": 95.41,
                 "DT": 96.17, "RF": 97.79, "GB": 98.27, "XGB": 98.21},
}
