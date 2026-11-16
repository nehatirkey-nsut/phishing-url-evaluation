# Leakage-free, imbalance-aware evaluation of fine-tuned phishing URL classifiers

Code, results and figures for the paper *Beyond Benchmark Accuracy: Leakage-Free and
Imbalance-Aware Evaluation of Fine-Tuned Phishing URL Classifiers*. The project re-examines a
recent SCIE-indexed study that reported 97–98% accuracy for eight fine-tuned machine learning
classifiers on phishing benchmarks
([Abdul Samad et al., 2023](https://doi.org/10.3390/electronics12071642)) and tests three gaps
in how such results are obtained and reported.

| Experiment | Research question | Data |
|---|---|---|
| **E1** Protocol comparison | Does balancing and scaling the whole dataset before cross-validation (as in the reference study) inflate accuracy compared with leakage-free nested cross-validation? | UCI Phishing Websites, Mendeley |
| **E2** Imbalance stress test | How do precision, recall, MCC and PR-AUC change when legitimate URLs outnumber phishing URLs 10:1 and 100:1, and does an analytic formula predict it? | PhiUSIIL (2024) |
| **E3** Deployment cost | What are the training and per-URL prediction costs, and what is lost with URL-only lexical features instead of the dataset's page and external features? | PhiUSIIL (2024) |

## Main results

* **E1.** On UCI, the reference protocol scored higher than leakage-free nested CV for all
  eight classifiers (by 0.25–0.65 points, 2.73 for naive Bayes). On the already balanced
  Mendeley data the two agreed within 0.26 points. Best leakage-free accuracy: XGBoost,
  97.11% (UCI) and 98.87% (Mendeley).
* **E2.** With URL-only features, precision fell from ≥ 99.2% at 1:1 to 54.9–99.2% at 1:100,
  within 0.47 points of the prediction `TPR / (TPR + r·FPR)`, while accuracy rose.
* **E3.** All models predict in under 70 µs per URL; the 23 lexical features take 11.8 µs per
  URL to compute. A single PhiUSIIL feature (`URLSimilarityIndex`) separates the classes with
  99.66% accuracy, and every legitimate PhiUSIIL URL is a bare HTTPS home page.

Raw numbers are in `results/*.csv`; the paper source is in `../paper/`.

## Repository layout

```
run_experiments.py      runs E1, E2, E3; writes CSVs, LaTeX tables and figures
analyse_features.py     single-feature analysis and URL-form statistics for PhiUSIIL
make_static_figures.py  motivation figure and architecture diagram (no data needed)
phishgap/
  config.py             settings and the reference study's reported accuracies
  data.py               dataset loaders (+ synthetic data for smoke tests)
  features.py           23 URL-only lexical features
  resampling.py         SMOTE, applied inside fit() only (prevents leakage)
  models.py             the 8 classifiers and their hyper-parameter grids
  experiments.py        E1, E2, E3 and the LaTeX table writer
  plots.py              figures
results/                CSV results, run.log, environment.json, tables/*.tex
```

## Setup

Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Datasets.** UCI Phishing Websites and PhiUSIIL download automatically on the first run
(through `ucimlrepo`) and are cached in `data/`. The Mendeley dataset must be downloaded by
hand from <https://doi.org/10.17632/h3cgnj8hft.1>: put `Phishing_Legitimate_full.arff` (or a
CSV export) in `data/mendeley/`. Without it, E1 runs on UCI only.

## Running

```bash
python run_experiments.py --synthetic --fast   # smoke test on generated data (~30 s); numbers are meaningless
python run_experiments.py                      # full run used for the paper (~6 min on an Apple M2)
python run_experiments.py --only e1            # E1 only (or --only e2 for E2 and E3)
python run_experiments.py --tables-only        # rebuild tables and figures from saved CSVs
python analyse_features.py                     # single-feature analysis (Table 10 of the paper)
python make_static_figures.py                  # Figs. 1 and 2 of the paper
```

`--fast` uses smaller grids and fewer folds for a quicker check. Progress and timings are
written to `results/run.log`.

## Outputs

| File | Paper element |
|---|---|
| `results/environment.json` | Experimental setup (hardware and library versions) |
| `results/tables/table_e1_uci.tex`, `table_e1_mendeley.tex` | Tables 5 and 6 (E1) |
| `results/tables/table_e2_provided.tex`, `table_e2_lexical.tex` | Tables 7 and 8 (E2) |
| `results/tables/table_e3.tex` | Table 9 (E3) |
| `results/tables/table_features.tex` | Table 10 (feature analysis) |
| `../paper/fig_e1_protocols.pdf`, `../paper/fig_e2_imbalance.pdf` | Figs. 3 and 4 |
| `../paper/fig_motivation.pdf`, `../paper/fig_architecture.pdf` | Figs. 1 and 2 |

## Method notes

* **Leakage-free pipeline (protocol B).** Min-max scaling, SMOTE, constant-column removal,
  SelectKBest and the classifier are fitted on training folds only. SMOTE is implemented in
  `phishgap/resampling.py` as a classifier wrapper so that it can sit inside scikit-learn
  cross-validation; hyper-parameters are chosen by an inner 3-fold CV and performance is
  measured on 5 outer folds that are never resampled.
* **Reference protocol (protocol A).** Scale and SMOTE-balance the full dataset before
  cross-validation (the reference study states this in its Sections 4 and 5.3), tune by grid
  search, and report the best cross-validated score.
* **Naive Bayes.** The reference study used Bernoulli NB on UCI and Gaussian NB on Mendeley;
  this code uses Gaussian NB on both.
* **E2/E3** train each classifier once, single-threaded, with default hyper-parameters on a
  stratified sample of 20,000 PhiUSIIL URLs (`--n-train`) and test on a held-out 30% split.
* **Randomness.** All random steps use seed 42 (`phishgap/config.py`).

## AI assistance

This code was written, and the experiments were run, with an AI assistant (Claude, Anthropic),
as permitted by the course instructor. All results come from running this code on the public
datasets; none were edited by hand.

## Data and references

* R. Mohammad, L. McCluskey. *Phishing Websites*. UCI Machine Learning Repository (2012). <https://doi.org/10.24432/C51W2X>
* C. L. Tan. *Phishing Dataset for Machine Learning: Feature Evaluation*. Mendeley Data (2018). <https://doi.org/10.17632/h3cgnj8hft.1>
* A. Prasad, S. Chandra. PhiUSIIL: a diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security* 136, 103545 (2024). <https://doi.org/10.1016/j.cose.2023.103545>
* S. R. Abdul Samad et al. Analysis of the performance impact of fine-tuned machine learning model for phishing URL detection. *Electronics* 12(7), 1642 (2023). <https://doi.org/10.3390/electronics12071642>
