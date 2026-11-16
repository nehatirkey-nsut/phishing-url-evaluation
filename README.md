# Leakage-free and imbalance-aware evaluation of fine-tuned phishing URL classifiers

* [`code/`](code/): experiment code, raw results and instructions (start with [`code/README.md`](code/README.md))
* [`paper/`](paper/): LaTeX source of the paper (Springer Nature template), bibliography, figures and compiled PDF

Reproduce every table and figure:

```bash
cd code
pip install -r requirements.txt
python run_experiments.py
python analyse_features.py
python make_static_figures.py
```
