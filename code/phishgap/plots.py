"""Figures for the paper (vector PDF, sized for the Springer single-column layout)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
PALETTE = ["#1f4e79", "#b4451f", "#2e7d4f", "#7b4f9e", "#c28a00", "#4a8fb5", "#8c564b", "#555555"]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def motivation(path):
    """Precision implied by the reference study's own precision/recall as phishing becomes rare.

    Operating points: tuned XGBoost (P = 98.537 %, R = 97.98 %) and tuned SVM
    (P = R = 96.66 %) on the balanced Mendeley data, Table 12 of Abdul Samad et al. (2023).
    On a balanced test set FPR = R (1/P - 1); at ratio r, precision = TPR / (TPR + r FPR).
    """
    fpr = lambda p, r_: r_ * (1 / p - 1)
    curves = [("Tuned XGBoost [ref.] (P = 98.54%, R = 97.98%)", 0.9798, fpr(0.98537, 0.9798), PALETTE[0], "-", (7, 7)),
              ("Tuned SVM [ref.] (P = R = 96.66%)", 0.9666, fpr(0.9666, 0.9666), PALETTE[1], "--", (7, -7)),
              ("Reference line: TPR = 98%, FPR = 0.1%", 0.98, 0.001, "#555555", ":", (7, 7))]
    r = np.logspace(0, 3, 400)
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    for label, tpr, f, colour, ls, off in curves:
        ax.plot(r, 100 * tpr / (tpr + r * f), color=colour, ls=ls, lw=1.6, label=label)
        p100 = 100 * tpr / (tpr + 100 * f)
        ax.plot([100], [p100], "o", color=colour, ms=4, zorder=3)
        ax.annotate(f"{p100:.0f}%", xy=(100, p100), xytext=off, textcoords="offset points", fontsize=8,
                    color=colour, va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.axvline(100, color="#999999", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(115, 3, "1 : 100", fontsize=7.5, color="#666666", va="bottom")
    ax.set(xscale="log", xlim=(1, 1000), ylim=(0, 100),
           xlabel="Legitimate URLs per phishing URL, $r$ (log scale)", ylabel="Precision (%)")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def architecture(path):
    """Block diagram of the evaluation framework implemented in this repository."""
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.set(xlim=(0, 10), ylim=(0.6, 6.3))
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=8, weight="bold")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=7, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     lw=0.9, color="#333333"))

    box(0.1, 3.7, 2.5, 2.5, "Data sources", "UCI Phishing\nWebsites (2012)\nMendeley (2018)\nPhiUSIIL (2024)\n+ raw URLs", "#eef3f8")
    box(3.0, 3.7, 3.4, 2.5, "Features", "Provided features\n(URL, page, external)\n\nURL-only lexical\nfeatures (23, this work)", "#f3f3f3")
    box(6.8, 3.7, 3.1, 2.5, "Training folds only", "Min-max scaling\nSMOTE balancing\nSelectKBest (ANOVA F)\nGrid search (inner CV)\n8 classifiers", "#fbf1ec")
    box(0.1, 0.7, 3.0, 2.3, "E1: protocol", "A: preprocess all data,\nreport best CV score\n(reference study)\nB: nested CV,\nno leakage", "#eef6f0")
    box(3.5, 0.7, 3.0, 2.3, "E2: imbalance", "Test sets with\nr = 1, 10, 100\nlegitimate per phishing\nObserved vs predicted\nprecision", "#eef6f0")
    box(6.9, 0.7, 3.0, 2.3, "E3: cost", "Training time\nPrediction time / URL\nExtraction time / URL\nMCC at r = 1 and 100", "#eef6f0")
    arrow(2.6, 4.95, 3.0, 4.95)
    arrow(6.4, 4.95, 6.8, 4.95)
    for x in (1.6, 5.0, 8.4):
        arrow(8.35, 3.7, x, 3.0)
    fig.tight_layout()
    return _save(fig, path)


def placeholder(path, text):
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="#666666")
    return _save(fig, path)


def e1_protocols(e1, path):
    """Accuracy gap between protocol A (reference) and protocol B (leakage-free), per classifier."""
    keys = list(dict.fromkeys(e1["dataset"]))
    models = list(dict.fromkeys(e1["model"]))
    names = {"uci": "UCI (SMOTE applied)", "mendeley": "Mendeley (already balanced)"}
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    y = np.arange(len(models))
    height = 0.8 / len(keys)
    for j, key in enumerate(keys):
        d = e1[e1["dataset"] == key].set_index("model").reindex(models)
        gap = d["A_accuracy"] - d["B_accuracy"]
        pos = y - 0.4 + height * (j + 0.5)
        ax.barh(pos, gap, height=height * 0.9, color=PALETTE[j], label=names.get(key, key), zorder=2)
        for g, p_ in zip(gap, pos):
            ax.text(max(g, 0) + 0.04, p_, f"{g:+.2f}", va="center", fontsize=6.5, color="#333333")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlim(-0.25, max(3.0, float((e1["A_accuracy"] - e1["B_accuracy"]).max()) + 0.4))
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy of protocol A minus protocol B (percentage points)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def e2_imbalance(e2, path):
    """Observed precision vs imbalance ratio, one panel per feature set."""
    sets = list(dict.fromkeys(e2["features"]))
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2, 3.1), sharey=True, squeeze=False)
    models = list(dict.fromkeys(e2["model"]))
    for ax, fs in zip(axes[0], sets):
        d = e2[e2["features"] == fs]
        for j, m in enumerate(models):
            dm = d[d["model"] == m].sort_values("ratio")
            ax.errorbar(dm["ratio"], dm["precision"], yerr=dm["precision_sd"], marker="o", ms=3,
                        lw=1.1, capsize=2, color=PALETTE[j % len(PALETTE)], label=m)
            ax.plot(dm["ratio"], dm["precision_predicted"], ls=":", lw=0.9, color=PALETTE[j % len(PALETTE)])
        ax.set(xscale="log", ylim=(0, 102), xlabel="Legitimate per phishing URL, $r$",
               title="Provided features" if fs == "provided" else "URL-only lexical features")
        ax.set_xticks(sorted(d["ratio"].unique()), [str(v) for v in sorted(d["ratio"].unique())])
        ax.title.set_fontsize(9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axes[0][0].set_ylabel("Precision (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)
