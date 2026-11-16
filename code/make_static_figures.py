"""Write the figures that do not depend on experiment results into ../paper:

fig_motivation.pdf    precision vs imbalance, computed from the reference study's numbers
fig_architecture.pdf  block diagram of the evaluation framework
fig_e1_protocols.pdf, fig_e2_imbalance.pdf  placeholders (only if missing), so the
                      paper compiles before the experiments have been run
"""
from pathlib import Path

from phishgap import plots

PAPER = Path(__file__).resolve().parent.parent / "paper"

if __name__ == "__main__":
    print(plots.motivation(PAPER / "fig_motivation.pdf"))
    print(plots.architecture(PAPER / "fig_architecture.pdf"))
    for name in ("fig_e1_protocols.pdf", "fig_e2_imbalance.pdf"):
        if not (PAPER / name).exists():
            print(plots.placeholder(PAPER / name, f"{name}\nRun: python run_experiments.py"))
