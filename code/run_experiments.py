"""Run the experiments and write results, LaTeX tables and figures.

Examples
--------
    python run_experiments.py --synthetic --fast     # 2-minute smoke test, numbers are meaningless
    python run_experiments.py --fast                 # real data, reduced grids (about 15-30 min)
    python run_experiments.py                        # full run used for the paper
    python run_experiments.py --only e2              # one experiment
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from phishgap import data, plots
from phishgap.config import CLASSIFIERS
from phishgap.experiments import environment, run_e1, run_e2_e3, write_tables
from phishgap.models import available

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["e1", "e2"], help="run only E1, or only E2+E3")
    ap.add_argument("--fast", action="store_true", help="smaller CV and grids, fewer repeats")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on generated data")
    ap.add_argument("--models", nargs="+", default=CLASSIFIERS, help="subset of " + " ".join(CLASSIFIERS))
    ap.add_argument("--n-train", type=int, default=20000, help="PhiUSIIL training rows used in E2/E3")
    ap.add_argument("--results", default=str(ROOT / "results"), help="output folder")
    ap.add_argument("--paper-dir", default=str(ROOT.parent / "paper"), help="where figures for the paper go")
    ap.add_argument("--tables-only", action="store_true",
                    help="rebuild LaTeX tables and figures from saved CSV results without rerunning")
    args = ap.parse_args()

    if args.tables_only:
        res = Path(args.results)
        read = lambda name: pd.read_csv(res / name) if (res / name).exists() else None
        e1, e2, e3 = read("e1_protocols.csv"), read("e2_imbalance.csv"), read("e3_cost.csv")
        if e1 is not None:
            plots.e1_protocols(e1, Path(args.paper_dir) / "fig_e1_protocols.pdf")
        if e2 is not None:
            plots.e2_imbalance(e2, Path(args.paper_dir) / "fig_e2_imbalance.pdf")
        write_tables(res, e1, e2, e3)
        print(f"Tables rebuilt in {res / 'tables'}; figures in {args.paper_dir}")
        return

    out = Path(args.results) / ("synthetic" if args.synthetic else "")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures" if args.synthetic else Path(args.paper_dir)
    models = available(args.models)
    log_file = (out / "run.log").open("a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} | fast={args.fast} synthetic={args.synthetic}")
    log(f"environment: {environment(out)}")
    e1 = e2 = e3 = None

    if args.only in (None, "e1"):
        if args.synthetic:
            datasets = [data.synthetic_tabular()]
        else:
            datasets = [data.load_uci()]
            mendeley = data.load_mendeley()
            if mendeley is None:
                log("[E1] Mendeley CSV not found in data/mendeley/ - running E1 on UCI only")
            else:
                datasets.append(mendeley)
        e1 = run_e1(datasets, models, fast=args.fast, log=log)
        e1.to_csv(out / "e1_protocols.csv", index=False)
        plots.e1_protocols(e1, fig_dir / "fig_e1_protocols.pdf")

    if args.only in (None, "e2"):
        X, y, urls, _ = data.synthetic_urls() if args.synthetic else data.load_phiusiil()
        e2, e3 = run_e2_e3(X, y, urls, models, n_train=args.n_train, fast=args.fast, log=log)
        e2.to_csv(out / "e2_imbalance.csv", index=False)
        e3.to_csv(out / "e3_cost.csv", index=False)
        plots.e2_imbalance(e2, fig_dir / "fig_e2_imbalance.pdf")

    write_tables(out, e1, e2, e3)
    rel = lambda q: Path(q).resolve().relative_to(ROOT.parent) if Path(q).resolve().is_relative_to(ROOT.parent) else q
    log(f"=== done. Results in {rel(out)}, figures in {rel(fig_dir)}")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    warnings.filterwarnings("ignore", message="Features .* are constant")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    main()
