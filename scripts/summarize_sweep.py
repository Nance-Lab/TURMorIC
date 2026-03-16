#!/usr/bin/env python
"""
summarise_sweep.py  —  Aggregate VAMPIRE k-fold sweep results.

Produces two CSV files in the sweep directory:
  sweep_all_runs.csv      — one row per run (all 450)
  sweep_aggregated.csv    — one row per parameter set, averaged across seeds

Usage:
    python summarise_sweep.py --sweep-dir /path/to/vampire_sweep
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_run_label(label: str) -> dict:
    """
    Parse a run label like 'c5_pc10_p50_s42' into a dict of parameters.

    Returns
    -------
    dict with keys: num_clusters, num_pc, num_points, seed
    """
    params = {}
    for part in label.split("_"):
        if part.startswith("pc"):
            params["num_pc"] = int(part[2:])
        elif part.startswith("p"):
            params["num_points"] = int(part[1:])
        elif part.startswith("c"):
            params["num_clusters"] = int(part[1:])
        elif part.startswith("s"):
            params["seed"] = int(part[1:])
    return params


def _load_run(metrics_file: Path) -> dict | None:
    """
    Load a single shape_consistency_metrics.csv and return a summary record.
    Returns None if the file cannot be parsed.
    """
    run_dir = metrics_file.parent
    if run_dir.name == "plots":
        run_dir = run_dir.parent
    run_label = run_dir.name

    try:
        df = pd.read_csv(metrics_file)
    except Exception as exc:
        logger.warning(f"Could not read {metrics_file}: {exc}")
        return None

    required = {"improvement_percent", "mean_dist_after", "mean_dist_before"}
    if not required.issubset(df.columns):
        logger.warning(
            f"Missing columns in {metrics_file}. "
            f"Expected {required}, got {set(df.columns)}"
        )
        return None

    params = _parse_run_label(run_label)
    if not params:
        logger.warning(f"Could not parse run label: {run_label}")
        return None

    return {
        **params,
        "run_label": run_label,
        "mean_pct_improvement": df["improvement_percent"].mean(),
        "std_pct_improvement": df["improvement_percent"].std(),
        "mean_dist_before": df["mean_dist_before"].mean(),
        "mean_dist_after": df["mean_dist_after"].mean(),
        "std_dist_after": df["mean_dist_after"].std(),
        "n_negative": int((df["improvement_percent"] < 0).sum()),
        "n_shape_modes": len(df),
        # coefficient of variation — lower means more uniform consistency
        "cv_dist_after": (
            df["mean_dist_after"].std() / df["mean_dist_after"].mean()
            if df["mean_dist_after"].mean() > 0 else float("nan")
        ),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def summarise_sweep(sweep_dir: str | Path) -> pd.DataFrame:
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    metrics_files = sorted(sweep_dir.rglob("shape_consistency_metrics.csv"))
    logger.info(f"Found {len(metrics_files)} metrics files in {sweep_dir}")

    if not metrics_files:
        raise FileNotFoundError(
            "No shape_consistency_metrics.csv files found. "
            "Have any jobs completed yet?"
        )

    records = [_load_run(f) for f in metrics_files]
    records = [r for r in records if r is not None]
    logger.info(f"Successfully loaded {len(records)} runs")

    all_runs = pd.DataFrame(records).sort_values("mean_dist_after")

    # ── per-run output ────────────────────────────────────────────────────────
    all_runs_path = sweep_dir / "sweep_all_runs.csv"
    all_runs.to_csv(all_runs_path, index=False)
    logger.info(f"Saved per-run results → {all_runs_path}")

    print("\n=== Top 20 individual runs (by mean dist after alignment) ===")
    print(
        all_runs.head(20)[
            ["run_label", "num_clusters", "num_pc", "num_points", "seed",
             "mean_pct_improvement", "mean_dist_after", "n_negative", "cv_dist_after"]
        ].to_string(index=False)
    )

    # ── aggregated across seeds ───────────────────────────────────────────────
    group_cols = ["num_clusters", "num_pc", "num_points"]

    agg = (
        all_runs.groupby(group_cols)
        .agg(
            mean_pct_improvement=("mean_pct_improvement", "mean"),
            std_pct_improvement=("mean_pct_improvement", "std"),
            mean_dist_after=("mean_dist_after", "mean"),
            std_dist_after=("mean_dist_after", "std"),
            mean_dist_before=("mean_dist_before", "mean"),
            mean_n_negative=("n_negative", "mean"),
            max_n_negative=("n_negative", "max"),
            mean_cv_dist_after=("cv_dist_after", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )

    # robustness: low std across seeds is better than low mean alone
    agg["dist_plus_std"] = agg["mean_dist_after"] + agg["std_dist_after"]

    # rank: fewest (worst-case) negatives → lowest dist+std → highest improvement
    agg = agg.sort_values(
        ["max_n_negative", "dist_plus_std", "mean_pct_improvement"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg) + 1))

    agg_path = sweep_dir / "sweep_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    logger.info(f"Saved aggregated results → {agg_path}")

    print("\n=== Top 20 parameter sets (aggregated across seeds) ===")
    print(
        agg.head(20)[
            ["rank", "num_clusters", "num_pc", "num_points", "n_seeds",
             "mean_pct_improvement", "std_pct_improvement",
             "mean_dist_after", "std_dist_after",
             "mean_n_negative", "max_n_negative"]
        ].to_string(index=False)
    )

    return agg


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise VAMPIRE k-fold sweep results")
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Root directory of the sweep (contains one subdirectory per run)",
    )
    args = parser.parse_args()
    summarise_sweep(args.sweep_dir)