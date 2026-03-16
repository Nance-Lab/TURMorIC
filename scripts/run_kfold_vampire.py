"""
K-fold cross-validation pipeline for microglia morphology analysis.

Trains VAMPIRE shape models across K folds, aligns cluster labels using
contour-based Hungarian matching, and saves predictions + diagnostic plots.

Usage
-----
    # Single run
    python scripts/run_kfold_vampire.py \\
        --data-path /gscratch/cheme/nlsschim/data/cross_species/rat/thresholded/tifs/female \\
        --output-base ./output/rat_female \\
        --num-clusters 7 \\
        --num-pc 10 \\
        --num-points 50 \\
        --n-folds 5 \\
        --random-state 42

    # Minimal (uses defaults for everything except required args)
    python scripts/run_kfold_vampire.py \\
        --data-path /path/to/data \\
        --output-base ./output/run1

    # Via SLURM array (parameters injected from params file)
    sbatch scripts/submit_array.sh

Outputs (under output-base/)
-----------------------------
    fold_N/
        train/
        val/
        model_vampire_kfold_…pickle
        apply-properties_…csv
    plots/
        contours_overlay.png/pdf
        alignment_before_after.png/pdf
        shape_consistency_metrics.csv
    predictions_aligned.csv
    predictions_unaligned.csv
    run_config.json               ← records exact parameters used
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from turmoric.utils import (
    extract_age_region_condition,
    split_images_into_kfolds_balanced,
)
from turmoric.vampire_model import (
    align_kfold_predictions_with_contours,
    compare_alignment_effect,
    compare_contours_overlay,
    compute_shape_consistency_metrics,
    train_and_predict_vampire,
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAMPIRE k-fold cross-validation pipeline for microglia morphology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Root directory containing .tif image files (searched recursively).",
    )
    parser.add_argument(
        "--output-base",
        required=True,
        type=Path,
        help="Directory to write all outputs. Created if it does not exist.",
    )

    # VAMPIRE model parameters
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=7,
        help="Number of shape-mode clusters.",
    )
    parser.add_argument(
        "--num-pc",
        type=int,
        default=10,
        help=(
            "Number of principal components. Must be a fixed integer — "
            "never use 0 here. Run the scree-plot diagnostic to find the "
            "elbow for your dataset."
        ),
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        help="Number of contour points per cell.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vampire_kfold",
        help="Name tag embedded in the model pickle filename.",
    )

    # Cross-validation parameters
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        default=None,
        help=(
            "Total samples per condition across all folds. "
            "Must be divisible by --n-folds. "
            "If omitted, auto-determined from the smallest condition."
        ),
    )
    parser.add_argument(
        "--reference-fold",
        type=int,
        default=0,
        help="Fold index used as the alignment reference (0-indexed).",
    )

    # File discovery
    parser.add_argument(
        "--file-extension",
        type=str,
        default=".tif",
        help="File extension to search for.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args(argv)


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    """
    Persist the exact parameters used for this run to a JSON file.

    This ensures every output directory is self-documenting — you can
    always trace back which parameters produced which results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    output_dir : Path
        Directory to write ``run_config.json``.
    """
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    logging.getLogger(__name__).info(f"Run config saved to: {config_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    args.output_base.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_base / "plots"
    plots_dir.mkdir(exist_ok=True)

    save_run_config(args, args.output_base)

    logger.info("=" * 60)
    logger.info("VAMPIRE K-fold pipeline")
    logger.info("=" * 60)
    logger.info(f"  data_path    : {args.data_path}")
    logger.info(f"  output_base  : {args.output_base}")
    logger.info(f"  num_clusters : {args.num_clusters}")
    logger.info(f"  num_pc       : {args.num_pc}")
    logger.info(f"  num_points   : {args.num_points}")
    logger.info(f"  n_folds      : {args.n_folds}")
    logger.info(f"  random_state : {args.random_state}")
    logger.info(f"  ref_fold     : {args.reference_fold}")

    # ── Step 1: Balanced stratified k-fold split ───────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Splitting data into balanced k-folds")
    logger.info("=" * 60)

    fold_splits, usage_stats = split_images_into_kfolds_balanced(
        data_path=str(args.data_path),
        n_folds=args.n_folds,
        file_extension=args.file_extension,
        random_state=args.random_state,
        condition_extractor=extract_age_region_condition,
        samples_per_condition=args.samples_per_condition,
    )

    logger.info(
        f"Using {usage_stats['files_used']}/{usage_stats['total_files']} files | "
        f"{usage_stats['conditions']} conditions | "
        f"{usage_stats['samples_per_condition_per_fold']} samples/condition/fold"
    )

    # ── Step 2: Train and predict per fold ────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Training VAMPIRE models")
    logger.info("=" * 60)

    all_predictions = []
    model_paths = []

    for fold_idx, (train_files, val_files) in enumerate(fold_splits):
        fold_label = f"fold_{fold_idx + 1}"
        logger.info(f"\n{'─' * 40}")
        logger.info(f"Fold {fold_idx + 1} / {args.n_folds}")
        logger.info(f"{'─' * 40}")

        model_path, predictions_df = train_and_predict_vampire(
            train_files=train_files,
            val_files=val_files,
            output_dir=str(args.output_base / fold_label),
            fold_name=fold_label,
            model_name=args.model_name,
            num_points=args.num_points,
            num_clusters=args.num_clusters,
            num_pc=args.num_pc,
        )

        all_predictions.append(predictions_df)
        model_paths.append(str(model_path))
        logger.info(f"Fold {fold_idx + 1} complete — {model_path}")

    # ── Step 3: Align shape modes ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Aligning shape modes via contour-based Hungarian matching")
    logger.info("=" * 60)

    aligned_predictions = align_kfold_predictions_with_contours(
        all_predictions=all_predictions,
        model_paths=model_paths,
        reference_fold_idx=args.reference_fold,
    )

    # ── Step 4: Save predictions ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Saving predictions")
    logger.info("=" * 60)

    combined_aligned = pd.concat(aligned_predictions, ignore_index=True)
    combined_unaligned = pd.concat(all_predictions, ignore_index=True)

    out_aligned = args.output_base / "predictions_aligned.csv"
    out_unaligned = args.output_base / "predictions_unaligned.csv"

    combined_aligned.to_csv(out_aligned, index=False)
    combined_unaligned.to_csv(out_unaligned, index=False)

    logger.info(f"Aligned   : {out_aligned}")
    logger.info(f"Unaligned : {out_unaligned}")

    # ── Step 5: Diagnostic plots and metrics ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Generating diagnostic plots and metrics")
    logger.info("=" * 60)

    compare_contours_overlay(
        model_paths=model_paths,
        output_path=str(plots_dir / "contours_overlay.png"),
        fold_names=[f"Fold {i + 1}" for i in range(args.n_folds)],
    )

    compare_alignment_effect(
        model_paths=model_paths,
        output_path=str(plots_dir / "alignment_before_after.png"),
        reference_fold_idx=args.reference_fold,
    )

    metrics_df = compute_shape_consistency_metrics(
        model_paths=model_paths,
        reference_fold_idx=args.reference_fold,
    )
    metrics_df.to_csv(plots_dir / "shape_consistency_metrics.csv", index=False)

    # Flag any shape modes with negative improvement
    negative = metrics_df[metrics_df["improvement_percent"] < 0]
    if not negative.empty:
        logger.warning(
            f"Shape modes with negative alignment improvement "
            f"(may indicate ambiguous clusters): "
            f"{negative['shape_mode'].tolist()}"
        )

    logger.info("\nShape consistency summary:")
    logger.info(f"\n{metrics_df.to_string(index=False)}")
    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()