#!/usr/bin/env python

import json
from pathlib import Path

from helpers import ensure_parent, setup_logger
from cv_sr_helpers import (
    load_stability, validate_k_ref, build_stability_df,
    choose_plateau_threshold, log_stability_curve,
)

def main():

    # -- Snakemake interface --------------------------------------------------
    stability_paths    = list(snakemake.input.stability)
    fine_nucnorms_path = snakemake.input.fine_nucnorms
    summary_out        = snakemake.output.summary_out
    best_threshold_out = snakemake.output.best_threshold_out
    n_factors          = int(snakemake.params.n_factors)
    k_pivot            = int(snakemake.params.k_pivot)
    one_se_multiplier  = float(snakemake.params.one_se_multiplier)
    abs_tolerance      = float(snakemake.params.abs_tolerance)
    rel_tolerance      = float(snakemake.params.rel_tolerance)
    log_path           = snakemake.log[0] if snakemake.log else None
    logger             = setup_logger(Path(__file__).stem, log_path)

    # -- Load and validate ----------------------------------------------------
    records = [load_stability(p) for p in stability_paths]
    k_ref   = validate_k_ref(records)

    if k_pivot not in k_ref:
        raise ValueError(
            f"k_pivot={k_pivot} is not present in the stability k grid {k_ref}."
        )

    # -- Build full (coarse + fine) stability table ---------------------------
    df = build_stability_df(
        records, k_ref,
        extra_fields=["mean_gap_angle", "mean_energy", "se_energy"],
    )

    n_before = len(records)
    if len(df) < n_before:
        logger.warning(f"Dropped {n_before - len(df)} duplicate nucnorm entries.")

    # Tag coarse vs fine using the authoritative fine nucnorm list
    with open(fine_nucnorms_path) as fh:
        fine_set = set(int(line.strip()) for line in fh if line.strip())
    df["grid"] = df["nucnorm"].apply(lambda r: "fine" if int(r) in fine_set else "coarse")

    logger.info(f"Total nucnorm grid points : {len(df)}")
    logger.info(f"  coarse : {(df['grid'] == 'coarse').sum()}")
    logger.info(f"  fine   : {(df['grid'] == 'fine').sum()}")
    logger.info(f"k_pivot                   : {k_pivot}")

    # -- Apply plateau threshold criterion ------------------------------------
    score_col = f"mean_dist_k{k_pivot}"
    se_col    = f"se_dist_k{k_pivot}"

    selected, best_row, threshold = choose_plateau_threshold(
        df, score_col, se_col,
        one_se_multiplier, abs_tolerance, rel_tolerance,
    )
    selected_r    = float(selected["nucnorm"])
    best_r        = float(best_row["nucnorm"])
    selected_dist = float(selected[score_col])
    best_dist     = float(best_row[score_col])

    # -- Log ------------------------------------------------------------------
    log_stability_curve(logger, df, score_col, se_col, best_r, selected_r,
                        grid_col="grid",
                        header=f"Full stability table at k={k_pivot}:")
    logger.info(f"Stability minimum      : nucnorm={best_r:.0f}  "
                f"mean_dist={best_dist:.6f}")
    logger.info(f"Tolerance threshold    : {threshold:.6f}  "
                f"(1SE×{one_se_multiplier} + abs={abs_tolerance} + rel={rel_tolerance})")
    logger.info(f"Plateau onset selected : nucnorm={selected_r:.0f}  "
                f"mean_dist={selected_dist:.6f}")

    # -- Save summary CSV -----------------------------------------------------
    ensure_parent(summary_out)
    df.to_csv(summary_out, index=False)
    logger.info(f"Saved stability summary  : {summary_out}")

    # -- Save best threshold JSON ---------------------------------------------
    best_threshold = {
        "nucnorm_full":      selected_r,
        "k_pivot":           k_pivot,
        "mean_dist":         selected_dist,
        "stability_minimum": {
            "nucnorm_full":  best_r,
            "mean_dist":     best_dist,
        },
        "threshold":         threshold,
        "one_se_multiplier": one_se_multiplier,
        "abs_tolerance":     abs_tolerance,
        "rel_tolerance":     rel_tolerance,
    }

    ensure_parent(best_threshold_out)
    with open(best_threshold_out, "w") as fh:
        json.dump(best_threshold, fh, indent=2, sort_keys=False)
    logger.info(f"Saved best threshold     : {best_threshold_out}")
    logger.info(f"Best nucnorm_full        : {selected_r:.0f}")


if __name__ == "__main__":
    main()
