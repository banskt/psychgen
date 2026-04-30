#!/usr/bin/env python

import numpy as np
from pathlib import Path

from helpers import ensure_parent, setup_logger
from cv_sr_helpers import (
    load_stability, validate_k_ref, build_stability_df,
    choose_plateau_threshold, log_stability_curve,
)


def main():

    # -- Snakemake interface --------------------------------------------------
    coarse_stability  = snakemake.input.coarse_stability
    fine_nucnorms_out = snakemake.output.fine_nucnorms
    n_points          = int(snakemake.params.n_points)
    k_pivot           = int(snakemake.params.k_pivot)
    one_se_multiplier = float(snakemake.params.one_se_multiplier)
    abs_tolerance     = float(snakemake.params.abs_tolerance)
    rel_tolerance     = float(snakemake.params.rel_tolerance)
    log_path          = snakemake.log[0] if snakemake.log else None
    logger            = setup_logger(Path(__file__).stem, log_path)

    # -- Load and validate ----------------------------------------------------
    records = [load_stability(p) for p in coarse_stability]
    k_ref   = validate_k_ref(records)

    if k_pivot not in k_ref:
        raise ValueError(
            f"k_pivot={k_pivot} is not present in the stability k grid {k_ref}."
        )

    # -- Build coarse stability table -----------------------------------------
    # Only mean_dist and se_dist needed — fine grid bracket uses no other fields
    df        = build_stability_df(records, k_ref)
    score_col = f"mean_dist_k{k_pivot}"
    se_col    = f"se_dist_k{k_pivot}"

    # -- Find plateau onset ---------------------------------------------------
    selected, best_row, threshold = choose_plateau_threshold(
        df, score_col, se_col,
        one_se_multiplier, abs_tolerance, rel_tolerance,
    )
    selected_r = float(selected["nucnorm"])
    best_r     = float(best_row["nucnorm"])

    # -- Boundary checks ------------------------------------------------------
    selected_idx = df.index[df["nucnorm"] == selected_r][0]

    if selected_idx == 0:
        raise ValueError(
            f"Plateau onset at k={k_pivot} is the smallest coarse nucnorm "
            f"({selected_r:.0f}); extend the coarse grid downward."
        )
    if selected_idx == len(df) - 1:
        raise ValueError(
            f"Plateau onset at k={k_pivot} is the largest coarse nucnorm "
            f"({selected_r:.0f}); extend the coarse grid upward."
        )

    lo = float(df.loc[selected_idx - 1, "nucnorm"])
    hi = float(df.loc[selected_idx + 1, "nucnorm"])

    # -- Fine grid: power-of-2 spacing symmetric around selected_r -----------
    if n_points < 2:
        raise ValueError(f"n_points must be an even integer >= 2, got {n_points}.")
    if n_points % 2 != 0:
        logger.warning(
            f"n_points={n_points} is odd; will use {2 * (n_points // 2)} fine points."
        )
    n_each    = n_points // 2
    left_exp  = np.linspace(np.log2(lo),         np.log2(selected_r), n_each + 2)[1:-1]
    right_exp = np.linspace(np.log2(selected_r), np.log2(hi),         n_each + 2)[1:-1]
    fine_grid = sorted(set(int(round(v)) for v in np.concatenate([2**left_exp, 2**right_exp])))

    if len(fine_grid) < n_points:
        logger.warning(
            f"Requested {n_points} fine points but got {len(fine_grid)} unique "
            "integer radii after rounding."
        )

    # -- Log ------------------------------------------------------------------
    log_stability_curve(logger, df, score_col, se_col, best_r, selected_r,
                        header=f"Coarse stability curve at k={k_pivot}:")
    logger.info(f"Stability minimum      : nucnorm={best_r:.0f}  "
                f"mean_dist={best_row[score_col]:.6f}")
    logger.info(f"Tolerance threshold    : {threshold:.6f}  "
                f"(1SE×{one_se_multiplier} + abs={abs_tolerance} + rel={rel_tolerance})")
    logger.info(f"Plateau onset selected : nucnorm={selected_r:.0f}  "
                f"mean_dist={selected[score_col]:.6f}")
    logger.info(f"Fine bracket           : [{lo:.0f}, {hi:.0f}]")
    logger.info(f"Fine grid              : {fine_grid}")

    # -- Save -----------------------------------------------------------------
    ensure_parent(fine_nucnorms_out)
    with open(fine_nucnorms_out, "w") as fh:
        for v in fine_grid:
            fh.write(f"{v}\n")


if __name__ == "__main__":
    main()
