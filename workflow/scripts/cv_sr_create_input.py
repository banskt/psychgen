#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

from helpers import ensure_parent, setup_logger


def main():

    # -- Snakemake interface --------------------------------------------------
    zscore_data = snakemake.input.zscore_data
    n_repeats   = int(snakemake.params.n_repeats)
    n_folds     = int(snakemake.params.n_folds)
    seed        = int(snakemake.params.seed)
    split_input = snakemake.output.split_input
    log_path    = snakemake.log[0] if snakemake.log else None
    logger      = setup_logger(Path(__file__).stem, log_path)

    # -- Validate params ------------------------------------------------------
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    zscore_path = Path(zscore_data)
    if zscore_path.suffix != ".csv":
        raise ValueError("Input data must be a .csv file containing Z.")
    out_path = Path(split_input)
    if out_path.suffix != ".npz":
        raise ValueError("Split input file must be a .npz file.")

    # -- Read data ------------------------------------------------------------
    Z_df = pd.read_csv(zscore_path, header=0, index_col=0)
    row_names = np.array(Z_df.index.tolist())
    col_names = np.array(Z_df.columns.tolist())
    Z = Z_df.to_numpy()
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")
    N, P = Z.shape

    # -- Eligible columns -----------------------------------------------------
    # Currently all columns are eligible. In future this may restrict to
    # columns with sufficient observations across traits.
    eligible_cols = np.arange(P, dtype=np.int32)
    P_eligible    = eligible_cols.size

    if P_eligible < n_folds:
        raise ValueError(
            f"Fewer eligible columns ({P_eligible}) than n_folds ({n_folds})."
        )

    # -- Fold sizes -----------------------------------------------------------
    # Distribute remainder across the first (P_eligible % n_folds) folds so
    # that fold sizes differ by at most 1. Sizes are constant across repeats.
    base_size  = P_eligible // n_folds
    remainder  = P_eligible % n_folds
    fold_sizes = np.full(n_folds, base_size, dtype=np.int32)
    fold_sizes[:remainder] += 1                        # first `remainder` folds get +1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    logger.info(f"Matrix shape         : {N} traits × {P} SNPs")
    logger.info(f"Eligible columns     : {P_eligible}")
    logger.info(f"n_repeats            : {n_repeats}")
    logger.info(f"n_folds              : {n_folds}")
    logger.info(f"Base fold size       : {base_size}  (+1 for first {remainder} folds)")
    logger.info(f"Fold sizes           : {fold_sizes.tolist()}")

    # -- Generate fold assignments --------------------------------------------
    # assignments[rep, j] = f  means eligible column j belongs to fold f
    # in repeat rep.  Values in {0, 1, ..., n_folds - 1}.
    rng         = np.random.default_rng(seed)
    assignments = np.full((n_repeats, P_eligible), -1, dtype=np.int8)
    # but what if n_folds exceed np.int8?
    if n_folds > np.iinfo(np.int8).max:
        raise ValueError(
            f"n_folds={n_folds} exceeds int8 capacity. "
            f"Use a wider dtype for assignments."
        )

    for rep in range(n_repeats):
        perm = rng.permutation(P_eligible)
        for f in range(n_folds):
            idx = perm[boundaries[f]:boundaries[f + 1]]
            assignments[rep, perm[boundaries[f]:boundaries[f + 1]]] = f

    # -- Validate -------------------------------------------------------------
    for rep in range(n_repeats):
        if np.any(assignments[rep] < 0):
            raise RuntimeError(f"Repeat {rep}: Some eligible columns were not assigned to a fold.")

        if np.any(assignments[rep] >= n_folds):
            raise RuntimeError(f"Repeat {rep}: Invalid fold labels found.")

        counts = np.bincount(assignments[rep], minlength=n_folds)
        if not np.array_equal(np.sort(counts), np.sort(fold_sizes)):
            raise RuntimeError(
                f"Repeat {rep}: fold count mismatch. "
                f"Expected {np.sort(fold_sizes).tolist()}, got {np.sort(counts).tolist()}."
            )
    logger.info("Fold assignment validation passed.")

    # -- Save -----------------------------------------------------------------
    ensure_parent(out_path)
    np.savez_compressed(
        out_path,
        assignments   = assignments,     # (n_repeats, P_eligible)  int8
        eligible_cols = eligible_cols,   # (P_eligible,)            int32  [= arange(P) for now]
        row_names     = row_names,       # (N,)                     str    [for debugging]
        col_names     = col_names,       # (P,)                     str    [for debugging]
        fold_sizes    = fold_sizes,      # (n_folds,)               int32
        split_seed    = seed,
        n_repeats     = n_repeats,
        n_folds       = n_folds,
        p_total       = P,
    )

    logger.info(f"Saved split assignments to: {out_path}")


if __name__ == "__main__":
    main()
