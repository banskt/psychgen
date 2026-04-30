#!/usr/bin/env python

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import fit_clorinn, ensure_parent, setup_logger


def load_split_input(path):
    with np.load(path, allow_pickle=False) as data:
        assignments   = data["assignments"]    # (n_repeats, P_eligible)  int8
        eligible_cols = data["eligible_cols"]  # (P_eligible,)            int32
        fold_sizes    = data["fold_sizes"]     # (n_folds,)               int32
        p_total       = int(data["p_total"])
        n_folds       = int(data["n_folds"])
        n_repeats     = int(data["n_repeats"])
    return assignments, eligible_cols, fold_sizes, p_total, n_folds, n_repeats


def extract_subspace(X):
    """
    Compute the full thin SVD of X and return left singular vectors and values.
    All N left singular vectors are stored; so the aggregation step can 
    slice U[:, :k] without re-running SVD.

    Parameters
    ----------
    X : ndarray, shape (N, P)
        Fitted low-rank matrix for one fold.

    Returns
    -------
    U : ndarray, shape (N, N)
        Full matrix of left singular vectors (columns).
    s : ndarray, shape (N,)
        Singular values in descending order.
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U, s


def main():

    # -- Snakemake interface --------------------------------------------------
    cv_input      = snakemake.input.cv_input
    zscore_data   = snakemake.input.zscore_data
    nucnorm_full  = float(snakemake.wildcards.nucnorm)
    repeat_id     = int(snakemake.wildcards.repeat_id)
    model         = snakemake.params.cv_model
    solver        = snakemake.params.cv_solver
    max_iter      = int(snakemake.params.cv_max_iter)
    pgd_max_iter  = int(snakemake.params.cv_pgd_max_iter)
    svd_max_iter  = snakemake.params.cv_svd_max_iter   # may be None
    svd_method    = snakemake.params.cv_svd_method
    dg_tol        = float(snakemake.params.cv_dg_tol)
    sparse_scale  = float(snakemake.params.cv_sparse_scale)
    fit_result_out = snakemake.output.fit_result
    subspace_out  = snakemake.output.subspace
    metrics_out   = snakemake.output.fit_metrics
    log_path      = snakemake.log[0] if snakemake.log else None
    logger        = setup_logger(Path(__file__).stem, log_path)

    if svd_max_iter in [None, "None", "none", "null", ""]:
        svd_max_iter = None

    # -- Load split assignments -----------------------------------------------
    assignments, eligible_cols, fold_sizes, p_total, n_folds, n_repeats = \
        load_split_input(cv_input)

    if repeat_id < 0 or repeat_id >= n_repeats:
        raise ValueError(
            f"repeat_id={repeat_id} out of range for n_repeats={n_repeats}."
        )

    # -- Load full Z ----------------------------------------------------------
    Z = pd.read_csv(Path(zscore_data), header=0, index_col=0).to_numpy()
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")
    N, P = Z.shape
    if P != p_total:
        raise ValueError(
            f"Z has {P} columns but split input records p_total={p_total}."
        )

    logger.info(f"Matrix shape : {N} traits × {P} SNPs")
    logger.info(f"nucnorm_full : {nucnorm_full:g}")
    logger.info(f"repeat_id    : {repeat_id}")
    logger.info(f"n_folds      : {n_folds}")

    # -- Fit one model per fold -----------------------------------------------
    fit_artifacts = {}   # fold_id -> full fit_clorinn dict
    U_list        = {}   # fold_id -> U  (N, N)
    s_list        = {}   # fold_id -> s  (N,)
    metrics_list  = []

    for fold_id in range(n_folds):
        col_idx    = eligible_cols[assignments[repeat_id] == fold_id]
        n_cols_fit = col_idx.size
        r_fit      = nucnorm_full * np.sqrt(n_cols_fit / eligible_cols.size)

        logger.info(
            f"  fold {fold_id}: n_cols={n_cols_fit}, "
            f"r_fit={r_fit:.4f}  (scale={np.sqrt(n_cols_fit / p_total):.6f})"
        )

        Z_fold = Z[:, col_idx]

        fit_artifact = fit_clorinn(
            Z_fold,
            r_fit,
            sparse_scale = sparse_scale,
            model        = model,
            solver       = solver,
            max_iter     = max_iter,
            pgd_max_iter = pgd_max_iter,
            svd_max_iter = svd_max_iter,
            svd_method   = svd_method,
            tol          = dg_tol,
        )
        fit_result = fit_artifact["final_result"]

        U, s = extract_subspace(fit_result.X)
        fit_artifacts[fold_id] = fit_artifact
        U_list[fold_id]        = U
        s_list[fold_id]        = s

        metrics_list.append({
            "nucnorm_full":    nucnorm_full,
            "nucnorm_fit":     round(r_fit, 4),
            "n_cols_eligible": eligible_cols.size,
            "n_cols_total":    p_total,
            "n_cols_fit":      n_cols_fit,
            "n_folds":         n_folds,
            "repeat_id":       repeat_id,
            "fold_id":         fold_id,
            "model":           model,
            "solver":          solver,
            "n_iter":          int(fit_result.n_iter),
            "converged":       bool(fit_result.converged),
            "message":         str(fit_result.message),
        })

        logger.info(
            f"  fold {fold_id}: n_iter={fit_result.n_iter}  {fit_result.message}"
        )

    # -- Save pickle: all folds in one file -----------------------------------
    ensure_parent(fit_result_out)
    with open(fit_result_out, "wb") as fh:
        pickle.dump(fit_artifacts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # -- Save subspace: all folds in one file ---------------------------------
    # Full thin-SVD left vectors stored (shape N × N each) so that the
    # aggregation step can freely vary k = 1..N without re-loading pickles.
    ensure_parent(subspace_out)
    np.savez_compressed(
        subspace_out,
        **{f"U_f{f}": U_list[f] for f in range(n_folds)},
        **{f"s_f{f}": s_list[f] for f in range(n_folds)},
        n_folds      = n_folds,
        nucnorm_full = nucnorm_full,
        repeat_id    = repeat_id,
    )

    # -- Save metrics: all folds in one file ----------------------------------
    ensure_parent(metrics_out)
    with open(metrics_out, "w") as fh:
        json.dump(metrics_list, fh, indent=2, sort_keys=True)

    logger.info(f"Saved fit pickle  : {fit_result_out}")
    logger.info(f"Saved subspace    : {subspace_out}")
    logger.info(f"Saved fit metrics : {metrics_out}")


if __name__ == "__main__":
    main()
