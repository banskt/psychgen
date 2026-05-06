"""
Shared logic fitting Clorinn for split replication CV.
Used by:
    cv_sr_fit_clorinn_single_repeat.py
    cv_sr_fit_clorinn_batch_repeat.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import fit_clorinn, ensure_parent


def load_split_input(path):
    with np.load(path, allow_pickle=False) as data:
        assignments   = data["assignments"]    # (n_repeats, P_eligible) int8
        eligible_cols = data["eligible_cols"]  # (P_eligible,)           int32
        fold_sizes    = data["fold_sizes"]     # (n_folds,)              int32
        p_total       = int(data["p_total"])
        n_folds       = int(data["n_folds"])
        n_repeats     = int(data["n_repeats"])
    return assignments, eligible_cols, fold_sizes, p_total, n_folds, n_repeats


def extract_subspace(X):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U, s


def load_zscore_and_noise(zscore_path, noise_cov_path):
    Z = pd.read_csv(Path(zscore_path), header=0, index_col=0).to_numpy()
    noise_cov = pd.read_csv(Path(noise_cov_path), header=0, index_col=0).to_numpy()

    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")
    if noise_cov.ndim != 2:
        raise ValueError("Noise covariance must be a 2D matrix.")
    n0, n1 = noise_cov.shape
    if n0 != n1:
        raise ValueError(f"Noise covariance is not square: shape ({n0}, {n1}).")
    if Z.shape[0] != n0:
        raise ValueError(
            f"Z has {Z.shape[0]} rows but noise_cov has {n0}."
        )
    return Z, noise_cov


def parse_fit_params(snakemake_params):
    """Parse the params shared by both rules into a plain dict."""
    svd_max_iter = snakemake_params.cv_svd_max_iter
    if svd_max_iter in [None, "None", "none", "null", ""]:
        svd_max_iter = None
    else:
        svd_max_iter = int(svd_max_iter)

    return dict(
        max_iter     = int(snakemake_params.cv_max_iter),
        pgd_max_iter = int(snakemake_params.cv_pgd_max_iter),
        svd_max_iter = svd_max_iter,
        svd_method   = snakemake_params.cv_svd_method,
        dg_tol       = float(snakemake_params.cv_dg_tol),
    )


def fit_one_repeat(
    *,
    Z,
    noise_cov,
    assignments,
    eligible_cols,
    p_total,
    n_folds,
    n_repeats,
    repeat_id,
    model,
    solver,
    nucnorm_full,
    sparse_scale,
    fit_params, # max_iter, pgd_max_iter, svd_max_iter, svd_method, dg_tol
    fit_result_out,
    subspace_out,
    metrics_out,
    logger,
):
    if repeat_id < 0 or repeat_id >= n_repeats:
        raise ValueError(
            f"repeat_id={repeat_id} out of range for n_repeats={n_repeats}."
        )

    N, P = Z.shape
    if P != p_total:
        raise ValueError(
            f"Z has {P} columns but split input records p_total={p_total}."
        )

    logger.info("=" * 20)
    logger.info(f"repeat_id    : {repeat_id}")
    logger.info(f"nucnorm_full : {nucnorm_full:g}")
    logger.info(f"matrix shape : {N} traits × {P} SNPs")
    logger.info(f"n_folds      : {n_folds}")

    fit_artifacts = {}
    U_list        = {}
    s_list        = {}
    metrics_list  = []

    for fold_id in range(n_folds):
        col_idx    = eligible_cols[assignments[repeat_id] == fold_id]
        n_cols_fit = col_idx.size
        r_fit      = nucnorm_full * np.sqrt(n_cols_fit / eligible_cols.size)

        logger.info(
            f"  fold {fold_id}: n_cols={n_cols_fit}, r_fit={r_fit:.4f}, "
            f"scale={np.sqrt(n_cols_fit / eligible_cols.size):.6f}"
        )

        Z_fold = Z[:, col_idx]

        fit_artifact = fit_clorinn(
            Z_fold,
            r_fit,
            sparse_scale = sparse_scale,
            model        = model,
            solver       = solver,
            noise_cov    = noise_cov,
            max_iter     = fit_params["max_iter"],
            pgd_max_iter = fit_params["pgd_max_iter"],
            svd_max_iter = fit_params["svd_max_iter"],
            svd_method   = fit_params["svd_method"],
            tol          = fit_params["dg_tol"],
        )
        fit_result = fit_artifact["final_result"]
        U, s = extract_subspace(fit_result.X)

        fit_artifacts[fold_id] = fit_artifact
        U_list[fold_id]        = U
        s_list[fold_id]        = s

        metrics_list.append({
            "nucnorm_full":    nucnorm_full,
            "nucnorm_fit":     round(r_fit, 4),
            "n_cols_eligible": int(eligible_cols.size),
            "n_cols_total":    int(p_total),
            "n_cols_fit":      int(n_cols_fit),
            "n_folds":         int(n_folds),
            "repeat_id":       int(repeat_id),
            "fold_id":         int(fold_id),
            "model":           str(model),
            "solver":          str(solver),
            "n_iter":          int(fit_result.n_iter),
            "converged":       bool(fit_result.converged),
            "message":         str(fit_result.message),
        })

        logger.info(
            f"  fold {fold_id}: n_iter={fit_result.n_iter}, "
            f"converged={fit_result.converged}, message={fit_result.message}"
        )

    ensure_parent(fit_result_out)
    with open(fit_result_out, "wb") as fh:
        pickle.dump(fit_artifacts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    ensure_parent(subspace_out)
    np.savez_compressed(
        subspace_out,
        **{f"U_f{f}": U_list[f] for f in range(n_folds)},
        **{f"s_f{f}": s_list[f] for f in range(n_folds)},
        n_folds      = n_folds,
        nucnorm_full = nucnorm_full,
        repeat_id    = repeat_id,
    )

    ensure_parent(metrics_out)
    with open(metrics_out, "w") as fh:
        json.dump(metrics_list, fh, indent=2, sort_keys=True)

    logger.info(f"Saved fit pickle  : {fit_result_out}")
    logger.info(f"Saved subspace    : {subspace_out}")
    logger.info(f"Saved fit metrics : {metrics_out}")
