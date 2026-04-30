#!/usr/bin/env python

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import fit_clorinn, ensure_parent, setup_logger, run_with_snakemake_log


def load_split_input(path):
    with np.load(path, allow_pickle=False) as data:
        assignments   = data["assignments"]
        eligible_cols = data["eligible_cols"]
        fold_sizes    = data["fold_sizes"]
        p_total       = int(data["p_total"])
        n_folds       = int(data["n_folds"])
        n_repeats     = int(data["n_repeats"])
    return assignments, eligible_cols, fold_sizes, p_total, n_folds, n_repeats


def extract_subspace(X):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U, s


def fit_one_repeat(
    *,
    Z,
    assignments,
    eligible_cols,
    p_total,
    n_folds,
    n_repeats,
    nucnorm_full,
    repeat_id,
    model,
    solver,
    max_iter,
    pgd_max_iter,
    svd_max_iter,
    svd_method,
    dg_tol,
    sparse_scale,
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
            f"  fold {fold_id}: n_cols={n_cols_fit}, "
            f"r_fit={r_fit:.4f}, "
            f"scale={np.sqrt(n_cols_fit / eligible_cols.size):.6f}"
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


def main():
    cv_input     = snakemake.input.cv_input
    zscore_data  = snakemake.input.zscore_data

    nucnorm_full = float(snakemake.wildcards.nucnorm)
    repeat_ids   = [int(x) for x in snakemake.params.repeat_ids]

    model        = snakemake.params.cv_model
    solver       = snakemake.params.cv_solver
    max_iter     = int(snakemake.params.cv_max_iter)
    pgd_max_iter = int(snakemake.params.cv_pgd_max_iter)

    svd_max_iter = snakemake.params.cv_svd_max_iter
    if svd_max_iter in [None, "None", "none", "null", ""]:
        svd_max_iter = None
    else:
        svd_max_iter = int(svd_max_iter)

    svd_method   = snakemake.params.cv_svd_method
    dg_tol       = float(snakemake.params.cv_dg_tol)
    sparse_scale = float(snakemake.params.cv_sparse_scale)

    log_path = snakemake.log[0] if snakemake.log else None
    logger   = setup_logger(Path(__file__).stem, log_path)

    fit_result_paths = list(snakemake.output.fit_result)
    subspace_paths   = list(snakemake.output.subspace)
    metrics_paths    = list(snakemake.output.fit_metrics)

    if not (
        len(repeat_ids)
        == len(fit_result_paths)
        == len(subspace_paths)
        == len(metrics_paths)
    ):
        raise ValueError(
            "repeat_ids and output lists have inconsistent lengths: "
            f"repeat_ids={len(repeat_ids)}, "
            f"fit_result={len(fit_result_paths)}, "
            f"subspace={len(subspace_paths)}, "
            f"fit_metrics={len(metrics_paths)}."
        )

    logger.info(f"Batch fit for nucnorm={nucnorm_full:g}")
    logger.info(f"Repeat IDs: {repeat_ids}")

    assignments, eligible_cols, fold_sizes, p_total, n_folds, n_repeats = (
        load_split_input(cv_input)
    )

    Z = pd.read_csv(Path(zscore_data), header=0, index_col=0).to_numpy()
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")

    for repeat_id, fit_out, subspace_out, metrics_out in zip(
        repeat_ids,
        fit_result_paths,
        subspace_paths,
        metrics_paths,
    ):
        fit_one_repeat(
            Z               = Z,
            assignments     = assignments,
            eligible_cols   = eligible_cols,
            p_total         = p_total,
            n_folds         = n_folds,
            n_repeats       = n_repeats,
            nucnorm_full    = nucnorm_full,
            repeat_id       = repeat_id,
            model           = model,
            solver          = solver,
            max_iter        = max_iter,
            pgd_max_iter    = pgd_max_iter,
            svd_max_iter    = svd_max_iter,
            svd_method      = svd_method,
            dg_tol          = dg_tol,
            sparse_scale    = sparse_scale,
            fit_result_out  = fit_out,
            subspace_out    = subspace_out,
            metrics_out     = metrics_out,
            logger          = logger,
        )

    logger.info("Finished batch fit.")


if __name__ == "__main__":
    run_with_snakemake_log(main, snakemake)
