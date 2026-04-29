#!/usr/bin/env python

import json
import pickle
from pathlib import Path

import numpy as np
from helpers import fit_clorinn, ensure_parent, setup_logger


def load_cv_data(path):
    data = np.load(path)
    return data["Ztrue"], data["Ztrain"], data["Zmask"]


def heldout_metrics(ztrue, zhat, zmask):
    mask = np.asarray(zmask).astype(bool)
    if mask.sum() == 0:
        raise ValueError("Zmask does not contain any held-out entries.")

    residual = zhat[mask] - ztrue[mask]
    mse = np.mean(np.square(residual))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))

    return {
        "n_heldout": int(mask.sum()),
        "heldout_mse": float(mse),
        "heldout_rmse": float(rmse),
        "heldout_mae": float(mae),
    }


def main():

    # -- Snakemake interface ------
    cv_input =      snakemake.input.cv_input
    nucnorm =       float(snakemake.wildcards.nucnorm)
    model =         snakemake.params.cv_model
    solver =        snakemake.params.cv_solver
    max_iter =      snakemake.params.cv_max_iter
    pgd_max_iter =  snakemake.params.cv_pgd_max_iter
    svd_max_iter =  snakemake.params.cv_svd_max_iter
    svd_method =    snakemake.params.cv_svd_method
    dg_tol =        snakemake.params.cv_dg_tol
    sparse_scale =  snakemake.params.cv_sparse_scale
    fit_result_out = snakemake.output.cv_fit_out
    metrics_out =   snakemake.output.cv_metrics_out
    log_path =      snakemake.log[0] if snakemake.log else None
    logger =        setup_logger(Path(__file__).stem, log_path)

    # -- Fit model ------
    ztrue, ztrain, zmask = load_cv_data(cv_input)
    fit_result = fit_clorinn(
        ztrain,
        nucnorm,
        sparse_scale=sparse_scale,
        model=model,
        solver=solver,
        max_iter=max_iter,
        pgd_max_iter=pgd_max_iter,
        svd_max_iter=svd_max_iter,
        svd_method=svd_method,
        tol=dg_tol
    )

    # -- Save results ------
    metrics = heldout_metrics(ztrue, fit_result.X, zmask)
    metrics.update(
        {
            "model": model,
            "solver" : solver,
            "nucnorm": nucnorm,
            "max_iter": max_iter,
            "n_iter" : fit_result.n_iter,
        }
    )
    #model_dict = clorinn_to_dict(model)

    ensure_parent(fit_result_out)
    ensure_parent(metrics_out)

    with open(fit_result_out, "wb") as handle:
        pickle.dump(fit_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(metrics_out, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    logger.info(f"Finished model fit at nucnorm={nucnorm:g}")
    logger.info(f"Number of iterations: {fit_result.n_iter:d}")
    logger.info(f"{fit_result.message}")
    logger.info(f"Held-out MSE: {metrics['heldout_mse']:.10f}")


if __name__ == "__main__":
    main()
