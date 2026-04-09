#!/usr/bin/env python

import json
import pickle
from pathlib import Path

import numpy as np
from helpers import fit_clorinn, ensure_parent, clorinn_to_dict


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
    method =        snakemake.params.cv_method
    nucnorm =       float(snakemake.wildcards.nucnorm)
    max_iter =      snakemake.params.cv_max_iter
    svd_max_iter =  snakemake.params.cv_svd_max_iter
    model_out =     snakemake.output.cv_model_out
    metrics_out =   snakemake.output.cv_metrics_out

    # -- Fit model ------
    ztrue, ztrain, zmask = load_cv_data(cv_input)
    model = fit_clorinn(ztrain, method, nucnorm, max_iter=max_iter, svd_max_iter=svd_max_iter)

    # -- Save results ------
    metrics = heldout_metrics(ztrue, model.X, zmask)
    metrics.update(
        {
            "method": method,
            "nucnorm": nucnorm,
            "max_iter": max_iter,
            "n_iter" : len(model.steps),
        }
    )
    model_dict = clorinn_to_dict(model)

    ensure_parent(model_out)
    ensure_parent(metrics_out)

    with open(model_out, "wb") as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(metrics_out, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"Finished model fit at nucnorm={nucnorm:g}")
    print(f"Held-out MSE: {metrics["heldout_mse"]:.10f}")


if __name__ == "__main__":
    main()
