#!/usr/bin/env python

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import fit_clorinn, ensure_parent, clorinn_to_dict, setup_logger



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit the final full-data Clorinn model using the best CV threshold and "
            "a trait-level noise covariance Sigma."
        )
    )
    parser.add_argument("--input-matrix", required=True, help="Path to full Z matrix in .csv format.")
    parser.add_argument("--sigma-matrix", required=True, help="Path to Sigma matrix in .csv format.")
    parser.add_argument("--best-threshold", required=True, help="Text file containing the selected threshold.")
    parser.add_argument("--method", required=True, choices=["nnm", "nnm-sparse", "nnm-corr"], help="Clorinn model type.")
    parser.add_argument("--trait-set", required=True, help="Trait-set label, used only for metadata.")
    parser.add_argument("--maxiter", required=True, type=int, help="Maximum number of optimization iterations.")
    parser.add_argument("--model-out", required=True, help="Output pickle for the fitted model.")
    parser.add_argument("--lowrank-out", required=True, help="Output .npy for the final low-rank estimate on the original Z scale.")
    parser.add_argument("--pca-out", required=True, help="Output .npz for the final loadings and factors.")
    parser.add_argument("--metrics-out", required=True, help="Output JSON metadata file.")
    return parser.parse_args()


def read_threshold(path):
    with open(path, "r") as fh:
        value = fh.read().strip()
    if value == "":
        raise ValueError("best_threshold file is empty.")
    return float(value)


def main():

    # -- Snakemake interface ------
    zscore_data =       snakemake.input.zscore_data
    noise_error_data =  snakemake.input.noise_error_data
    best_threshold =    snakemake.input.
    method =            snakemake.params.method
    max_iter =          snakemake.params.max_iter
    svd_max_iter =      snakemake.params.svd_max_iter
    model_out =         snakemake.output.model_out
    lowrank_out =       snakemake.output.lowrank_out
    tsvd_out =          snakemake.output.tsvd_out
    metrics_out =       snakemake.output.metrics_out
    log_path =          snakemake.log[0] if snakemake.log else None
    logger =            setup_logger(Path(__file__).stem, log_path)

    raw_matrix_path = Path(args.raw_matrix)
    sigma_matrix_path = Path(args.sigma_matrix)
    if raw_matrix_path.suffix != ".npy":
        raise ValueError("raw_matrix must be a .npy file.")
    if sigma_matrix_path.suffix != ".npy":
        raise ValueError("sigma_matrix must be a .npy file.")

    Z = np.load(raw_matrix_path)
    Sigma = np.load(sigma_matrix_path)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square 2D matrix.")
    if Sigma.shape[0] != Z.shape[1]:
        raise ValueError("Sigma dimension must match the number of columns of Z.")

    if np.isnan(Z).any():
        raise NotImplementedError(
            "This generic final-fit script uses column whitening with Sigma and therefore "
            "requires a complete Z matrix. For Z with NaNs, replace this script with your "
            "masked Sigma-aware Clorinn implementation."
        )

    best_threshold = read_threshold(args.best_threshold)
    sigma_inv_sqrt, sigma_sqrt, evals, evals_clipped = get_sigma_inverse_sqrt(
        Sigma, args.sigma_eigen_floor
    )

    Z_whitened = Z @ sigma_inv_sqrt
    Z0 = np.zeros_like(Z_whitened)

    if args.method == "nnm":
        model = FrankWolfe(
            model="nnm",
            max_iter=args.maxiter,
            svd_max_iter=None,
            show_progress=True,
        )
        model.fit(Z_whitened, best_threshold, X0=Z0)
    elif args.method == "nnm-sparse":
        model = FrankWolfe(
            model="nnm-sparse",
            max_iter=args.maxiter,
            svd_max_iter=None,
            show_progress=True,
        )
        model.fit(Z_whitened, (best_threshold, 100.0), X0=Z0)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    X_whitened = model.X_
    X_original_scale = X_whitened @ sigma_sqrt

    model_dict = class_to_dict(model)
    model_dict["best_threshold_"] = best_threshold
    model_dict["sigma_matrix_file_"] = str(sigma_matrix_path)
    model_dict["sigma_eigenvalues_"] = evals
    model_dict["sigma_eigenvalues_clipped_"] = evals_clipped
    model_dict["sigma_inverse_sqrt_"] = sigma_inv_sqrt
    model_dict["sigma_sqrt_"] = sigma_sqrt
    model_dict["X_original_scale_"] = X_original_scale

    model_out = Path(args.model_out)
    lowrank_out = Path(args.lowrank_out)
    metrics_out = Path(args.metrics_out)
    threshold_out = Path(args.threshold_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    lowrank_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    threshold_out.parent.mkdir(parents=True, exist_ok=True)

    with open(model_out, "wb") as fh:
        pickle.dump(model_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(lowrank_out, X_original_scale)

    metrics = {
        "trait_set": args.trait_set,
        "method": args.method,
        "best_threshold": best_threshold,
        "max_iter": args.maxiter,
        "raw_matrix_file": str(raw_matrix_path),
        "sigma_matrix_file": str(sigma_matrix_path),
        "sigma_eigen_floor": args.sigma_eigen_floor,
        "min_sigma_eigenvalue": float(np.min(evals)),
        "min_sigma_eigenvalue_clipped": float(np.min(evals_clipped)),
        "model_file": str(model_out),
        "lowrank_file": str(lowrank_out),
        "n_rows": int(Z.shape[0]),
        "n_cols": int(Z.shape[1]),
    }
    with open(metrics_out, "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(threshold_out, "w") as fh:
        fh.write(f"{best_threshold:g}\n")

    print(f"Finished final full-data fit at nucnorm={best_threshold:g}")
    print(f"Saved low-rank estimate to: {lowrank_out}")


if __name__ == "__main__":
    main()
