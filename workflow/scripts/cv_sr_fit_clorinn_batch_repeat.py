#!/usr/bin/env python
from pathlib import Path

from helpers import setup_logger, run_with_snakemake_log
from cv_sr_fit_clorinn_common import (
    load_split_input,
    load_zscore_and_noise,
    parse_fit_params,
    fit_one_repeat,
)


def main():
    log_path = snakemake.log[0] if snakemake.log else None
    logger   = setup_logger(Path(__file__).stem, log_path)

    cv_input       = snakemake.input.cv_input
    zscore_data    = snakemake.input.zscore_data
    noise_cov_data = snakemake.input.noise_cov_data

    nucnorm_full = float(snakemake.wildcards.nucnorm)
    repeat_ids   = [int(x) for x in snakemake.params.repeat_ids]

    model        = snakemake.params.cv_model
    solver       = snakemake.params.cv_solver
    sparse_scale = float(snakemake.params.cv_sparse_scale)
    fit_params   = parse_fit_params(snakemake.params)

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
    Z, noise_cov = load_zscore_and_noise(
        zscore_data,
        noise_cov_path = noise_cov_data,
        model = model
    )

    for repeat_id, fit_result_out, subspace_out, metrics_out in zip(
        repeat_ids, fit_result_paths, subspace_paths, metrics_paths,
    ):
        fit_one_repeat(
            Z              = Z,
            noise_cov      = noise_cov,
            assignments    = assignments,
            repeat_id      = repeat_id,
            eligible_cols  = eligible_cols,
            p_total        = p_total,
            n_folds        = n_folds,
            n_repeats      = n_repeats,
            model          = model,
            solver         = solver,
            nucnorm_full   = nucnorm_full,
            sparse_scale   = sparse_scale,
            fit_params     = fit_params,
            fit_result_out = fit_result_out,
            subspace_out   = subspace_out,
            metrics_out    = metrics_out,
            logger         = logger,
        )

    logger.info("Finished batch fit.")


if __name__ == "__main__":
    run_with_snakemake_log(main, snakemake)
