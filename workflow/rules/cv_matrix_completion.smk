# =============================================================================
# Matrix completion cross-validation pipeline
#
# Two-stage hyperparameter search:
#   1. Coarse  – geometrically-spaced nuclear-norm values (powers of 2)
#   2. Fine    – denser grid bracketing the coarse optimum (via checkpoint)
#
# =============================================================================
# NOTE: CVMCPaths, CVMCLogs, CVMCConfig, and make_cv_matrix_completion_config
# should ideally live in a companion module (e.g. workflow/lib/cv_mc_config.py)
# so they can be unit-tested with a mock config dict outside Snakemake.
# They are kept here until that module exists.
#
# import sys
# import os
# sys.path.insert(0, os.path.join(workflow.basedir, "lib"))
# from cv_mc_config import CVMCPaths, CVMCLogs, CVMCConfig, make_cv_matrix_completion_config
#
# Insert at position 0 ensures local cv_mc_config is found 
# before any installed package that might share the same name.
# =============================================================================
from dataclasses import dataclass


# -- Config dataclasses -------------------------------------------------------

@dataclass(frozen=True)
class CVMCPaths:
    results_dir:        str
    zscore_data:        str
    cv_input:           str
    fine_nucnorm_range: str
    summary:            str
    best_threshold:     str
    fitresult_pattern:      str
    metrics_pattern:     str


@dataclass(frozen=True)
class CVMCLogs:
    create_input:    str
    fit_model:       str  # wildcard pattern — contains {nucnorm}
    find_fine_range: str
    summarize:       str


@dataclass(frozen=True)
class CVMCConfig:
    model:            str
    solver:           str
    prefix:           str
    max_iter:         int
    pgd_max_iter:     int
    svd_max_iter:     int
    svd_method:       str
    dg_tol:           float
    sparse_scale:     float
    coarse_nucnorms:  tuple
    n_fine_points:    int
    enabled:          bool
    holdout_fraction: float
    mask_seed:        int
    paths:            CVMCPaths
    logs:             CVMCLogs


def make_cv_matrix_completion_config(config, get_data_path) -> CVMCConfig:
    _cv   = config["cross_validation"]
    _cvmc = _cv["matrix_completion"]

    results_dir = get_data_path(config["paths"]["cv"]["matrix_completion_dir"])
    model       = str(_cv["model"])
    solver      = str(_cv["solver"])
    prefix      = f"{solver}_{model}".replace("-", "_")

    return CVMCConfig(
        model            = model,
        solver           = solver,
        prefix           = prefix,
        max_iter         = int(_cv["max_iter"]),
        pgd_max_iter     = int(_cv["pgd_max_iter"]),
        svd_max_iter     = int(_cv["svd_max_iter"]),
        svd_method       = str(_cv["svd_method"]),
        dg_tol           = float(_cv["dg_tol"]),
        sparse_scale     = float(_cv["sparse_scale"]),
        coarse_nucnorms  = tuple(int(x) for x in _cv["nucnorm_coarse"]),
        n_fine_points    = int(_cv["n_fine_points"]),
        enabled          = bool(_cvmc["enabled"]),
        holdout_fraction = float(_cvmc["holdout_fraction"]),
        mask_seed        = int(_cvmc["mask_seed"]),
        paths = CVMCPaths(
            results_dir        = results_dir,
            zscore_data        = get_data_path(config["paths"]["input"]["zscore"]),
            cv_input           = f"{results_dir}/cv_input/zscore_cv.npz",
            fine_nucnorm_range = f"{results_dir}/{prefix}_fine_nucnorm_range.txt",
            summary            = f"{results_dir}/summary/{prefix}_cv_metrics.csv",
            best_threshold     = f"{results_dir}/summary/{prefix}_best_threshold.json",
            fitresult_pattern  = f"{results_dir}/fit_result/{prefix}_r{{nucnorm}}.pkl",
            metrics_pattern    = f"{results_dir}/metrics/{prefix}_r{{nucnorm}}.json",
        ),
        logs = CVMCLogs(
            create_input    = f"{results_dir}/logs/{prefix}_create_input.log",
            fit_model       = f"{results_dir}/logs/{prefix}_r{{nucnorm}}.log",
            find_fine_range = f"{results_dir}/logs/{prefix}_find_fine_range.log",
            summarize       = f"{results_dir}/logs/{prefix}_summarize.log",
        ),
    )


# -- Instantiate --------------------------------------------------------------

CVMC = make_cv_matrix_completion_config(config, get_data_path)

# Snakemake-visible wildcard patterns — static strings exposed at module level
# so Snakemake can inspect them without calling a function.
CVMC_FITRESULT_PATTERN  = CVMC.paths.fitresult_pattern
CVMC_METRICS_PATTERN = CVMC.paths.metrics_pattern

# Precomputed coarse-grid file lists
CVMC_COARSE_NUCNORM_MODEL_FILES  = expand(CVMC_FITRESULT_PATTERN,  nucnorm=CVMC.coarse_nucnorms)
CVMC_COARSE_NUCNORM_METRIC_FILES = expand(CVMC_METRICS_PATTERN, nucnorm=CVMC.coarse_nucnorms)


# -- Wildcard constraints -----------------------------------------------------

wildcard_constraints:
    nucnorm = r"\d+"


# -- Helper functions ---------------------------------------------------------

def _cv_mc_fine_nucnorms(wildcards):
    chk = checkpoints.cv_mc_find_fine_grid_nucnorms.get()
    with open(chk.output.fine_nucnorms) as fh:
        return [line.strip() for line in fh if line.strip()]


def _cv_mc_fine_nucnorm_metric_files(wildcards):
    return expand(CVMC_METRICS_PATTERN, nucnorm=_cv_mc_fine_nucnorms(wildcards))


# -- Rules --------------------------------------------------------------------

rule cv_mc_all:
    input:
        CVMC.paths.summary,
        CVMC.paths.best_threshold

rule cv_mc_create_input:
    input:
        zscore_data      = CVMC.paths.zscore_data,
    output:
        zmask_out        = CVMC.paths.cv_input,
    params:
        holdout_fraction = CVMC.holdout_fraction,
        seed             = CVMC.mask_seed,
    log:
        CVMC.logs.create_input,
    resources:
        cpus_per_task = 1,
        mem_mb        = 16000,
        runtime       = 60,
    script:
        "../scripts/cv_mc_create_input.py"


rule cv_mc_fit_clorinn:
    input:
        cv_input         = CVMC.paths.cv_input,
    output:
        cv_fit_out       = CVMC_FITRESULT_PATTERN,
        cv_metrics_out   = CVMC_METRICS_PATTERN,
    params:
        cv_model         = CVMC.model,
        cv_solver        = CVMC.solver,
        cv_max_iter      = CVMC.max_iter,
        cv_pgd_max_iter  = CVMC.pgd_max_iter,
        cv_svd_max_iter  = CVMC.svd_max_iter,
        cv_svd_method    = CVMC.svd_method,
        cv_dg_tol        = CVMC.dg_tol,
        cv_sparse_scale  = CVMC.sparse_scale,
    log:
        CVMC.logs.fit_model,
    resources:
        cpus_per_task = 16,
        mem_mb        = 200000,
        runtime       = 4320,
    script:
        "../scripts/cv_mc_fit_clorinn.py"


checkpoint cv_mc_find_fine_grid_nucnorms:
    input:
        cv_coarse_metrics = CVMC_COARSE_NUCNORM_METRIC_FILES,
    output:
        fine_nucnorms     = CVMC.paths.fine_nucnorm_range,
    params:
        n_points          = CVMC.n_fine_points,
    log:
        CVMC.logs.find_fine_range,
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/cv_mc_find_fine_grid_nucnorms.py"


rule cv_mc_summarize:
    input:
        cv_metrics         = lambda wildcards: CVMC_COARSE_NUCNORM_METRIC_FILES + _cv_mc_fine_nucnorm_metric_files(wildcards),
    output:
        summary_out        = CVMC.paths.summary,
        best_threshold_out = CVMC.paths.best_threshold,
    log:
        CVMC.logs.summarize,
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/cv_mc_aggregate_metrics.py"
