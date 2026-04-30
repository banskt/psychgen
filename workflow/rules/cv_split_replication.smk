# =============================================================================
# Split-replication cross-validation pipeline
#
# Assesses stability of the inferred trait subspace U across SNP splits,
# rather than held-out entrywise prediction error.
#
# Two-stage hyperparameter search:
#   1. Coarse  – geometrically-spaced nuclear-norm values (powers of 2)
#   2. Fine    – denser grid bracketing the coarse stability plateau (via checkpoint)
#
# Pipeline stages:
#   create_input        – partition SNPs into (n_repeats × n_folds) folds once
#   fit_clorinn         – fit n_folds models per (nucnorm, repeat_id); scale r by sqrt(f)
#   aggregate_stability – compute pairwise projection distances for one nucnorm
#   find_fine_range     – checkpoint: identify fine grid from coarse stability plateau
#   summarize           – collect all stability metrics; write summary and best threshold
#
# =============================================================================
# NOTE: CVSRPaths, CVSRLogs, CVSRConfig, and make_cv_split_replication_config
# should ideally live in a companion module (e.g. workflow/lib/cv_sr_config.py)
# so they can be unit-tested with a mock config dict outside Snakemake.
# They are kept here until that module exists.
#
# import sys
# import os
# sys.path.insert(0, os.path.join(workflow.basedir, "lib"))
# from cv_sr_config import CVSRPaths, CVSRLogs, CVSRConfig, make_cv_split_replication_config
#
# Insert at position 0 ensures local cv_sr_config is found
# before any installed package that might share the same name.
# =============================================================================
from dataclasses import dataclass


# -- Config dataclasses -------------------------------------------------------

@dataclass(frozen=True)
class CVSRPaths:
    results_dir:         str
    zscore_data:         str
    cv_input:            str
    fine_nucnorm_range:  str
    summary:             str
    best_threshold:      str
    fitresult_pattern:   str  # wildcard pattern — contains {nucnorm}, {repeat_id}
    subspace_pattern:    str  # wildcard pattern — contains {nucnorm}, {repeat_id}
    fit_metrics_pattern: str  # wildcard pattern — contains {nucnorm}, {repeat_id}
    stability_pattern:   str  # wildcard pattern — contains {nucnorm}


@dataclass(frozen=True)
class CVSRLogs:
    create_input:        str
    fit_model:           str  # wildcard pattern — contains {nucnorm}, {repeat_id}
    fit_model_batch:     str  # wildcard pattern — contains {nucnorm}
    aggregate_stability: str  # wildcard pattern — contains {nucnorm}
    find_fine_range:     str
    summarize:           str


@dataclass(frozen=True)
class CVSRConfig:
    model:             str
    solver:            str
    prefix:            str
    max_iter:          int
    pgd_max_iter:      int
    svd_max_iter:      int
    svd_method:        str
    dg_tol:            float
    sparse_scale:      float
    coarse_nucnorms:   tuple
    n_fine_points:     int
    enabled:           bool
    n_repeats:         int
    n_folds:           int
    split_seed:        int
    n_factors:         int
    k_pivot:           int
    one_se_multiplier: float
    abs_tolerance:     float
    rel_tolerance:     float
    paths:             CVSRPaths
    logs:              CVSRLogs


def make_cv_split_replication_config(config, get_data_path) -> CVSRConfig:
    _cv   = config["cross_validation"]
    _cvsr = _cv["split_replication"]

    results_dir = get_data_path(config["paths"]["cv"]["split_replication_dir"])
    model       = str(_cv["model"])
    solver      = str(_cv["solver"])
    prefix      = f"{solver}_{model}".replace("-", "_")

    return CVSRConfig(
        model             = model,
        solver            = solver,
        prefix            = prefix,
        max_iter          = int(_cv["max_iter"]),
        pgd_max_iter      = int(_cv["pgd_max_iter"]),
        svd_max_iter      = int(_cv["svd_max_iter"]),
        svd_method        = str(_cv["svd_method"]),
        dg_tol            = float(_cv["dg_tol"]),
        sparse_scale      = float(_cv["sparse_scale"]),
        coarse_nucnorms   = tuple(int(x) for x in _cv["nucnorm_coarse"]),
        n_fine_points     = int(_cv["n_fine_points"]),
        enabled           = bool(_cvsr["enabled"]),
        n_repeats         = int(_cvsr["n_repeats"]),
        n_folds           = int(_cvsr["n_folds"]),
        split_seed        = int(_cvsr["split_seed"]),
        n_factors         = int(_cvsr["n_factors"]),
        k_pivot           = int(_cvsr["k_pivot"]),
        one_se_multiplier = float(_cvsr["selection"]["one_se_multiplier"]),
        abs_tolerance     = float(_cvsr["selection"]["abs_tolerance"]),
        rel_tolerance     = float(_cvsr["selection"]["rel_tolerance"]),
        paths = CVSRPaths(
            results_dir         = results_dir,
            zscore_data         = get_data_path(config["paths"]["input"]["zscore"]),
            cv_input            = f"{results_dir}/cv_input/split_assignments.npz",
            fine_nucnorm_range  = f"{results_dir}/{prefix}_fine_nucnorm_range.txt",
            summary             = f"{results_dir}/summary/{prefix}_stability_metrics.csv",
            best_threshold      = f"{results_dir}/summary/{prefix}_best_threshold.json",
            fitresult_pattern   = f"{results_dir}/fit_result/{prefix}_r{{nucnorm}}_s{{repeat_id}}.pkl",
            subspace_pattern    = f"{results_dir}/subspace/{prefix}_r{{nucnorm}}_s{{repeat_id}}.npz",
            fit_metrics_pattern = f"{results_dir}/fit_metrics/{prefix}_r{{nucnorm}}_s{{repeat_id}}.json",
            stability_pattern   = f"{results_dir}/stability/{prefix}_r{{nucnorm}}.json",
        ),
        logs = CVSRLogs(
            create_input        = f"{results_dir}/logs/{prefix}_create_input.log",
            fit_model           = f"{results_dir}/logs/{prefix}_r{{nucnorm}}_s{{repeat_id}}.log",
			fit_model_batch     = f"{results_dir}/logs/{prefix}_r{{nucnorm}}_all_repeats.log",
            aggregate_stability = f"{results_dir}/logs/{prefix}_stability_r{{nucnorm}}.log",
            find_fine_range     = f"{results_dir}/logs/{prefix}_find_fine_range.log",
            summarize           = f"{results_dir}/logs/{prefix}_summarize.log",
        ),
    )


# -- Instantiate --------------------------------------------------------------

CVSR = make_cv_split_replication_config(config, get_data_path)

# Snakemake-visible wildcard patterns — static strings exposed at module level
# so Snakemake can inspect them without calling a function.
CVSR_FITRESULT_PATTERN   = CVSR.paths.fitresult_pattern
CVSR_SUBSPACE_PATTERN    = CVSR.paths.subspace_pattern
CVSR_FIT_METRICS_PATTERN = CVSR.paths.fit_metrics_pattern
CVSR_STABILITY_PATTERN   = CVSR.paths.stability_pattern

# Index lists — resolved once at load time from config
CVSR_REPEAT_IDS = list(range(CVSR.n_repeats))

# Precomputed coarse-grid file lists
CVSR_COARSE_NUCNORM_FITRESULT_FILES = expand(
        CVSR_FITRESULT_PATTERN, nucnorm=CVSR.coarse_nucnorms, repeat_id=CVSR_REPEAT_IDS)
CVSR_COARSE_NUCNORM_SUBSPACE_FILES  = expand(
        CVSR_SUBSPACE_PATTERN, nucnorm=CVSR.coarse_nucnorms, repeat_id=CVSR_REPEAT_IDS)
CVSR_COARSE_NUCNORM_STABILITY_FILES = expand(
        CVSR_STABILITY_PATTERN, nucnorm=CVSR.coarse_nucnorms)


# -- Wildcard constraints -----------------------------------------------------

wildcard_constraints:
    nucnorm  = r"\d+",
    repeat_id = r"\d+",


# -- Helper functions ---------------------------------------------------------

def _cv_sr_fine_nucnorms(wildcards):
    chk = checkpoints.cv_sr_find_fine_grid_nucnorms.get()
    with open(chk.output.fine_nucnorms) as fh:
        return [line.strip() for line in fh if line.strip()]


def _cv_sr_fine_nucnorm_stability_files(wildcards):
    return expand(CVSR_STABILITY_PATTERN, nucnorm=_cv_sr_fine_nucnorms(wildcards))


# -- Rules --------------------------------------------------------------------

rule cv_sr_all:
    input:
        CVSR.paths.summary,
        CVSR.paths.best_threshold


rule cv_sr_create_input:
    input:
        zscore_data = CVSR.paths.zscore_data,
    output:
        split_input = CVSR.paths.cv_input,
    params:
        n_repeats   = CVSR.n_repeats,
        n_folds     = CVSR.n_folds,
        seed        = CVSR.split_seed,
    log:
        CVSR.logs.create_input,
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/cv_sr_create_input.py"


#rule cv_sr_fit_clorinn:
#    input:
#        cv_input    = CVSR.paths.cv_input,
#        zscore_data = CVSR.paths.zscore_data,
#    output:
#        fit_result  = CVSR_FITRESULT_PATTERN,
#        subspace    = CVSR_SUBSPACE_PATTERN,
#        fit_metrics = CVSR_FIT_METRICS_PATTERN,
#    params:
#        cv_model        = CVSR.model,
#        cv_solver       = CVSR.solver,
#        cv_max_iter     = CVSR.max_iter,
#        cv_pgd_max_iter = CVSR.pgd_max_iter,
#        cv_svd_max_iter = CVSR.svd_max_iter,
#        cv_svd_method   = CVSR.svd_method,
#        cv_dg_tol       = CVSR.dg_tol,
#        cv_sparse_scale = CVSR.sparse_scale,
#    log:
#        CVSR.logs.fit_model,
#    resources:
#        cpus_per_task = 4,
#		# realistic peak is around ~500MB.
#		# 107 × 46,000 = 4,922,000 entries
#		# 4,922,000 × 8 bytes = 39.4 MB ≈ 37.6 MiB
#		# For FW, copies and solver temporaries should require around 13x = 500MB.
#		# Add pandas CSV parsing overhead, Python imports, NumPy/SciPy/sklearn, 
#		# BLAS/LAPACK workspace, logging, pickling, and allocator fragmentation. 
#		# Real peak is probably more like 2GB.
#		# NYGC comes with ~8000MB per core, but let's be aggresive.
#        mem_mb        = 8000, 
#        runtime       = 60,
#    script:
#        "../scripts/cv_sr_fit_clorinn.py"


rule cv_sr_fit_clorinn_batch:
    input:
        cv_input    = CVSR.paths.cv_input,
        zscore_data = CVSR.paths.zscore_data,
    output:
        fit_result = expand(
            CVSR_FITRESULT_PATTERN,
            nucnorm="{nucnorm}",
            repeat_id=CVSR_REPEAT_IDS,
        ),
        subspace = expand(
            CVSR_SUBSPACE_PATTERN,
            nucnorm="{nucnorm}",
            repeat_id=CVSR_REPEAT_IDS,
        ),
        fit_metrics = expand(
            CVSR_FIT_METRICS_PATTERN,
            nucnorm="{nucnorm}",
            repeat_id=CVSR_REPEAT_IDS,
        ),
    params:
        repeat_ids      = CVSR_REPEAT_IDS,
        cv_model        = CVSR.model,
        cv_solver       = CVSR.solver,
        cv_max_iter     = CVSR.max_iter,
        cv_pgd_max_iter = CVSR.pgd_max_iter,
        cv_svd_max_iter = CVSR.svd_max_iter,
        cv_svd_method   = CVSR.svd_method,
        cv_dg_tol       = CVSR.dg_tol,
        cv_sparse_scale = CVSR.sparse_scale,
    log:
        CVSR.logs.fit_model_batch,
    resources:
        cpus_per_task = 4,
        mem_mb        = 4000,
        runtime       = 120,
    script:
        "../scripts/cv_sr_fit_clorinn_batch.py"


rule cv_sr_aggregate_stability:
    input:
        subspaces = expand(CVSR_SUBSPACE_PATTERN,
                           repeat_id=CVSR_REPEAT_IDS,
                           allow_missing=True),
    output:
        stability = CVSR_STABILITY_PATTERN,
    params:
        n_factors = CVSR.n_factors,
        n_repeats = CVSR.n_repeats,
        n_folds   = CVSR.n_folds,
    log:
        CVSR.logs.aggregate_stability,
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/cv_sr_aggregate_stability.py"


checkpoint cv_sr_find_fine_grid_nucnorms:
    input:
        coarse_stability = CVSR_COARSE_NUCNORM_STABILITY_FILES,
    output:
        fine_nucnorms    = CVSR.paths.fine_nucnorm_range,
    params:
        n_points          = CVSR.n_fine_points,
        k_pivot           = CVSR.k_pivot,
        one_se_multiplier = CVSR.one_se_multiplier,
        abs_tolerance     = CVSR.abs_tolerance,
        rel_tolerance     = CVSR.rel_tolerance,
    log:
        CVSR.logs.find_fine_range,
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/cv_sr_find_fine_grid_nucnorms.py"


rule cv_sr_summarize:
    input:
        stability          = lambda wildcards: CVSR_COARSE_NUCNORM_STABILITY_FILES + _cv_sr_fine_nucnorm_stability_files(wildcards),
        fine_nucnorms      = CVSR.paths.fine_nucnorm_range,
    output:
        summary_out        = CVSR.paths.summary,
        best_threshold_out = CVSR.paths.best_threshold,
    params:
        n_factors         = CVSR.n_factors,
        k_pivot           = CVSR.k_pivot,
        one_se_multiplier = CVSR.one_se_multiplier,
        abs_tolerance     = CVSR.abs_tolerance,
        rel_tolerance     = CVSR.rel_tolerance,
    log:
        CVSR.logs.summarize,
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/cv_sr_summarize.py"
