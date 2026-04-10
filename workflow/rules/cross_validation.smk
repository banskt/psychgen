# =============================================================================
# Cross-validation pipeline
#
# Two-stage hyperparameter search:
#   1. Coarse  – geometrically-spaced nuclear-norm values (powers of 2)
#   2. Fine    – denser grid bracketing the coarse optimum (via checkpoint)
#
# =============================================================================

# -- Unpack config ------------------------------------------------------------
CV_RESULTS_DIR  = config["paths"]["cv_results_dir"]
METHOD          = config["cross_validation"]["method"]
MAX_ITER        = config["cross_validation"]["max_iter"]
SVD_MAX_ITER    = config["cross_validation"]["svd_max_iter"]
COARSE_NUCNORMS = [int(x) for x in config["cross_validation"]["nucnorm_coarse"]]

COARSE_CV_MODEL_FILES = expand(
    f"{CV_RESULTS_DIR}/models/{METHOD}_cv_model_r{{nucnorm}}.pkl",
    nucnorm=COARSE_NUCNORMS,
)

COARSE_CV_METRIC_FILES = expand(
    f"{CV_RESULTS_DIR}/metrics/{METHOD}_cv_metrics_r{{nucnorm}}.json",
    nucnorm=COARSE_NUCNORMS,
)

# -- Wildcard constraints -----------------------------------------------------
wildcard_constraints:
    nucnorm = r"\d+"

# -- Helper functions ---------------------------------------------------------
def _fine_nucnorms(wildcards):
    chk = checkpoints.find_fine_range.get()
    with open(chk.output.fine_nucnorms) as fh:
        return [line.strip() for line in fh if line.strip()]


def _fine_cv_metric_files(wildcards):
    return expand(
        f"{CV_RESULTS_DIR}/metrics/{METHOD}_cv_metrics_r{{nucnorm}}.json",
        nucnorm=_fine_nucnorms(wildcards),
    )


def _all_targets(wildcards):
    return (
        COARSE_CV_METRIC_FILES
        + _fine_cv_metric_files(wildcards)
        + [f"{CV_RESULTS_DIR}/summary/{METHOD}_cv_metrics.tsv",
		   f"{CV_RESULTS_DIR}/{METHOD}_fine_nucnorm_range.txt"
		  ]
    )


# -- Rules --------------------------------------------------------------------
rule all:
    input:
        _all_targets


rule create_cv_input:
    input:
        zscore_data = config["paths"]["zscore"],
    output:
        zmask_out   = config["paths"]["cv_input"],
    params:
        holdout_fraction = config["cross_validation"]["holdout_fraction"],
        seed             = config["cross_validation"]["mask_seed"],
    log:
        f"{CV_RESULTS_DIR}/logs/create_cv_input.log"
    resources:
        cpus_per_task = 1,
        mem_mb        = 16000,
        runtime       = 60,
    script:
        "../scripts/make_cv_input.py"


rule cross_validation:
    input:
        cv_input       = config["paths"]["cv_input"],
    output:
        cv_model_out   = f"{CV_RESULTS_DIR}/models/{METHOD}_cv_model_r{{nucnorm}}.pkl",
        cv_metrics_out = f"{CV_RESULTS_DIR}/metrics/{METHOD}_cv_metrics_r{{nucnorm}}.json",
    params:
        cv_method       = METHOD,
        cv_max_iter     = MAX_ITER,
        cv_svd_max_iter = SVD_MAX_ITER,
    log:
        f"{CV_RESULTS_DIR}/logs/{METHOD}_cv_model_r{{nucnorm}}.log",
    script:
        "../scripts/fit_cv_model.py"


checkpoint find_fine_range:
    input:
        cv_coarse_metrics = COARSE_CV_METRIC_FILES,
    output:
        fine_nucnorms     = f"{CV_RESULTS_DIR}/{METHOD}_fine_nucnorm_range.txt",
    params:
        n_points          = config["cross_validation"]["n_fine_points"],
    log:
        f"{CV_RESULTS_DIR}/logs/find_fine_range.log",
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/find_fine_grid_cv_nucnorms.py"


rule summarize_cv:
    default_target: True
    input:
        cv_metrics = lambda wildcards: COARSE_CV_METRIC_FILES + _fine_cv_metric_files(wildcards)
    output:
        summary_out        = f"{CV_RESULTS_DIR}/summary/{METHOD}_cv_metrics.tsv",
        best_threshold_out = f"{CV_RESULTS_DIR}/summary/{METHOD}_best_threshold.json",
    log:
        f"{CV_RESULTS_DIR}/logs/summarize_cv.log",
    resources:
        cpus_per_task = 1,
        mem_mb        = 8000,
        runtime       = 60,
    script:
        "../scripts/aggregate_cv_metrics.py"
