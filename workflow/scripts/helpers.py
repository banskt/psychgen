#!/usr/bin/env python

from pathlib import Path
from clorinn.optimize import FrankWolfe, ProjectedGradientDescent
from contextlib import redirect_stdout, redirect_stderr
import logging


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def clorinn_to_dict_(instance, property_list=None):
    payload = {}
    if property_list is None:
        _skip      = {"logger_", "nnm_"}
        property_list = [k for k in vars(instance).keys() if k not in _skip]
    for name in property_list:
        payload[name] = getattr(instance, name)
    return payload


def fit_clorinn(
    ztrain,
    nucnorm,
    sparse_scale=1.0,
    model="nnm",
    solver="fw",
    max_iter=1000,
    pgd_max_iter=3,
    svd_max_iter=None,
    svd_method='left-gram',
    tol=1e-3,
    step_tol=1e-4,
    rel_tol=1e-8,
    verbose=1,
    X0=None,
    noise_cov=None,
    mask=None
):
    """
    Fit a Clorinn model with solver

    Parameters
    ----------
    ztrain : ndarray
        Training matrix.
    nucnorm : float
        Nuclear norm threshold.
    sparse_scale : float
        L1 threshold used only for method="nnm-sparse".
    model : {"nnm", "nnm-sparse", "nnm-corr"}
        Model type in Clorinn.
    solver : {"fw", "pgd", "pgd-fw"}
        Solver type in Clorinn.
    max_iter, svd_max_iter, svd_method, tol, step_tol, rel_tol, verbose
        Passed to Clorinn. 
        For `svd_method`, Clorinn default is 'power', but here we
        use 'left-gram' because the data structure (only ~100 traits).
    pgd_max_iter : int, default=5
        Maximum number of PGD iterations
    X0 : ndarray or None
        Initial value for X0. If None, uses zeros_like(ztrain).
    noise_cov : ndarray
        noise covariance matrix
    mask : ndarray of bool
        True for entries to exclude.

    Returns
    -------
    result
        Fitted <Clorinn Solver>.result
    """

    fw_kwargs = dict(
        model=model,
        max_iter=max_iter,
        svd_max_iter=svd_max_iter,
        tol=tol,
        step_tol=step_tol,
        rel_tol=rel_tol,
        verbose=verbose,
        svd_method=svd_method,
        stop_criteria=("duality_gap",),
    )

    pgd_kwargs = dict(
        model=model,
        max_iter=pgd_max_iter,
        rel_tol=rel_tol,
        verbose=verbose,
        stop_criteria=("relative_loss",),
    )

    if model not in {"nnm", "nnm-sparse", "nnm-corr"}:
        raise ValueError(f"Unsupported model: {model}")

    if solver not in {"fw", "pgd", "pgd-fw"}:
        raise ValueError(f"Unsupported solver: {solver}")

    if solver == "fw": 
        clorinn = FrankWolfe(**fw_kwargs)

    if solver == "pgd":
        clorinn = ProjectedGradientDescent(**pgd_kwargs)

    if solver == "pgd-fw":
        pgd_kwargs.update(
            stop_criteria=("relative_loss",),  # ("boundary_active", "relative_loss",) 
        )
        clorinn = ProjectedGradientDescent(**pgd_kwargs)

    fit_kwargs = dict(
        radius=nucnorm,
        mask=mask,
        X0=X0,
    )

    if model == "nnm-sparse":
        fit_kwargs.update(sparse_scale=sparse_scale)

    if model == "nnm-corr":
        fit_kwargs.update(noise_cov=noise_cov)

    clorinn = clorinn.fit(ztrain, **fit_kwargs)

    if solver == "pgd-fw":
        clorinn_fw = FrankWolfe(**fw_kwargs)
        fit_kwargs.update(X0 = clorinn.result.X)
        clorinn_fw = clorinn_fw.fit(ztrain, **fit_kwargs)
        return clorinn_fw.result

    return clorinn.result


def run_with_snakemake_log(func, snakemake, *args, **kwargs):
    log_path = snakemake.log[0] if getattr(snakemake, "log", None) else None
    if log_path:
        with open(log_path, "w") as log_handle, \
             redirect_stdout(log_handle), \
             redirect_stderr(log_handle):
            return func(*args, **kwargs)
    else:
        return func(*args, **kwargs)



def setup_logger(name, log_path=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_path:
        ensure_parent(log_path)
        handler = logging.FileHandler(log_path, mode="w")
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
