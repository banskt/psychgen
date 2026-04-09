#!/usr/bin/env python

from pathlib import Path
from clorinn.optimize import FrankWolfe

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def clorinn_to_dict(instance, property_list=None):
    payload = {}
    if property_list is None:
        _skip      = {"logger_", "nnm_"}
        property_list = [k for k in vars(instance).keys() if k not in _skip]
    for name in property_list:
        payload[name] = getattr(instance, name)
    return payload


def fit_clorinn(
    ztrain,
    method,
    nucnorm,
    max_iter=1000,
    svd_max_iter=None,
    tol=1e-3,
    step_tol=1e-4,
    rel_tol=1e-8,
    show_progress=True,
    X0=None,
    sparse_l1=100.0,
    L_inv=None,
    Sigma_inv=None,
    **kwargs,
):
    """
    Fit a Clorinn Frank-Wolfe model.

    Parameters
    ----------
    ztrain : ndarray
        Training matrix.
    method : {"nnm", "nnm-sparse", "nnm-corr"}
        Model type.
    nucnorm : float
        Nuclear norm threshold.
    max_iter, svd_max_iter, tol, step_tol, rel_tol, show_progress
        Passed to FrankWolfe.
    X0 : ndarray or None
        Initial value for X0. If None, uses zeros_like(ztrain).
    sparse_l1 : float
        L1 threshold used only for method="nnm-sparse".
    L_inv : ndarray or None
        inverse of the Cholesky decomposition of Sigma, used only for method="nnm-corr".
    Sigma_inv : ndarray or None
        inverse of noise covariance Sigma, used only for method="nnm-corr".
    **kwargs
        Any additional keyword arguments passed to FrankWolfe(...).

    Returns
    -------
    model
        Fitted FrankWolfe model.
    """

    fw_kwargs = dict(
        model=method,
        max_iter=max_iter,
        svd_max_iter=svd_max_iter,
        tol=tol,
        step_tol=step_tol,
        rel_tol=rel_tol,
        show_progress=show_progress,
    )
    fw_kwargs.update(kwargs)

    if method not in {"nnm", "nnm-sparse", "nnm-corr"}:
        raise ValueError(f"Unsupported method: {method}")

    model = FrankWolfe(**fw_kwargs)

    if method == "nnm" or "nnm-corr":
        thres_arg = nucnorm
    else:  # nnm-sparse
        thres_arg = (nucnorm, sparse_l1)

    model.fit(ztrain, thres_arg, X0=X0, Sigma_inv=Sigma_inv, L_inv=L_inv)

    return model
