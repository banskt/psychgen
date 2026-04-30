#!/usr/bin/env python

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from helpers import ensure_parent, setup_logger


def load_subspace(path):
    with np.load(path, allow_pickle=False) as data:
        n_folds      = int(data["n_folds"])
        repeat_id    = int(data["repeat_id"])
        nucnorm_full = float(data["nucnorm_full"])
        U_folds = [data[f"U_f{f}"] for f in range(n_folds)]
        s_folds = [data[f"s_f{f}"] for f in range(n_folds)]
    return U_folds, s_folds, repeat_id, nucnorm_full, n_folds


def chordal_subspace_distance(U1, U2, k):
    """
    Chordal subspace distance between the leading k-dimensional subspaces
    spanned by the first k columns of U1 and U2.

    For two orthonormal bases Qa = Ua[:, :k] and Qb = Ub[:, :k]:

        d(Qa, Qb) = sqrt(k - ||Qa^T Qb||_F^2)

    This equals sqrt(sum of squared sines of the k principal angles).
    d = 0 when the subspaces are identical; d = sqrt(k) when they are
    fully orthogonal.

    Note: ||Qa Qa^T - Qb Qb^T||_F = sqrt(2) * chordal_distance

    Parameters
    ----------
    U1, U2 : ndarray, shape (N, >= k)
        Matrices of left singular vectors (columns are orthonormal).
    k : int
        Number of leading factors to include.

    Returns
    -------
    float
    """
    Q1   = U1[:, :k]
    Q2   = U2[:, :k]
    gram = Q1.T @ Q2                           # (k, k)
    sq   = np.sum(gram ** 2)                   # ||Qa^T Qb||_F^2
    # clamp to [0, k] before sqrt to guard against tiny floating-point overshoot
    chordal = np.sqrt(np.clip(k - sq, 0.0, k))
    return float(chordal)


def projection_distance(U1, U2, k):
    """
    Normalized projection distance returns a value in [0,1]
    0 = identical k-dimensional subspaces
    1 = fully orthogonal k-dimensional subspaces
    
    For orthonormal bases Qa, Qb:

        ||Qa Qa^T - Qb Qb^T||_F^2 
        = 2k - 2 ||Qa^T Qb||_F^2 
        = 2(k - ||Qa^T Qb||_F^2)

    Therefore,

        ||Qa Qa^T - Qb Qb^T||_F / sqrt(2k) = chordal_distance / sqrt(k)

    """
    return float(chordal_subspace_distance(U1, U2, k) / np.sqrt(k))


def principal_angles(U1, U2, k):
    Q1 = U1[:, :k]
    Q2 = U2[:, :k]
    M  = Q1.T @ Q2
    s  = np.linalg.svd(M, compute_uv=False)
    s  = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return angles


def main():

    # -- Snakemake interface --------------------------------------------------
    subspace_paths = list(snakemake.input.subspaces)
    stability_out  = snakemake.output.stability
    nucnorm_full   = float(snakemake.wildcards.nucnorm)
    n_factors      = int(snakemake.params.n_factors)
    n_repeats      = int(snakemake.params.n_repeats)
    n_folds        = int(snakemake.params.n_folds)
    log_path       = snakemake.log[0] if snakemake.log else None
    logger         = setup_logger(Path(__file__).stem, log_path)

    if len(subspace_paths) != n_repeats:
        raise ValueError(
            f"Expected {n_repeats} subspace files, got {len(subspace_paths)}."
        )

    # -- Load all subspaces ---------------------------------------------------
    # Group by repeat: repeats[repeat_id] = list of (fold_id, U, s)
    # sorted by repeat_id for deterministic pair labelling.
    repeats = {}    # repeat_id -> list of (fold_id, U, s)

    for path in subspace_paths:
        U_folds, s_folds, repeat_id, r_full, n_folds_file = load_subspace(path)

        if not np.isclose(r_full, nucnorm_full):
            raise ValueError(
                f"Subspace file {path} has nucnorm_full={r_full} "
                f"but wildcard nucnorm={nucnorm_full}."
            )
        if n_folds_file != n_folds:
            raise ValueError(
                f"Subspace file {path} has n_folds={n_folds_file} "
                f"but params n_folds={n_folds}."
            )
        if repeat_id in repeats:
            raise RuntimeError(
                f"Duplicate subspace file for repeat_id={repeat_id}."
            )

        for fold_id, U in enumerate(U_folds):
            if U.shape[1] < n_factors:
                raise ValueError(
                    f"U for repeat {repeat_id} fold {fold_id} has only "
                    f"{U.shape[1]} columns but n_factors={n_factors}."
                )
        repeats[repeat_id] = [
            (fold_id, U, s)
            for fold_id, (U, s) in enumerate(zip(U_folds, s_folds))
        ]

    if len(repeats) != n_repeats:
        raise RuntimeError(
            f"Loaded {len(repeats)} unique repeats, expected {n_repeats}."
        )

    expected_repeats = list(range(n_repeats))
    observed_repeats = sorted(repeats.keys())
    if observed_repeats != expected_repeats:
        raise RuntimeError(
            f"Expected repeat IDs {expected_repeats}, got {observed_repeats}."
        )

    # -- Build within-repeat pairs --------------------------------------------
    # For n_folds=2: one pair per repeat (fold 0 vs fold 1).
    # For n_folds=3: C(3,2)=3 pairs per repeat.
    n_fold_pairs = n_folds * (n_folds - 1) // 2    # C(n_folds, 2)
    n_pairs      = n_repeats * n_fold_pairs

    pair_labels = []    # str, length n_pairs
    pair_U      = []    # list of (U_a, U_b) tuples, length n_pairs

    for repeat_id in sorted(repeats.keys()):
        folds = repeats[repeat_id]           # list of (fold_id, U, s)
        for (fi, U_i, _), (fj, U_j, _) in combinations(folds, 2):
            pair_labels.append(f"rep{repeat_id}_f{fi}-rep{repeat_id}_f{fj}")
            pair_U.append((U_i, U_j))

    if len(pair_U) != n_pairs:
        raise RuntimeError(
            f"Built {len(pair_U)} pairs, expected {n_pairs}."
        )

    logger.info(f"nucnorm_full  : {nucnorm_full:g}")
    logger.info(f"n_repeats     : {n_repeats}")
    logger.info(f"n_folds       : {n_folds}")
    logger.info(f"n_fold_pairs  : {n_fold_pairs}  (= C({n_folds}, 2), within-repeat)")
    logger.info(f"n_pairs total : {n_pairs}  (= n_repeats × n_fold_pairs)")
    logger.info(f"n_factors     : {n_factors}")

    # -- Compute pairwise projection distances for k = 1..n_factors ----------
    dist_matrix = np.empty((n_factors, n_pairs), dtype=np.float64)

    # Use principal angles for diagnostics
    mean_pa_matrix = np.empty((n_factors, n_pairs), dtype=np.float64)
    # largest principal angle - gap angle -  measures how far the two subspaces 
    # are from being aligned along their hardest direction.
    gap_angle_matrix = np.empty((n_factors, n_pairs), dtype=np.float64)

    for k_idx, k in enumerate(range(1, n_factors + 1)):
        for pair_idx, (U_a, U_b) in enumerate(pair_U):
            dist_matrix[k_idx, pair_idx] = projection_distance(U_a, U_b, k)
            angles = principal_angles(U_a, U_b, k)
            mean_pa_matrix[k_idx, pair_idx] = float(np.mean(angles))
            gap_angle_matrix[k_idx, pair_idx] = float(np.max(angles))

    # -- Summarise per k ------------------------------------------------------
    mean_dist = dist_matrix.mean(axis=1)                          # (n_factors,)
    if n_pairs > 1:
        se_dist = dist_matrix.std(axis=1, ddof=1) / np.sqrt(n_pairs)
    else:
        se_dist = np.full(n_factors, np.nan)
    mean_gap_angle = gap_angle_matrix.mean(axis=1)

    # -- Compute energy for each subspace for k = 1..n_factors ----------------
    # Build flat list of all (repeat_id, fold_id, s) in deterministic order
    # n_matrices = n_repeats × n_folds, independent of n_pairs
    mat_labels = []
    mat_s      = []
    for repeat_id in sorted(repeats.keys()):
        for fold_id, U, s in repeats[repeat_id]:
            mat_labels.append(f"rep{repeat_id}_f{fold_id}")
            mat_s.append(s)

    n_matrices    = len(mat_s)   # = n_repeats * n_folds

    # energy_matrix[k_idx, mat_idx] = cumulative explained variance at k
    energy_matrix = np.empty((n_factors, n_matrices), dtype=np.float64)
    for mat_idx, s in enumerate(mat_s):
        s2  = s ** 2
        cum = np.cumsum(s2) / np.sum(s2)           # shape (N,)
        for k_idx, k in enumerate(range(1, n_factors + 1)):
            energy_matrix[k_idx, mat_idx] = float(cum[k - 1])

    mean_energy = energy_matrix.mean(axis=1)       # (n_factors,)
    se_energy   = energy_matrix.std(axis=1, ddof=1) / np.sqrt(n_matrices)

    # -- Build output ---------------------------------------------------------
    stability = {
        "nucnorm_full":  nucnorm_full,
        "n_factors":     n_factors,
        "n_repeats":     n_repeats,
        "n_folds":       n_folds,
        "n_fold_pairs":  n_fold_pairs,
        "n_pairs":       n_pairs,       # for distance: one value per within-repeat pair
        "n_matrices":    n_matrices,    # for energy: one value per fitted model
        "pair_labels":   pair_labels,
        "mat_labels":    mat_labels,
        "by_k": [
            {
                "k":              k,
                "mean_dist":      float(mean_dist[k_idx]),
                "se_dist":        float(se_dist[k_idx]),
                "distances":      dist_matrix[k_idx].tolist(),
                "mean_gap_angle": float(mean_gap_angle[k_idx]), # gap angle averaged over all pairs
                "gap_angles":     gap_angle_matrix[k_idx].tolist(), # gap angle for each pair
                "pair_mean_pa":   mean_pa_matrix[k_idx].tolist(), # mean principal angle for each pair
                "mean_energy":    float(mean_energy[k_idx]),
                "se_energy":      float(se_energy[k_idx]),
                "energies":       energy_matrix[k_idx].tolist(), # length n_matrices, not n_pairs
            }
            for k_idx, k in enumerate(range(1, n_factors + 1))
        ],
    }

    # -- Save -----------------------------------------------------------------
    ensure_parent(stability_out)
    with open(stability_out, "w") as fh:
        json.dump(stability, fh, indent=2, sort_keys=False)

    logger.info(f"Saved stability metrics to: {stability_out}")
    logger.info(
        f"Mean projection distance at k=1      : "
        f"{mean_dist[0]:.6f} ± {se_dist[0]:.6f}"
    )
    logger.info(
        f"Mean projection distance at k={n_factors:<3d}    : "
        f"{mean_dist[-1]:.6f} ± {se_dist[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
