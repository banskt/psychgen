#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

from helpers import ensure_parent

def main():

    # -- Snakemake interface ------
    zscore_data      =  snakemake.input.zscore_data
    holdout_fraction =  snakemake.params.holdout_fraction
    seed =              snakemake.params.seed
    zmask_out =         snakemake.output.zmask_out

    # -- Read data ------
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout-fraction must be strictly between 0 and 1.")

    zscore_data_path = Path(zscore_data)
    if zscore_data_path.suffix != ".csv":
        raise ValueError("Input data must be a .csv file containing Z.")
    out_path = Path(zmask_out)
    if out_path.suffix != ".npz":
        raise ValueError("Intermediate file for CV input must be a .npz file.")

    Z = pd.read_csv(zscore_data_path, header = 0, index_col=0).to_numpy()
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D matrix.")

    # -- Create masked entry for CV training ------
    observed_mask = np.isfinite(Z)
    observed_indices = np.flatnonzero(observed_mask)
    n_observed = observed_indices.size
    if n_observed == 0:
        raise ValueError("Z has no observed entries.")

    n_holdout = int(np.floor(holdout_fraction * n_observed))
    if n_holdout == 0:
        raise ValueError("holdout-fraction is too small; no held-out entries were selected.")

    rng = np.random.default_rng(seed)
    heldout_flat = rng.choice(observed_indices, size=n_holdout, replace=False)

    Zmask = np.zeros_like(observed_mask, dtype=bool)
    Zmask.flat[heldout_flat] = True

    if np.any(~observed_mask & Zmask):
        raise RuntimeError("Held-out mask includes NaN entries from the original matrix.")

    Ztrain = Z.copy()
    Ztrain[Zmask] = np.nan

    ensure_parent(out_path)
    np.savez_compressed(out_path, Ztrue=Z, Ztrain=Ztrain, Zmask=Zmask)

    print(f"Saved CV input to: {out_path}")
    print(f"Matrix shape: {Z.shape}")
    print(f"Observed entries: {n_observed}")
    print(f"Held-out entries: {n_holdout}")
    print(f"Holdout fraction realized: {n_holdout / n_observed:.6f}")


if __name__ == "__main__":
    main()
