#!/usr/bin/env python

import json
import pandas as pd
import numpy as np
from pathlib import Path

from helpers import ensure_parent, setup_logger

def main():

    # -- Snakemake interface ------

    cv_metrics =            snakemake.input.cv_coarse_metrics
    fine_nucnorms_out =     snakemake.output.fine_nucnorms
    n_points =              snakemake.params.n_points
    log_path =              snakemake.log[0] if snakemake.log else None
    logger =                setup_logger(Path(__file__).stem, log_path)

    # -- Read data ------
    rows = []
    for metric_file in cv_metrics:
        with open(metric_file) as handle:
            rows.append(json.load(handle))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No metric files were provided.")

    # -- Find the threshold ------
    df = df.sort_values("nucnorm").reset_index(drop=True)
    best_row = df.sort_values(["heldout_mse", "nucnorm"], ascending=[True, True]).iloc[0]
    best_threshold = best_row["nucnorm"]

    # -- Raise error if the best threshold is at the boundary ------
    best_idx = df.index[df["nucnorm"] == best_threshold][0]
    if best_idx == 0:
        raise ValueError("Best nucnorm is the smallest value; no lower neighbor available.")
    if best_idx == len(df) - 1:
        raise ValueError("Best nucnorm is the largest value; no upper neighbor available.")

    # -- Find the fine grid ------
    lo = df.loc[best_idx - 1, "nucnorm"]
    hi = df.loc[best_idx + 1, "nucnorm"]
    mid = best_threshold

    # a simple geometric spacing
    # fine = np.geomspace(lo, hi, num=n_points + 2)[1:-1]
    # fine_grid = sorted(set(int(round(v)) for v in fine))

    # a sophisticated power-of-2 grid
    n_points_each_side = n_points // 2

    left_exp = np.linspace(np.log2(lo), np.log2(mid), n_points_each_side + 2)[1:-1]
    right_exp = np.linspace(np.log2(mid), np.log2(hi), n_points_each_side + 2)[1:-1]

    fine = np.concatenate([2 ** left_exp, 2 ** right_exp])
    fine_grid = sorted(set(int(round(v)) for v in fine))

    # -- Save results ------
    ensure_parent(fine_nucnorms_out)
    with open(fine_nucnorms_out, "w") as fh:
        for v in fine_grid:
            fh.write(f"{v}\n")

    logger.info(f"Coarse optimum : {best_threshold:.0f}")
    logger.info(f"MSE at optimum : {best_row['heldout_mse']:.6f}")
    logger.info(f"Fine bracket   : [{lo:.0f}, {hi:.0f}]")
    logger.info(f"Fine grid      : {fine_grid}")


if __name__ == "__main__":
    main()
