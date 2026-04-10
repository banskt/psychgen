#!/usr/bin/env python

import json
import pandas as pd

from helpers import ensure_parent, setup_logger

def main():

    # -- Snakemake interface ------
    cv_metrics =            snakemake.input.cv_metrics
    summary_out =           snakemake.output.summary_out
    best_threshold_out =    snakemake.output.best_threshold_out
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

    # -- Save results ------
    ensure_parent(summary_out)
    ensure_parent(best_threshold_out)

    df.to_csv(summary_out, index=False)
    with open(best_threshold_out, "w") as handle:
        json.dump({"best_threshold": float(best_threshold)}, handle)


if __name__ == "__main__":
    main()
