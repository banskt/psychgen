# =============================================================================
# Split-replication CV — shared stability helpers
# =============================================================================
import json
import pandas as pd
 
 
def load_stability(path):
    """
    Load one stability JSON (coarse or fine) and return a flat record.
    """
    with open(path) as fh:
        stab = json.load(fh)
    return {
        "nucnorm_full":   float(stab["nucnorm_full"]),
        "k":              [entry["k"]                for entry in stab["by_k"]],
        "mean_dist":      [entry["mean_dist"]        for entry in stab["by_k"]],
        "se_dist":        [entry["se_dist"]          for entry in stab["by_k"]],
        "mean_gap_angle": [entry["mean_gap_angle"]   for entry in stab["by_k"]],
        "mean_energy":    [entry["mean_energy"]      for entry in stab["by_k"]],
        "se_energy":      [entry["se_energy"]        for entry in stab["by_k"]],
    }
 
 
def validate_k_ref(records):
    """
    Assert that all stability records share the same k grid.
    Returns the common k list.
 
    Raises ValueError on mismatch.
    """
    if not records:
        raise ValueError("No stability records provided.")
    k_ref = records[0]["k"]
    for rec in records[1:]:
        if rec["k"] != k_ref:
            raise ValueError(
                f"Inconsistent k grids across stability files: "
                f"{k_ref} vs {rec['k']}."
            )
    return k_ref
 
 
def build_stability_df(records, k_ref, extra_fields=None):
    """
    Build a wide (nucnorm × k) DataFrame from a list of stability records.
 
    Always includes mean_dist and se_dist columns.
    extra_fields: list of field names from each record to also expand,
    e.g. ["mean_gap_angle", "mean_energy", "se_energy"].
 
    Returns a DataFrame sorted by nucnorm, deduplicated on nucnorm.
    """
    extra_fields = extra_fields or []
    col_dict = {
        "nucnorm": [rec["nucnorm_full"] for rec in records],
        **{
            f"mean_dist_k{k}": [rec["mean_dist"][k_idx] for rec in records]
            for k_idx, k in enumerate(k_ref)
        },
        **{
            f"se_dist_k{k}": [rec["se_dist"][k_idx] for rec in records]
            for k_idx, k in enumerate(k_ref)
        },
    }
    for field in extra_fields:
        col_dict.update({
            f"{field}_k{k}": [rec[field][k_idx] for rec in records]
            for k_idx, k in enumerate(k_ref)
        })
 
    df = (pd.DataFrame(col_dict)
            .sort_values("nucnorm")
            .drop_duplicates(subset="nucnorm", keep="first")
            .reset_index(drop=True))
    return df
 
 
def choose_plateau_threshold(df, score_column, se_column,
                              one_se_multiplier, abs_tolerance, rel_tolerance):
    """
    Select the smallest nucnorm whose score is within tolerance of the
    minimum — i.e. the onset of the stability plateau.
 
    threshold = best_mean + one_se_multiplier * best_se
                          + abs_tolerance
                          + rel_tolerance * |best_mean|
 
    Among all nucnorms with score <= threshold, return the smallest
    (most parsimonious) one as the plateau onset.
 
    Returns
    -------
    selected : Series   row for the plateau onset nucnorm
    best_row : Series   row for the minimum nucnorm
    threshold : float
    """
    best_idx  = df[score_column].idxmin()
    best_row  = df.loc[best_idx]
    best_mean = float(best_row[score_column])
    best_se   = float(best_row[se_column])
 
    threshold = (best_mean
                 + one_se_multiplier * best_se
                 + abs_tolerance
                 + rel_tolerance * abs(best_mean))
 
    eligible = df.loc[df[score_column] <= threshold].copy()
    selected = (eligible.sort_values("nucnorm", ascending=True).iloc[0]
                if not eligible.empty else best_row)
 
    return selected, best_row, threshold
 
 
def log_stability_curve(logger, df, score_col, se_col, best_r, selected_r,
                         grid_col=None, header=None):
    """
    Log the full stability curve at one k, annotating the minimum and
    selected (plateau onset) nucnorm.
 
    Parameters
    ----------
    grid_col : str or None
        Column name for coarse/fine label.  If None, label is omitted.
    header : str or None
        First log line.  Defaults to "Stability curve:".
    """
    logger.info(header or "Stability curve:")
    for _, row in df.iterrows():
        tag = ""
        if row["nucnorm"] == best_r and row["nucnorm"] == selected_r:
            tag = "  <-- minimum / plateau onset (selected)"
        elif row["nucnorm"] == best_r:
            tag = "  <-- minimum"
        elif row["nucnorm"] == selected_r:
            tag = "  <-- plateau onset (selected)"
        grid_label = f"[{row[grid_col]:6s}]  " if grid_col else ""
        logger.info(
            f"  {grid_label}nucnorm={row['nucnorm']:.0f}  "
            f"mean_dist={row[score_col]:.6f} ± {row[se_col]:.6f}"
            f"{tag}"
        )
