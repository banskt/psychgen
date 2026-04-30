configfile: "config/config.yaml"
configfile: "config/paths.yaml"
configfile: "config/cross_validation.yaml"

# get_data_path for this pipeline
include: "workflow/rules/paths.smk"

# Matrix-completion CV
# include: "workflow/rules/cv_matrix_completion.smk"

# Split-replication CV
include: "workflow/rules/cv_split_replication.smk"
