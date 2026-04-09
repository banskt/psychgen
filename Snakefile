import yaml

configfile: "config/config.yaml"

with open("config/paths.yaml") as f:
    path_config = yaml.safe_load(f)
config.update(path_config)

include: "workflow/rules/cross_validation.smk"
