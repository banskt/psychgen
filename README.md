# Analysis of Neuropsychiatric Disorders

This repository contains a Snakemake workflow for analyzing neuropsychiatric traits from PGC and UK Biobank.

The repository is code-only. Input matrices, logs, fitted models, and downstream outputs live on shared storage outside the repository and are referenced through `config/config.yaml`.

## Setup

Copy the example path file and edit the shared-storage paths.

```bash
cp config/paths.example.yaml config/paths.yaml
```

## Dry run

```bash
snakemake -n --profile profiles/nygc
```

## Run

```bash
snakemake --profile profiles/slurm --jobs 50
```

## Notes

- The profile assumes Snakemake's SLURM executor plugin model.
- The rule-level resource blocks carry your original intent from the `sbatch` template: 16 CPUs, 200 GB memory, 7-day walltime for model fitting.
