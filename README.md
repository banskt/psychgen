# Analysis of Neuropsychiatric Disorders

This repository contains a Snakemake workflow for analyzing neuropsychiatric traits from PGC and UK Biobank.

The repository is code-only. Input matrices, logs, fitted models, and downstream outputs live on shared storage outside the repository and are referenced through `config/config.yaml`.

## Setup

Copy the example path file and edit the shared-storage paths.

```bash
cp config/paths.example.yaml config/paths.yaml
```
The profile `nygc.yaml` provides a SLURM executor. Edit the executor as required.

## Dry run

```bash
snakemake -n --profile profiles/nygc.yaml
```

## Run

```bash
snakemake --profile profiles/nygc.yaml --jobs 50
```

## Notes

- The profile assumes Snakemake's SLURM executor plugin model.
