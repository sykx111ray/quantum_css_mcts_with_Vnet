# GitHub Release Checklist

## README
- A README is currently missing from the repository root.
- The README should explain:
  - project purpose
  - core dependencies
  - how to run the primary experiments
  - expected output files
  - how to reproduce the main release experiments

## Requirements
- A requirements file is not present at the repository root.
- The project should include a dependency list for:
  - torch
  - numpy
  - scipy
  - networkx
  - python-docx (if used by reporting scripts)
  - matplotlib (if plotting scripts are part of the release bundle)

## Entry points
- The main experiment entry points are present as Python scripts.
- However, the entry points should be documented clearly in the README for a new user.
- Suggested release entry points:
  - experiment_14_rowcol_baseline.py
  - experiment_17_rowcol_rank.py
  - experiment_20_loss_ablation.py
  - experiment_22_value_vs_baseline.py

## Missing release assets
- LICENSE: missing
- .gitignore: missing

## Recommended release actions
- Add a root README.
- Add requirements.txt.
- Add LICENSE.
- Add .gitignore covering:
  - __pycache__/
  - *.pyc
  - results/*.png and temporary outputs if desired
  - checkpoints/ if release should be source-only
  - .venv/

## Current release readiness
- Codebase is structurally understandable.
- Reproducibility is partially present through scripts and outputs.
- Packaging is not yet GitHub-release ready because the repo is missing README, requirements, LICENSE, and .gitignore.
