

### Temporary CSV / data artifacts
- Old raw CSVs that are superseded by the current experiment summaries.
- One-off per-run diagnostic CSVs that are not part of the final release narrative.
- Intermediate MCTS raw outputs that are only useful for local debugging.

### Debug logs
- Log files such as run logs from older experiments.
- Any debugging outputs that only exist to inspect a failed run.

### Cache / bytecode
- __pycache__ directories
- .pyc files
- stale compiled artifacts

### Temporary directories
- tmp/
- scratch/
- debug/
- any ad-hoc output folders created during experiments

### Unused checkpoints
- Old checkpoints from earlier experiments that are not referenced by the current main experiments.
- Duplicate checkpoints with similar names but no clear release purpose.
- Checkpoints for experiments that are not part of the final Teacher Release scope.

## Recommended to keep

### Core experiments
- All experiment scripts should be retained for reproducibility.
- Especially keep the main experiments used for the current release story:
  - Exp14
  - Exp17
  - Exp20
  - Exp22

### Core modules
- utils/
- value_network.py
- quantum_synthesizer.py
- quantum_registry.py
- quantum_mcts.py
- matrix_encoder.py
- policy_network.py
- fault_set_evaluator.py

### Release assets
- checkpoints/ for the final selected release model(s)
- design/docs that explain the methodology and experiment choices
- experiment audit and release documentation files

## Important note

Do not delete anything yet. This list is a recommendation only for a future cleanup pass.
