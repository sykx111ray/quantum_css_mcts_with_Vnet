# Surface d=9 Readiness Review

## Scope

This review checks only whether the current pipeline structure appears compatible with Surface d=9, without training or executing anything.

## What appears ready

### Dataset
- The current pipeline uses the QuantumCodeRegistry-based code configuration and the X-stabilizer initialization pattern.
- This structure is likely reusable for Surface d=9 if the registry contains a matching code configuration.

### Solver
- The solver entry point is generic and uses the code name plus topology.
- If a matching Surface d=9 code configuration exists, the current solver path should be reusable with little structural change.

### MCTS
- The MCTS scripts are structurally generic and operate over the state matrix plus action generation.
- They should be reusable for Surface d=9 if the state dimension and action space are handled correctly.

### ValueNet
- The value-network entry points are generic enough to accept the new matrix shapes and state representation.
- The main concern is whether the training data and checkpoint compatibility still match the new code instance.

### Verifier
- The verifier path is already wired to the registry-based config and the split X/Z stabilizer logic.
- If the Surface d=9 code config is valid and exposes the expected stabilizer/logical structure, the verifier should be reusable.

## Likely required changes

### 1) Code configuration
- A new registry entry or config for Surface d=9 would be needed if it is not already present.

### 2) Matrix dimensions / qubit count
- The current scripts assume 25 qubits and the existing Surface d=5 layout conventions.
- Surface d=9 will require re-checking the expected qubit count and matrix size assumptions.

### 3) Checkpoint compatibility
- Existing checkpoints are not automatically compatible with a new code instance or new matrix size unless retraining or re-checkpointing is performed.

### 4) Runtime cost
- The current pipeline is experiment-heavy and may become significantly more expensive for a larger code instance.

## Risks

- The biggest risk is not the MCTS or solver logic but the data/config compatibility for a new code instance.
- A second risk is whether the current value-network training pipeline and checkpoint naming assumptions still fit the new target.
- A third risk is that the verifier may need extra handling if the Surface d=9 code uses a different stabilizer/logical structure.

## Bottom line

The current pipeline appears structurally capable of supporting Surface d=9 in principle, but it is not yet a drop-in ready path. The main work needed is configuration compatibility, dimension assumptions, and checkpoint strategy rather than new algorithm development.
