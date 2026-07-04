# Experiment Consistency Audit for Teacher Release

## Scope

This audit reviews the core experiments requested for release preparation:
- Exp12
- Exp13
- Exp14
- Exp17
- Exp18
- Exp19
- Exp20
- Exp22

The review is based on static inspection of the current code and the existing experiment scripts.

## Summary

Most of the requested experiments already use X-only stabilizer initialization for the search matrix and are wired to the verifier path in the relevant MCTS evaluation code. The main remaining risk is not algorithmic logic but consistency across scripts and the possibility of hidden assumptions around full-config stabilizers vs X-only state construction.

## Per-experiment audit

### Exp12
- Init matrix: Yes, uses X-only stabilizers from the registry.
- Full config usage: The training/evaluation uses the registry code config and extracts X-only stabs for the MCTS matrix.
- build_css_logical_zero_prep: Used in rollout/circuit construction.
- Verifier: Not explicitly used in the main evaluation path inspected here; the script focuses on representation sweep and MCTS metrics rather than full verifier integration.
- Valid rate: Not clearly exposed as a primary output metric in the inspected code path.
- Risk: Medium. The experiment appears structurally consistent, but it is not as verifier-driven as Exp14/17/20/22.
- Recommendation: Keep as-is for release, but mark as lower-confidence for verifier-based validation.

### Exp13
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: The script uses the full registry config for some verifier-related logic, but the MCTS state uses X-only stabs.
- build_css_logical_zero_prep: Yes.
- Verifier: Yes, in the inspected MCTS evaluation path.
- Valid rate: Yes, computed in the MCTS evaluation summary.
- Risk: Low to medium.
- Recommendation: Suitable for release with no algorithm change required.

### Exp14
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: The script explicitly splits stabs into X/Z and uses verifier with full config.
- build_css_logical_zero_prep: Yes.
- Verifier: Yes, fully connected.
- Valid rate: Yes, computed and reported.
- Risk: Low.
- Recommendation: Good release candidate.

### Exp17
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: Yes, the script explicitly splits stabs and uses the verifier.
- build_css_logical_zero_prep: Yes.
- Verifier: Yes, fully connected.
- Valid rate: Yes, computed and reported.
- Risk: Low.
- Recommendation: Good release candidate.

### Exp18
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: The inspected logic uses the registry config and X-only state initialization for the search matrix.
- build_css_logical_zero_prep: Yes.
- Verifier: Not clearly integrated as a first-class output metric in the inspected code path.
- Valid rate: Not clearly exposed in the inspected main path.
- Risk: Medium.
- Recommendation: Keep for release, but document that its main output is interpretability/ranking analysis rather than verifier-driven evaluation.

### Exp19
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: The looked-at code uses X-only state init and the registry code config.
- build_css_logical_zero_prep: Not the main focus of this script.
- Verifier: Not a primary output metric in the inspected path.
- Valid rate: Not the main reporting target.
- Risk: Medium.
- Recommendation: Suitable as supporting analysis, not as a verifier benchmark.

### Exp20
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: Yes, the evaluation path explicitly uses the full config for verifier integration and the X-only list for the search matrix.
- build_css_logical_zero_prep: Yes.
- Verifier: Yes, fully connected.
- Valid rate: Yes, computed and reported.
- Risk: Low to medium.
- Recommendation: Good release candidate, but note the script still contains a known limitation around sibling-loss alignment and the old 24-row issue if the code path is reused outside the current setup.

### Exp22
- Init matrix: Yes, X-only stabilizers are used.
- Full config usage: Yes, the script explicitly splits X/Z stabs and uses the verifier.
- build_css_logical_zero_prep: Yes.
- Verifier: Yes, fully connected.
- Valid rate: Yes, computed and reported.
- Risk: Low.
- Recommendation: Good release candidate for the current release bundle.

## Key findings

### 1) X-only init matrix usage
The inspected core experiments consistently construct the MCTS/search matrix from X-only stabilizers:
- Exp12
- Exp13
- Exp14
- Exp17
- Exp18
- Exp19
- Exp20
- Exp22

This is consistent with the codebase’s interpretation that the search matrix is the X-stabilizer subsystem, while verifier checks both X and Z stabs.

### 2) Full-config stabilizer usage
The scripts that are intended for verifier-backed evaluation use the full registry config and split the stabs into X/Z sets before verification. This is present in Exp14, Exp17, Exp20, and Exp22.

### 3) build_css_logical_zero_prep
The rollout/circuit construction path is consistent across the MCTS evaluation scripts and uses the same helper from quantum_synthesizer.py.

### 4) Verifier integration
Verifier integration is strongest in:
- Exp14
- Exp17
- Exp20
- Exp22

It is weaker or secondary in:
- Exp12
- Exp18
- Exp19

### 5) Valid-rate reporting
Valid-rate is explicitly computed and reported in:
- Exp14
- Exp17
- Exp20
- Exp22

It is not the main reported metric in:
- Exp12
- Exp18
- Exp19

### 6) Exp20-like 24-row issue
No direct evidence of the same 24-row bug pattern was found in the inspected current code paths. However, the repository still contains older-style experimental scripts and outputs that may have used different assumptions in the past. For release purposes, the current main scripts appear to use the X-only matrix construction pattern consistently.

## Release recommendation

- Safe to include in a Teacher Release bundle: Exp14, Exp17, Exp20, Exp22.
- Suitable as supporting analysis: Exp12, Exp13, Exp18, Exp19.
- No algorithm changes are recommended for this audit pass.
- The main release risk is documentation clarity and reproducibility rather than a fundamental algorithmic mismatch.
