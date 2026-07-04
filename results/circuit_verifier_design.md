# CSS Stabilizer Correctness Verifier — Design Document

**Date:** 2026-07-02 (updated 2026-07-03 after bug-fix audit)
**Status:** Corrected, verified against state-vector ground truth (8/8 match). Integrated into Exp14/17/20.
**Scope:** Evaluation-only diagnostic layer. NO model, MCTS, loss, training, or
data-generation changes.

## 1. Motivation

The MCTSValue pipeline produces a `best_circuit` for each MCTS run. The cost
metric is **gate count only** — there is no explicit "does the circuit land
inside the CSS codespace?" check. The `fault_set_evaluator` then wraps the
circuit in a post-selection stage to enforce codespace membership, but
**nothing in the MCTS layer itself** certifies the circuit's stabilizer
properties.

This verifier adds that missing layer. It is a **diagnostic**, not a control
loop: it is called on the circuit *after* MCTS finishes, and the output is
recorded alongside the existing CNOT/depth metrics. The MCTS never sees the
verifier's output and cannot be guided by it (unless the user explicitly
chooses to do so in a future experiment — out of scope here).

## 2. API

```python
from utils.circuit_verifier import verify_css_circuit, split_stabs

# split_stabs: keep only pure-X / pure-Z generators
xs, zs = split_stabs(config["stabs"])

# verify_css_circuit: returns (is_valid, syndrome_error, diagnostics)
is_valid, syn_err, diag = verify_css_circuit(
    circuit,              # list of (gate, qubit[, qubit]) tuples
    stabilizer_X=xs,      # pure-X stabiliser strings, e.g. ["X0*X1*X2*X3"]
    stabilizer_Z=zs,      # pure-Z stabiliser strings (or None)
    logical_Z=logicals,   # Z-only logical operator strings (or None)
)
```

Supported gates: `'H'`, `'X'`, `'Z'`, `'CNOT'`, `'ID'`. Anything else raises
`ValueError` so that a bug in the synthesis pipeline is loud, not silent.

## 3. Algorithm — Aaronson-Gottesman tableau (corrected)

### 3.1 Core idea

For each code stabiliser S (X-stab or Z-stab), we want to know whether the
final state |ψ⟩ = U|0...0⟩ satisfies S|ψ⟩ = +|ψ⟩.  This is equivalent to

    ⟨0| U^dag S U |0⟩ = +1

We track the operator P = U^dag S U by evolving S through the inverses of
the circuit gates, using the Aaronson-Gottesman tableau.  After evolution,
we check whether P is a **pure-Z Pauli with r=0**, because

    ⟨0| P |0⟩ = +1   iff   P has only Z legs and no phase (r=0)
    ⟨0| P |0⟩ = -1   iff   P has only Z legs and r=1
    ⟨0| P |0⟩ = 0    iff   P has any X legs (⟨0|X|0⟩ = 0)

**Critical point:** ALL stabiliser rows — both X-stabs and Z-stabs — are
evaluated by the same criterion (pure-Z with r=0) after the reverse
propagation.  The earlier version that checked X-stabs as "pure-X with
r=0" was incorrect (it was checking whether S was a stabiliser of |0⟩,
which it is not).

### 3.2 Tableau update rules

Each gate is self-inverse (H^dag=H, CNOT^dag=CNOT, X^dag=X, Z^dag=Z), so
applying the **reversed** circuit list to row S gives U^dag S U.

| Gate     | Update rule                                                                 |
|----------|------------------------------------------------------------------------------|
| `H q`    | `x_q, z_q = z_q, x_q`;  `r ^= x_q & z_q` (before swap)                      |
| `X q`    | `z_q ^= 1`;  `r ^= x_q`                                                     |
| `Z q`    | `x_q ^= 1`;  `r ^= z_q`                                                     |
| `CNOT c->t` | `if x_c: x_t ^= 1`;  `if z_t: z_c ^= 1`;  r ^= 1 iff x_c & z_t & (x_t == z_c) — CHP-verified |
| `ID`     | no-op                                                                        |

**The CNOT rule differs from the MCTS X-stab matrix update.**  MCTS uses
`col_t ^= col_c` (GF(2) elimination on the X-half), while the verifier
uses the Pauli conjugation direction (CHP code) which is the opposite.
These are different operations for different purposes.

### 3.3 Verification against state-vector ground truth

Tested 8 circuits on Steane [[7,1,3]] (n_x=3, n_z=3):

| Circuit          | SV is_valid | AG is_valid | Match |
|------------------|-------------|-------------|-------|
| empty            | False       | False       | ✓     |
| H^7              | False       | False       | ✓     |
| encoder (manual) | True        | True        | ✓     |
| heuristic rollout| True        | True        | ✓     |
| -first H         | False       | False       | ✓     |
| -first CNOT      | False       | False       | ✓     |
| +extra CNOT      | False       | False       | ✓     |
| CNOT reverse     | False       | False       | ✓     |

**8/8 match.**  Reverse-order AG tableau with CHP-correct CNOT and
unified pure-Z evaluation is consistent with state-vector simulation.

## 6. Integration into experiments 14, 17, 20

All three experiment scripts call `verify_css_circuit` immediately after
`mcts.run()` returns, store `is_valid` / `syndrome_error` /
`x_syndrome_error` / `z_syndrome_error` / `is_logical_zero` alongside the
existing `cnot` / `depth` fields, and aggregate:

- `valid_rate`     — fraction of MCTS runs that produced a valid circuit
- `valid_cnot_mean`— mean CNOT of valid circuits only (for direct comparison
  with the project's main `cnot_mean`)
- `invalid_count`  — number of runs that produced an invalid circuit
- `z_syndrome_mean` / `x_syndrome_mean` — average verifier syndrome

CSV outputs are extended with these fields. Original metric column names
are preserved so any downstream consumer still parses the file correctly.
The verifier diagnostics are appended as new columns.

**No** changes were made to:
- `value_network.py` (model definitions)
- `quantum_mcts.py` (search logic)
- `quantum_synthesizer.py` (rollout / circuit building)
- `fault_set_evaluator.py` (FT cost)
- `experiment_label_comparison.py`, `experiment_3_search.py`, etc.

## 7. Out of scope

- **Mixed-Pauli generators** (e.g. the [[5,1,3]] perfect code) are skipped
  by `split_stabs` and the verifier falls back to checking whatever pure
  generators it can track. Tracking mixed-Pauli stabilisers requires the
  full Aaronson-Gottesman table with destabiliser rows; that's a follow-up.
- **T-gates / S-gates / non-Clifford gates** are not supported. We raise
  `ValueError` to make any non-Clifford insertion loud.
- **The verifier does not change the MCTS objective or any training signal.**
  It is a pure observation tool. A future experiment could close the loop
  by penalising invalid circuits during MCTS, but that would be a new
  experiment and is out of scope here.
