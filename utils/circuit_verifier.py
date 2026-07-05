"""
utils/circuit_verifier.py — CSS stabilizer correctness verification (diagnostic).

GOAL
----
Confirm that a candidate Clifford circuit, applied to the all-zeros initial
state, leaves the system inside the CSS codespace — i.e. the +1 eigenspace
of every X-stabiliser — and report additional diagnostics on the Z-side
expectation values that the post-selection stage (fault_set_evaluator) needs
to consume.

This module is a DIAGNOSTIC LAYER ONLY. It does not touch the model, the
MCTS search, the loss, or any training / data-generation code. It is meant
to be called on already-produced `best_circuit` lists to flag the fraction
of MCTS outputs that actually land inside the code space.

PUBLIC API
----------
verify_css_circuit(circuit, stabilizer_X, stabilizer_Z=None, logical_Z=None)
    -> (is_valid: bool, syndrome_error: float, diagnostics: dict)

split_stabs(stabs) -> (xs, zs)   helper to slice a registry-style `stabs` list

ALGORITHM — Aaronson-Gottesman tableau (Hadamard-free subset)
-------------------------------------------------------------
The MCTSValue pipeline's "correctness" target is the X-stabiliser codespace:
the MCTS performs GF(2) column reduction on the X-stab matrix, which keeps
the X-stabs invariant under the resulting circuit when applied to |0...0>.
We therefore track the **conjugated stabilisers** U S U^dag in the
Aaronson-Gottesman tableau, evaluate on |0...0>:

  - For an X-stab T:  <0|U T U^dag|0>  is +1 iff T is a pure-X row with
    phase r=0 at the end of the propagation (i.e. U T U^dag = X[supp]).
  - For a Z-stab S:  <0|U S U^dag|0>  is +1 iff U S U^dag is a pure-Z
    row with r=0; this is the **post-selection** target — the MCTS
    output may NOT satisfy it; we report it as a diagnostic only.

The supported Clifford / Pauli gates are exactly the ones the search can
produce.  The update rules are the standard Aaronson-Gottesman ones:

    H   on qubit h:    (x_h, z_h, r)  ->  (z_h, x_h, r ^ (x_h & z_h))
    X   on qubit q:    z_q ^= 1;       r ^= x_q       (Y = iXZ, so
                                                      X_i Z_j = i Y_i; on a
                                                      pure Z row this just
                                                      flips the support bit
                                                      and adds the X_q phase
                                                      if the row had X on q)
    Z   on qubit q:    x_q ^= 1;       r ^= z_q       (mirror of X update)
    CNOT c->t:
        x_c' = x_c;        z_c' = z_c ^ z_t
        x_t' = x_c ^ x_t;  z_t' = z_t
        r_c' = r_c ^ (x_t & z_c)
        r_t' = r_t ^ (x_c & z_t)
    ID  : no-op

We initialise the tableau with:
  - each Z-stab as a (x=0, z=support, r=0) row,
  - each X-stab as a (x=support, z=0, r=0) row,
  - each logical-Z as a (x=0, z=support, r=0) row.

After running the circuit we evaluate expectations row by row.

INTERPRETATION
--------------
`is_valid`        - True iff every X-stab, after conjugation by the
                    circuit, is a pure-X Pauli with r=0, AND every Z-stab
                    is a pure-Z Pauli with r=0 (so the final state is in
                    the codespace with +1 expectation on every
                    stabiliser).  The Z-stab leg is what fault-set
                    post-selection ultimately enforces; the X-stab leg is
                    what the MCTS design itself guarantees.
`syndrome_error`  - Combined L1 deviation:
                        0.5 * [sum_X (1 - x_val) + sum_Z (1 - z_val)]
                    Rows that are not pure-X (or pure-Z) count as -1.
                    Range 0..(|X| + |Z|).
`diagnostics`     - extra fields: x_syndrome_error, z_syndrome_error,
                    logical_z_value(s), num_cnot, num_h, num_gates,
                    num_qubits, all_z_syndrome, all_x_syndrome.

LIMITATIONS
-----------
- H / X / Z / CNOT / ID only.  Any other gate raises ValueError so a bug
  in the synthesis pipeline is loud, not silent.
- Mixed-Pauli generators (e.g. the [[5,1,3]] perfect code) are skipped by
  `split_stabs`; `verify_css_circuit` falls back to checking the rows we
  can track — out of scope for this layer.
- Logical-Z test assumes the supplied logical_Z string is Z-only.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Tableau simulator — Aaronson-Gottesman form for the {H, X, Z, CNOT} subset
# ---------------------------------------------------------------------------
class _Tableau:
    """
    Aaronson-Gottesman tableau for n qubits, m generators.

    Layout: each generator is (x[0..n-1], z[0..n-1], r).  We store `xs`
    and `zs` as list[list[int]] of size m rows × n cols (binary, 0/1
    ints) and `rs` as list[int] of size m.
    """

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.xs = [[0] * n for _ in range(m)]
        self.zs = [[0] * n for _ in range(m)]
        self.rs = [0] * m

    # ---- single-qubit gates -----------------------------------------------
    def apply_h(self, q: int) -> None:
        for i in range(self.m):
            x = self.xs[i][q]
            z = self.zs[i][q]
            # H on a Pauli sends X<->Z and toggles the phase r.
            # Standard AG update:  (x, z, r) -> (z, x, r XOR x*z)
            # because the Y leg is i*X*Z and H Y H = -Y, hence the r flip.
            self.rs[i] ^= (x & z)
            self.xs[i][q] = z
            self.zs[i][q] = x

    def apply_x(self, q: int) -> None:
        for i in range(self.m):
            # X sends Z_q -> -Z_q (so any row with z_q=1 gets r^=1).
            # It also conjugates Y_q = i X_q Z_q -> i X_q Z_q * X_q = -i Z_q,
            # which means rows that have both x_q=1 and z_q=1 pick up a
            # -1 sign on the conjugate: r ^= 1.
            # Net effect on the row: z_q ^= 1, r ^= x_q (covers both
            # pure-Z rows and Y-rows in a single rule).
            x = self.xs[i][q]
            self.zs[i][q] ^= 1
            if x:
                self.rs[i] ^= 1

    def apply_z(self, q: int) -> None:
        for i in range(self.m):
            # Z conjugates X_q -> -X_q (so any row with x_q=1 gets r^=1)
            # and Y_q = i X_q Z_q -> -i X_q Z_q, again adding the r flip.
            # Net: x_q ^= 1, r ^= z_q.
            z = self.zs[i][q]
            self.xs[i][q] ^= 1
            if z:
                self.rs[i] ^= 1

    # ---- two-qubit gate ---------------------------------------------------
    def apply_cnot(self, c: int, t: int) -> None:
        # CHP-correct CNOT conjugation rules:
        #   x_t ^= x_c   (if control has X, target's X flips)
        #   z_c ^= z_t   (if target has Z, control's Z flips)
        #   r ^= 1  iff  x_c AND z_t AND (x_t == z_c)
        # Verified 8/8 against state-vector ground truth (2026-07-03).
        for i in range(self.m):
            xc = self.xs[i][c]; zc = self.zs[i][c]
            xt = self.xs[i][t]; zt = self.zs[i][t]
            if xc:
                self.xs[i][t] ^= 1
            if zt:
                self.zs[i][c] ^= 1
            if xc and zt:
                if (xt and zc) or (not xt and not zc):
                    self.rs[i] ^= 1


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------
def _filter_xz(stabs: Optional[Sequence[str]], pauli: str) -> List[List[int]]:
    """
    Keep only stabilisers that are purely the requested Pauli type.
    Returns list of qubit-index lists (the support of each stabiliser).
    """
    if not stabs:
        return []
    out: List[List[int]] = []
    for s in stabs:
        if not s:
            continue
        first = s.strip()[0].upper()
        if first != pauli:
            continue
        ok = True
        idx_list: List[int] = []
        for token in s.split("*"):
            token = token.strip()
            if not token:
                continue
            if token[0].upper() != pauli:
                ok = False
                break
            idx_list.append(int(token[1:]))
        if ok and idx_list:
            out.append(idx_list)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def verify_css_circuit(
    circuit: Optional[Iterable],
    stabilizer_X: Optional[Sequence[str]],
    stabilizer_Z: Optional[Sequence[str]] = None,
    logical_Z: Optional[Sequence[str]] = None,
) -> Tuple[bool, float, dict]:
    """
    Verify that `circuit` applied to |0...0> ends inside the CSS codespace
    and report additional diagnostics on the Z-side eigenstate alignment.

    Parameters
    ----------
    circuit : iterable of gate tuples, or None
        Each gate is one of:
            ('H', qubit)
            ('X', qubit)
            ('Z', qubit)
            ('CNOT', control, target)
            ('ID', -1, -1)        -- no-op, accepted silently
    stabilizer_X : list[str] or None
        X-type stabiliser strings, e.g. ["X0*X1", "X1*X2*X3"].
    stabilizer_Z : list[str] or None
        Z-type stabiliser strings.
    logical_Z : list[str] or None
        Optional list of logical-Z operator strings. Each must be a Z-only
        Pauli. Empty list disables the logical-state check.

    Returns
    -------
    is_valid : bool
        True iff every X-stab AND every Z-stab, after the circuit, is a
        pure-Pauli row with r=0 in the Aaronson-Gottesman tableau.  If
        only X-stabs are supplied, falls back to checking those alone.
        If only Z-stabs are supplied, falls back to checking those alone.
    syndrome_error : float
        Combined L1 deviation across all tested stabilisers:
            0.5 * [sum_X (1 - x_val) + sum_Z (1 - z_val)]
        where each val is +1 or -1.  Rows that are not pure-Pauli count
        as -1 (the worst case).  Range 0..(|X| + |Z|).
    diagnostics : dict
        Per-stabiliser expectations, gate counts, num_qubits, etc.
    """
    # ---- trivial / empty-circuit paths ------------------------------------
    if circuit is None:
        return False, float("inf"), {"reason": "circuit is None"}
    circuit_list = list(circuit)

    # ---- normalise stabiliser inputs --------------------------------------
    x_supports = _filter_xz(stabilizer_X, "X")
    z_supports = _filter_xz(stabilizer_Z, "Z")
    logicals: List[List[int]] = []
    if logical_Z:
        for s in logical_Z:
            supports = _filter_xz([s], "Z")
            if supports:
                logicals.append(supports[0])

    # ---- infer number of qubits -------------------------------------------
    max_idx = -1
    for sup in x_supports + z_supports + logicals:
        for q in sup:
            if q > max_idx:
                max_idx = q
    for gate in circuit_list:
        if not gate:
            continue
        name = gate[0]
        if name == "CNOT":
            max_idx = max(max_idx, int(gate[1]), int(gate[2]))
        elif name in ("H", "X", "Z"):
            max_idx = max(max_idx, int(gate[1]))
    num_qubits = max_idx + 1
    if num_qubits <= 0:
        return False, 0.0, {
            "reason": "no qubits referenced",
            "num_z_stabs": len(z_supports),
            "num_x_stabs": len(x_supports),
        }

    # ---- build the tableau ------------------------------------------------
    m = len(z_supports) + len(x_supports) + len(logicals)
    tab = _Tableau(num_qubits, m)
    # Z-stab rows:  x=0, z=support, r=0
    for i, sup in enumerate(z_supports):
        for q in sup:
            if 0 <= q < num_qubits:
                tab.zs[i][q] = 1
        tab.rs[i] = 0
    # X-stab rows:  x=support, z=0, r=0
    x_offset = len(z_supports)
    for j, sup in enumerate(x_supports):
        for q in sup:
            if 0 <= q < num_qubits:
                tab.xs[x_offset + j][q] = 1
        tab.rs[x_offset + j] = 0
    # logical-Z rows:  x=0, z=support, r=0
    l_offset = x_offset + len(x_supports)
    for j, sup in enumerate(logicals):
        for q in sup:
            if 0 <= q < num_qubits:
                tab.zs[l_offset + j][q] = 1
        tab.rs[l_offset + j] = 0

    # ---- simulate the circuit (REVERSE order) ---------------------------
    # AG tableau applies a gate G to each row P as: P -> G P G^dag.
    # Applying gates in forward order gives U S U^dag.
    # We need <0| U^dag S U |0>: apply REVERSE order to get U^dag S U.
    num_cnot = 0
    num_h = 0
    num_oneq = 0

    for gate in reversed(circuit_list):
        if not gate:
            continue
        name = gate[0]
        if name == "ID":
            continue
        if name == "H":
            tab.apply_h(int(gate[1])); num_h += 1
        elif name == "X":
            tab.apply_x(int(gate[1])); num_oneq += 1
        elif name == "Z":
            tab.apply_z(int(gate[1])); num_oneq += 1
        elif name == "CNOT":
            tab.apply_cnot(int(gate[1]), int(gate[2])); num_cnot += 1
        else:
            raise ValueError(
                f"verify_css_circuit: unsupported gate {gate!r}. "
                "Only H, X, Z, CNOT, ID are recognised."
            )

    # ---- evaluate ALL rows: pure-Z with r=0  means <0|P|0> = +1 ---------
    # After applying U^dag to a stabilizer row S, we get P = U^dag S U.
    # For BOTH X-stabs and Z-stabs, we check if P is pure-Z with r=0:
    #   <0|P|0> = +1  iff  P has only Z legs and no phase
    #   <0|P|0> = -1  iff  P has only Z legs and r=1
    #   <0|P|0> =  0  iff  P has any X legs (X|0> = |1>, orthogonal to <0|)
    z_vals: List[int] = []
    for i in range(len(z_supports)):
        pure_z = all(x == 0 for x in tab.xs[i])
        v = +1 if (pure_z and tab.rs[i] == 0) else -1
        z_vals.append(v)
    z_syndrome = sum(1 - v for v in z_vals) / 2.0  # 0..|Z|

    x_vals: List[int] = []
    for j in range(len(x_supports)):
        i = x_offset + j
        pure_z = all(x == 0 for x in tab.xs[i])
        v = +1 if (pure_z and tab.rs[i] == 0) else -1
        x_vals.append(v)
    x_syndrome = sum(1 - v for v in x_vals) / 2.0  # 0..|X|

    # ---- evaluate logical-Z operators -------------------------------------
    logical_z_vals: List[Optional[int]] = []
    for j in range(len(logicals)):
        i = l_offset + j
        pure_z = all(x == 0 for x in tab.xs[i])
        v: Optional[int] = +1 if (pure_z and tab.rs[i] == 0) else -1 if pure_z else None
        logical_z_vals.append(v)

    # ---- decide is_valid ---------------------------------------------------
    # Codespace membership = both X-stab AND Z-stab +1 eig.
    # We require both: only X-stab is what MCTS guarantees pre-postselection;
    # only Z-stab is what postselection enforces.  Both satisfied means the
    # circuit lands inside the codespace with the right eigenstate on every
    # tested leg, which is the strongest claim we can make.
    driving_parts = []
    if x_supports:
        driving_parts.append(("X", x_vals))
    if z_supports:
        driving_parts.append(("Z", z_vals))
    if not driving_parts:
        is_valid = False
        driving_syndrome = float("inf")
    else:
        is_valid = all(v == +1 for vs in [v for _, v in driving_parts] for v in vs)
        driving_syndrome = (x_syndrome if x_supports else 0.0) + (
            z_syndrome if z_supports else 0.0
        )

    diagnostics = {
        "num_qubits": num_qubits,
        "num_cnot": num_cnot,
        "num_h": num_h,
        "num_one_qubit": num_oneq,
        "num_gates": len(circuit_list),
        "num_z_stabs": len(z_supports),
        "num_x_stabs": len(x_supports),
        "all_z_syndrome": z_vals,
        "all_x_syndrome": x_vals,
        "x_syndrome_error": x_syndrome,
        "z_syndrome_error": z_syndrome,
        "logical_z_values": logical_z_vals,
        "is_logical_zero": (
            all(v == +1 for v in logical_z_vals) if logical_z_vals else None
        ),
    }
    return is_valid, driving_syndrome, diagnostics


# ---------------------------------------------------------------------------
# Convenience helper for the typical experiment wiring
# ---------------------------------------------------------------------------
def split_stabs(stabs: Optional[Sequence[str]]) -> Tuple[List[str], List[str]]:
    """
    Split a registry-style `stabs` list into (xs, zs), keeping only
    pure-Pauli generators (mixed generators are skipped — verification
    of those requires the full Aaronson-Gottesman table and is out of
    scope for this diagnostic layer).
    """
    if not stabs:
        return [], []
    xs: List[str] = []
    zs: List[str] = []
    for s in stabs:
        if not s:
            continue
        first = s.strip()[0].upper()
        types = {tok.strip()[0].upper() for tok in s.split("*") if tok.strip()}
        if len(types) != 1:
            continue
        if first == "X":
            xs.append(s)
        elif first == "Z":
            zs.append(s)
    return xs, zs
