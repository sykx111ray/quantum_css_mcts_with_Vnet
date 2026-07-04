# quantum_synthesizer.py
import numpy as np
import networkx as nx
import random


def build_css_logical_zero_prep(prefix_cnots, final_pivots):
    valid_cnots = [g for g in prefix_cnots if g[0] != "ID"]
    uncompute_cnots = optimize_circuit(valid_cnots)
    return [("H", int(p)) for p in final_pivots] + list(reversed(uncompute_cnots))


def optimize_circuit(gates):
    def is_self_inverse(g):
        return g[0] in ["H", "X", "Z", "CNOT"]

    def same_gate(g1, g2):
        return g1 == g2

    def cnot_qubits(g):
        return int(g[1]), int(g[2])

    def disjoint(g1, g2):
        q1 = set(cnot_qubits(g1))
        q2 = set(cnot_qubits(g2))
        return q1.isdisjoint(q2)

    def can_swap(g1, g2):
        if g1[0] == "CNOT" and g2[0] == "CNOT":
            return disjoint(g1, g2)
        if g1[0] in ["H", "X", "Z"] and g2[0] == "CNOT":
            q = int(g1[1])
            c, t = cnot_qubits(g2)
            return q != c and q != t
        if g1[0] == "CNOT" and g2[0] in ["H", "X", "Z"]:
            q = int(g2[1])
            c, t = cnot_qubits(g1)
            return q != c and q != t
        if g1[0] in ["H", "X", "Z"] and g2[0] in ["H", "X", "Z"]:
            return int(g1[1]) != int(g2[1])
        return False

    current = list(gates)
    if not current:
        return current

    changed = True
    max_passes = 10
    passes = 0

    while changed and passes < max_passes:
        passes += 1
        changed = False

        i = 0
        while i < len(current) - 1:
            g1, g2 = current[i], current[i + 1]
            if can_swap(g1, g2) and str(g1) > str(g2):
                current[i], current[i + 1] = current[i + 1], current[i]
                changed = True
                if i > 0:
                    i -= 1
                    continue
            i += 1

        reduced = []
        for g in current:
            if reduced and is_self_inverse(g) and same_gate(reduced[-1], g):
                reduced.pop()
                changed = True
            else:
                reduced.append(g)

        current = reduced

    return current


class HeuristicRolloutSolver:
    def __init__(self, topo_edges, num_qubits, code_name="", precompute_path_threshold=100):
        self.num_qubits = num_qubits
        self.graph = nx.Graph(topo_edges)
        self.precompute_path_threshold = precompute_path_threshold
        self.path_cache = {}
        self.paths = None
        self.code_name = code_name # 仅保留变量以保证外部接口兼容

        if self.num_qubits <= self.precompute_path_threshold:
            self.paths = dict(nx.all_pairs_shortest_path(self.graph))

    def _get_path(self, control, target):
        if control == target:
            return [control]
        if self.paths is not None:
            return self.paths[control][target]

        key = (control, target)
        rev_key = (target, control)
        if key in self.path_cache:
            return self.path_cache[key]
        if rev_key in self.path_cache:
            return list(reversed(self.path_cache[rev_key]))

        path = nx.shortest_path(self.graph, control, target)
        self.path_cache[key] = path
        return path

    def _route_cnot(self, control, target):
        if self.graph.has_edge(control, target):
            return [('CNOT', control, target)]

        path = self._get_path(control, target)
        gates = []
        swaps = []

        for i in range(len(path) - 1, 1, -1):
            u, v = path[i - 1], path[i]
            gates.extend([('CNOT', u, v), ('CNOT', v, u), ('CNOT', u, v)])
            swaps.append((u, v))

        gates.append(('CNOT', control, path[1]))

        for u, v in reversed(swaps):
            gates.extend([('CNOT', u, v), ('CNOT', v, u), ('CNOT', u, v)])

        return gates

    def solve_remainder_randomized(self, current_matrix, randomize=True):
        """严格随机 GF(2) 高斯列消元策略，适用于所有类型的 Code"""
        matrix = current_matrix.copy()
        rows, cols = matrix.shape
        rollout_gates = []
        used_cols = set()
        
        row_order = list(range(rows))
        if randomize:
            random.shuffle(row_order)

        for r in row_order:
            available_cols = [c for c in range(cols) if matrix[r, c] == 1 and c not in used_cols]

            if not available_cols:
                if not np.any(matrix[r]):
                    continue
                return None, None 

            pivot_col = random.choice(available_cols) if randomize else available_cols[0]
            used_cols.add(pivot_col)

            targets = [c for c in range(cols) if c != pivot_col and matrix[r, c] == 1]

            for target_col in targets:
                matrix[:, target_col] ^= matrix[:, pivot_col]
                routed_gates = self._route_cnot(control=pivot_col, target=target_col)
                rollout_gates.extend(routed_gates)

        final_pivots = []
        for r in range(rows):
            ones_in_row = np.where(matrix[r] == 1)[0]
            if len(ones_in_row) == 1:
                final_pivots.append(int(ones_in_row[0]))
            elif len(ones_in_row) > 1:
                return None, None

        return rollout_gates, final_pivots

    def solve_remainder(self, current_matrix, randomize=True):
        max_attempts = 10 if randomize else 1
        for _ in range(max_attempts):
            rollout_gates, final_pivots = self.solve_remainder_randomized(current_matrix, randomize)
            if rollout_gates is not None and final_pivots is not None:
                return rollout_gates, final_pivots
        return None, None