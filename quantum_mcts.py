import itertools
import math
import os
import pickle
import random
import sys

import networkx as nx
import numpy as np

import torch

from fault_set_evaluator import FTEvaluator
from quantum_registry import QuantumCodeRegistry
from quantum_synthesizer import HeuristicRolloutSolver, build_css_logical_zero_prep
from value_network import CostNormalizer, SteaneValueNet, RowColValueNet, RowColRankValueNet, matrix_to_input, _gf2_rank
from policy_network import SteanePolicyNet, action_to_index, NUM_ACTIONS


def split_css_stabilizers(stabs):
    x_stabs = []
    z_stabs = []
    for stab in stabs:
        terms = [p.strip() for p in stab.split("*") if p.strip()]
        paulis = {term[0] for term in terms}
        if paulis == {"X"}:
            x_stabs.append(stab)
        elif paulis == {"Z"}:
            z_stabs.append(stab)
        else:
            raise ValueError(f"Non-CSS stabilizer encountered: {stab}")
    return x_stabs, z_stabs


def get_grid_edges(d):
    edges = []
    for r in range(d):
        for c in range(d):
            if c < d - 1:
                edges.append((r * d + c, r * d + c + 1))
            if r < d - 1:
                edges.append((r * d + c, (r + 1) * d + c))
    return edges


def get_all_to_all_edges(num_qubits):
    return list(itertools.combinations(range(num_qubits), 2))


def gf2_rank(matrix):
    """Compute rank of binary matrix over GF(2) via Gaussian elimination."""
    m = matrix.copy().astype(np.uint8)
    nrows, ncols = m.shape
    rank = 0
    for col in range(ncols):
        pivot = None
        for row in range(rank, nrows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(nrows):
            if row != rank and m[row, col] == 1:
                m[row] ^= m[rank]
        rank += 1
        if rank == nrows:
            break
    return rank


def extract_state_features(matrix_state):
    """Extract domain-specific features from a GF(2) matrix state."""
    rank = gf2_rank(matrix_state)
    row_weights = matrix_state.sum(axis=1).tolist()
    col_weights = matrix_state.sum(axis=0).tolist()
    active_qubits = int(np.sum(np.any(matrix_state, axis=0)))
    return {
        "rank": rank,
        "row_weights": row_weights,
        "col_weights": col_weights,
        "active_qubits": active_qubits,
    }


class GateLevelMCTSEnv:
    def __init__(self, stabs, num_data, topo_edges):
        self.num_data = num_data
        self.graph = nx.Graph(topo_edges)
        self.M = len(stabs)
        self.init_matrix = np.zeros((self.M, num_data), dtype=int)
        for i, stab in enumerate(stabs):
            for p in stab.split("*"):
                self.init_matrix[i, int(p.strip()[1:])] = 1

    def apply_physical_action(self, matrix, action):
        new_mat = matrix.copy()
        if action[0] == "CNOT":
            new_mat[:, action[2]] ^= new_mat[:, action[1]]
        return new_mat

    def get_promising_actions(self, matrix, nums, current_circuit):
        actions = []
        active_qubits = np.where(matrix.any(axis=0))[0]
        
        layer_depth = len(current_circuit)
        cycle_group = layer_depth % 4 
        
        for q in active_qubits:
            if q % 4 != cycle_group:
                continue
                
            for neighbor in self.graph.neighbors(q):
                if neighbor >= self.num_data or q >= self.num_data:
                    continue
                actions.extend([("CNOT", q, neighbor), ("CNOT", neighbor, q)])

        actions.append(("ID", -1, -1))

        if current_circuit:
            last = current_circuit[-1]
            actions = [a for a in actions if a != last]

        actions = list(set(actions))
        random.shuffle(actions)
        return actions[:nums]


class GateLevelMCTSNode:
    def __init__(self, prefix_circuit, matrix_state, parent=None, action=None):
        self.prefix_circuit = prefix_circuit
        self.matrix_state = matrix_state
        self.parent = parent
        self.children = []
        self.untried_actions = []
        self.is_fully_expanded = False
        self.visits = 0
        self.total_cost = 0.0
        self.action = action


class QuantumGateMCTS:
    def __init__(self, env, ft_evaluator, code_target="", iterations=5000, rollouts=5,
                 warmup_iters=2500, action_nums=10, collect_data=False, collect_interval=4,
                 run_id=0, use_value_net=False, value_net_path=None,
                 use_policy_net=False, policy_net_path=None, c_puct=2.0,
                 verbose=True):
        self.env = env
        self.ft_evaluator = ft_evaluator
        self.iterations = iterations
        self.rollouts_per_node = rollouts
        self.warmup_iters = warmup_iters
        self.nums = action_nums
        self.current_iter = 0
        self.code_target = code_target
        self.run_id = run_id

        self.root = GateLevelMCTSNode(prefix_circuit=[], matrix_state=self.env.init_matrix)
        self.root.untried_actions = self.env.get_promising_actions(
            self.root.matrix_state, nums=action_nums, current_circuit=[])

        self.solver = HeuristicRolloutSolver(
            self.env.graph.edges(), self.env.num_data, code_name=self.code_target)
        self.global_best_cost = float("inf")
        self.global_best_circuit = None

        self.collect_data = collect_data
        self.collect_interval = collect_interval
        self.training_samples = []
        self.policy_samples = []

        self.use_value_net = use_value_net
        self.value_net = None
        self._vn_normalizer = None
        self._vn_device = torch.device("cpu")
        self._vn_eval_count = 0
        self._vn_path = value_net_path
        self._vn_is_rowcol = False      # set by _load_value_network
        self._vn_is_rowcol_rank = False  # set by _load_value_network
        self._vn_max_rank = 1.0          # for rank normalisation
        self.global_best_predicted = float("inf")
        self.global_best_predicted_iter = -1
        self.convergence_history = []
        self.verbose = verbose
        if self.use_value_net:
            self._load_value_network(value_net_path)

        self.use_policy_net = use_policy_net
        self.policy_net = None
        self._pn_device = torch.device("cpu")
        self.c_puct = c_puct
        if self.use_policy_net:
            self._load_policy_network(policy_net_path)

    def select(self, node):
        while node.children and node.is_fully_expanded:
            node = min(
                node.children,
                key=lambda c: (c.total_cost / (c.visits + 1e-6))
                - 1.5 * math.sqrt(math.log(node.visits + 1) / (c.visits + 1e-6)),
            )
        return node

    def expand(self, node):
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        if not node.untried_actions:
            node.is_fully_expanded = True

        child = GateLevelMCTSNode(
            prefix_circuit=node.prefix_circuit + [action],
            matrix_state=self.env.apply_physical_action(node.matrix_state, action),
            parent=node,
            action=action,
        )
        child.untried_actions = self.env.get_promising_actions(child.matrix_state, self.nums, child.prefix_circuit)
        node.children.append(child)
        return child

    def simulate(self, node):
        best_local_cost = float("inf")
        best_base_cost = 0
        best_ft_stats = None

        for _ in range(self.rollouts_per_node):
            rollout_gates, final_pivots = self.solver.solve_remainder(node.matrix_state, randomize=True)
            if rollout_gates is None or final_pivots is None:
                continue

            prep_circuit = build_css_logical_zero_prep(node.prefix_circuit + rollout_gates, final_pivots)
            base_cost = len(prep_circuit)

            if self.current_iter >= self.warmup_iters:
                ft_stats = self.ft_evaluator.evaluate_postselected(prep_circuit)
                total_cost = base_cost + ft_stats["ft_cost"]
                final_circuit = ft_stats["full_circuit"]
            else:
                total_cost = base_cost
                ft_stats = None
                final_circuit = prep_circuit

            if total_cost < best_local_cost:
                best_local_cost = total_cost
                best_base_cost = base_cost
                best_ft_stats = ft_stats

            if total_cost < self.global_best_cost:
                self.global_best_cost = total_cost
                self.global_best_circuit = final_circuit
                if self.verbose:
                    if self.current_iter >= self.warmup_iters:
                        print(
                            "-> [POST-SELECTED FT BEST] "
                            f"Total Cost: {total_cost:.1f} "
                            f"(Prep: {base_cost}, "
                            f"Decoder: {ft_stats['decoder_backend']}, "
                            f"Fault order: {ft_stats['fault_order']}, "
                            f"Sampling: {ft_stats['sampling_mode']}, "
                            f"FT Cost: {ft_stats['ft_cost']:.1f}, "
                            f"accepted harmful: {ft_stats['accepted_harmful_faults']}, "
                            f"rejected harmful: {ft_stats['rejected_harmful_faults']}, "
                            f"accepted tolerable: {ft_stats['accepted_tolerable_faults']}, "
                            f"rejected tolerable: {ft_stats['rejected_tolerable_faults']}, "
                            f"decoder fail: {ft_stats['decoder_failures']}, "
                            f"hook: {ft_stats['hook_faults']}, "
                            f"verification: {ft_stats['verification_gates']}, "
                        )
                    else:
                        print(f"-> [RAW-BEST]: {base_cost}")

        if best_local_cost == float("inf"):
            return 1e9

        if self.collect_data and (self.current_iter % self.collect_interval == 0):
            features = extract_state_features(node.matrix_state)
            sample = {
                "run_id": self.run_id,
                "iter": self.current_iter,
                "matrix_state": node.matrix_state.tolist(),
                "prefix_len": len(node.prefix_circuit),
                "depth": len(node.prefix_circuit),
                "total_cost": float(best_local_cost),
                "rank": features["rank"],
                "row_weights": features["row_weights"],
                "col_weights": features["col_weights"],
                "active_qubits": features["active_qubits"],
            }
            if best_ft_stats is not None:
                sample["phase"] = "ft"
                sample["ft_cost"] = float(best_ft_stats["ft_cost"])
                sample["base_cost"] = int(best_base_cost)
            else:
                sample["phase"] = "warmup"
                sample["ft_cost"] = None
                sample["base_cost"] = int(best_base_cost)
            self.training_samples.append(sample)

        return best_local_cost

    def backpropagate(self, node, cost):
        while node:
            node.visits += 1
            node.total_cost += cost
            node = node.parent

    def export_training_data(self, filepath):
        """Export collected training samples to JSON."""
        import json
        with open(filepath, "w") as f:
            json.dump(self.training_samples, f, indent=2)
        print(f"Exported {len(self.training_samples)} training samples to {filepath}")

    def collect_policy_samples(self, min_visits=10):
        """Traverse tree post-search to collect (state, visit_distribution) pairs
        for Policy Network training. Only records nodes with >= min_visits."""
        self.policy_samples = []

        def _traverse(node):
            if not node.children or node.visits < min_visits:
                return
            total_v = sum(c.visits for c in node.children)
            if total_v == 0:
                return
            visit_dist = [(c.action, c.visits / total_v) for c in node.children]
            self.policy_samples.append({
                "run_id": self.run_id,
                "matrix_state": node.matrix_state.tolist(),
                "visit_distribution": visit_dist,
            })
            for child in node.children:
                _traverse(child)

        _traverse(self.root)
        print(f"Collected {len(self.policy_samples)} policy samples "
              f"(min_visits={min_visits})")

    def _load_value_network(self, ckpt_path):
        """Load trained Value Network from checkpoint.
        Supports both SteaneValueNet (flatten+MLP) and RowColValueNet (row/col transformer).
        """
        if ckpt_path is None:
            ckpt_path = "checkpoints/steane_vn_remaining.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Value Network checkpoint not found: {ckpt_path}\n"
                f"Run train_value_network.py first.")
        ckpt = torch.load(ckpt_path, map_location=self._vn_device, weights_only=False)

        model_type = ckpt.get("model_type", "steane")
        if model_type == "rowcol_rank":
            model = RowColRankValueNet(
                embed_dim=ckpt.get("embed_dim", 128),
                nhead=ckpt.get("nhead", 4),
                num_layers=ckpt.get("num_layers", 2),
                hidden_dims=ckpt["config"]["hidden_dims"],
                max_size=ckpt.get("max_size", 200),
                dropout=0.0,
            )
            self._vn_is_rowcol = False
            self._vn_is_rowcol_rank = True
            self._vn_max_rank = float(ckpt.get("max_rank", 100.0))
        elif model_type == "rowcol":
            model = RowColValueNet(
                embed_dim=ckpt.get("embed_dim", 128),
                nhead=ckpt.get("nhead", 4),
                num_layers=ckpt.get("num_layers", 2),
                hidden_dims=ckpt["config"]["hidden_dims"],
                max_size=ckpt.get("max_size", 200),
                dropout=0.0,
            )
            self._vn_is_rowcol = True
            self._vn_is_rowcol_rank = False
        else:
            input_dim = ckpt["input_dim"]
            model = SteaneValueNet(
                input_dim=input_dim,
                hidden_dims=ckpt["config"]["hidden_dims"],
                dropout=0.0,
            )
            self._vn_is_rowcol = False
            self._vn_is_rowcol_rank = False

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._vn_device)
        model.eval()
        self.value_net = model
        self._vn_normalizer = CostNormalizer()
        self._vn_normalizer.load_state_dict(ckpt["normalizer"])
        metrics = ckpt.get("metrics", {})
        if self.verbose:
            print(f"[V-Net] Loaded from {ckpt_path}")
            if self._vn_is_rowcol:
                print(f"  Type: RowCol, Embed dim: {ckpt.get('embed_dim', 128)}")
            else:
                print(f"  Type: MLP, Input dim: {input_dim}")
            print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Train metrics: R2={metrics.get('r2', 'N/A'):.4f} "
                  f"Pearson={metrics.get('pearson_r', 'N/A'):.4f}")
            print(f"  Normalizer: mean={self._vn_normalizer.mean:.2f} "
                  f"std={self._vn_normalizer.std:.2f}")
            print(f"  Device: {self._vn_device}")

    def _load_policy_network(self, ckpt_path):
        """Load trained Policy Network from checkpoint."""
        if ckpt_path is None:
            ckpt_path = "checkpoints/steane_pn.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Policy Network checkpoint not found: {ckpt_path}\n"
                f"Run train_policy_network.py first.")
        ckpt = torch.load(ckpt_path, map_location=self._pn_device, weights_only=False)
        input_dim = ckpt["input_dim"]
        model = SteanePolicyNet(
            input_dim=input_dim,
            hidden_dims=ckpt["config"]["hidden_dims"],
            num_actions=ckpt["num_actions"],
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self._pn_device)
        model.eval()
        self.policy_net = model
        top1 = ckpt.get("top1_accuracy", "N/A")
        if self.verbose:
            print(f"[P-Net] Loaded from {ckpt_path}")
            print(f"  Input dim: {input_dim}, Actions: {ckpt['num_actions']}")
            print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Top-1 Accuracy: {top1:.4f}" if isinstance(top1, float) else f"  Top-1: {top1}")
            print(f"  PUCT c: {self.c_puct}")

    def select_puct(self, node):
        """PUCT selection: bias UCT towards policy-preferred actions.
        Higher P(s,a) → lower key → preferred child.
        Does NOT prune actions — all children remain eligible.
        """
        while node.children and node.is_fully_expanded:
            if getattr(self, '_vn_is_rowcol', False):
                state_tensor = torch.from_numpy(
                    node.matrix_state.astype(np.float32)
                ).float().to(self._pn_device)
            else:
                state_input = matrix_to_input(node.matrix_state, include_features=True)
                state_tensor = torch.from_numpy(state_input).float().to(self._pn_device)

            child_indices = [action_to_index(c.action) for c in node.children]

            with torch.no_grad():
                logits = self.policy_net(state_tensor.unsqueeze(0)).squeeze()
                cand_logits = logits[child_indices]
                probs = torch.softmax(cand_logits, dim=0)

            def _key(c, p):
                exploit = c.total_cost / (c.visits + 1e-6)
                explore = (self.c_puct * p * math.sqrt(node.visits + 1)
                           / (1 + c.visits))
                return exploit - explore

            pairs = list(zip(node.children, probs.tolist()))
            node = min(pairs, key=lambda cp: _key(cp[0], cp[1]))[0]
        return node

    def _single_rollout(self, node):
        """Perform exactly one rollout for periodic V-Net verification."""
        rollout_gates, final_pivots = self.solver.solve_remainder(
            node.matrix_state, randomize=True)
        if rollout_gates is None or final_pivots is None:
            return float("inf")
        prep_circuit = build_css_logical_zero_prep(
            node.prefix_circuit + rollout_gates, final_pivots)
        base_cost = len(prep_circuit)
        if self.current_iter >= self.warmup_iters:
            ft_stats = self.ft_evaluator.evaluate_postselected(prep_circuit)
            return base_cost + ft_stats["ft_cost"]
        return float(base_cost)

    def evaluate(self, node):
        """Value Network evaluation: predict remaining cost from current state.
        Replaces simulate() when use_value_net=True.
        Periodic verification rollout every 100 evaluations.
        """
        if getattr(self, '_vn_is_rowcol_rank', False):
            # RowColRankValueNet takes (matrix, rank) tuple
            st = torch.from_numpy(
                node.matrix_state.astype(np.float32)
            ).float().to(self._vn_device).unsqueeze(0)  # (1, R, C)
            rk = float(_gf2_rank(node.matrix_state)) / self._vn_max_rank
            rt = torch.tensor([[rk]], dtype=torch.float32, device=self._vn_device)
            state_tensor = (st, rt)
        elif getattr(self, '_vn_is_rowcol', False):
            # RowColValueNet takes raw matrix tensor
            state_tensor = torch.from_numpy(
                node.matrix_state.astype(np.float32)
            ).float().to(self._vn_device)
            state_tensor = state_tensor.unsqueeze(0)  # (1, R, C)
        else:
            # SteaneValueNet takes flat vector
            state_input = matrix_to_input(node.matrix_state, include_features=True)
            state_tensor = torch.from_numpy(state_input).float().to(self._vn_device)
            state_tensor = state_tensor.unsqueeze(0)  # (1, flat_dim)

        with torch.no_grad():
            if isinstance(state_tensor, tuple):
                v_norm = self.value_net(*state_tensor).item()
            else:
                v_norm = self.value_net(state_tensor).item()
        v_remaining = self._vn_normalizer.denormalize(v_norm)
        v_remaining = max(0.0, v_remaining)
        estimated_total = len(node.prefix_circuit) + v_remaining

        if estimated_total < self.global_best_predicted:
            self.global_best_predicted = estimated_total
            self.global_best_predicted_iter = self.current_iter

        self._vn_eval_count += 1
        if self._vn_eval_count % 100 == 0:
            true_cost = self._single_rollout(node)
            if true_cost != float("inf"):
                err = abs(estimated_total - true_cost)
                pred_remaining = v_remaining
                true_remaining = true_cost - len(node.prefix_circuit)
                if true_cost < self.global_best_cost:
                    self.global_best_cost = true_cost
                if self.verbose:
                    print(f"  [V-Net verify #{self._vn_eval_count}] "
                          f"iter={self.current_iter} pred_total={estimated_total:.1f} "
                          f"true_total={true_cost:.1f} err={err:.1f} "
                          f"pred_rem={pred_remaining:.1f} true_rem={true_remaining:.1f}")

        return estimated_total

    def run(self, record_interval=100):
        eval_fn = self.evaluate if self.use_value_net else self.simulate
        select_fn = self.select_puct if self.use_policy_net else self.select
        parts = []
        if self.use_value_net:
            parts.append("V-Net")
        if self.use_policy_net:
            parts.append("PUCT")
        mode = " + ".join(parts) if parts else "Baseline MCTS"
        if self.verbose:
            print(f"State preparating... [{mode}]")
        for i in range(self.iterations):
            self.current_iter = i
            if i == self.warmup_iters and not self.use_value_net:
                if self.verbose:
                    print("\n=============================================")
                    print("Starting Post-Selected FT Evaluation")
                    print("=============================================\n")
                self.global_best_cost = float("inf")

            leaf = select_fn(self.root)
            child = self.expand(leaf)
            cost = eval_fn(child)
            self.backpropagate(child, cost)

            if (i + 1) % record_interval == 0:
                entry = {
                    "iter": i + 1,
                    "best_cost": self.global_best_cost,
                }
                if self.use_value_net:
                    entry["best_predicted"] = self.global_best_predicted
                self.convergence_history.append(entry)

            if (i + 1) % 500 == 0 and self.verbose:
                msg = f"Epoch {i + 1}/{self.iterations} | Best Cost: {self.global_best_cost}"
                if (self.use_value_net
                        and self.global_best_predicted != float("inf")):
                    msg += (f" | VN Predicted: {self.global_best_predicted:.1f}"
                            f" @iter {self.global_best_predicted_iter}")
                print(msg)

        if self.collect_data:
            nsamp = len(self.training_samples)
            print(f"\nData collection complete: {nsamp} value network samples "
                  f"(interval={self.collect_interval})")

        return self.global_best_circuit


if __name__ == "__main__":
    USE_VN = "--vn" in sys.argv
    USE_PN = "--pn" in sys.argv
    target = "7_1_3_Steane_Code"
    config = QuantumCodeRegistry.get_code(target)
    parts = []
    if USE_VN:
        parts.append("V-Net")
    if USE_PN:
        parts.append("PUCT")
    mode_label = " + ".join(parts) if parts else "Baseline MCTS"
    print(f"Generating circuit for code: {target} [{mode_label}]")

    x_stabs_only, z_stabs_only = split_css_stabilizers(config["stabs"])
    if not x_stabs_only or not z_stabs_only:
        raise ValueError(
            f"Target code {target} is not a valid CSS code for logical-|0> preparation.")
    d = 3
    num_data = 7

    FT_EVALUATOR_MODE = "postselected"

    ft_evaluator = FTEvaluator(
        code_target=target, num_qubits=num_data, d=d,
        subfault_budget=10000, use_gpu=True, device="cuda:0",
        fast_decoder_type="uf", verify_interval=100,
    )
    env = GateLevelMCTSEnv(
        stabs=x_stabs_only, num_data=num_data,
        topo_edges=get_all_to_all_edges(num_data))

    vn_path = "checkpoints/steane_vn_remaining.pt" if USE_VN else None
    pn_path = "checkpoints/steane_pn.pt" if USE_PN else None
    mcts = QuantumGateMCTS(
        env, ft_evaluator, code_target=target,
        iterations=3500, rollouts=25, warmup_iters=3500, action_nums=10,
        use_value_net=USE_VN, value_net_path=vn_path,
        use_policy_net=USE_PN, policy_net_path=pn_path, c_puct=2.0,
    )

    import time
    t0 = time.time()
    best_circuit = mcts.run()
    elapsed = time.time() - t0

    suffix_parts = []
    if USE_VN:
        suffix_parts.append("vn")
    if USE_PN:
        suffix_parts.append("puct")
    suffix = "_".join(suffix_parts) if suffix_parts else "baseline"
    save_name = f"{target}_{FT_EVALUATOR_MODE}_{suffix}_circuit.pkl"
    with open(save_name, "wb") as f:
        pickle.dump(best_circuit, f)

    print(f"\n[{mode_label}] Elapsed: {elapsed:.1f}s")
    print(f"Best cost: {mcts.global_best_cost}")
    print(f"Saved best circuit to '{save_name}'")