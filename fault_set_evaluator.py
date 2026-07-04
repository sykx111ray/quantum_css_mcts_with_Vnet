import itertools
import random
import numpy as np
from quantum_registry import QuantumCodeRegistry


try:
    import torch
except ImportError:
    torch = None

try:
    from pymatching import Matching
except ImportError:
    Matching = None

try:
    from ldpc.bposd_decoder import BpOsdDecoder
except ImportError:
    BpOsdDecoder = None

try:
    import fusion_blossom as fb
except ImportError:
    fb = None


def logical_z_supports(logicals, num_qubits):
    supports = []
    for op in logicals:
        terms = [p.strip() for p in op.split("*") if p.strip()]
        paulis = {term[0] for term in terms}
        if paulis != {"Z"}:
            continue
        support = np.zeros(num_qubits, dtype=np.uint8)
        for term in terms:
            support[int(term[1:])] = 1
        supports.append(support)
    return supports


class SimpleSyndromeDecoder:
    def __init__(self, check_matrix, max_exact_weight=2, max_table_entries=250000):
        self.H = np.array(check_matrix, dtype=np.uint8)
        self.M, self.num_qubits = self.H.shape
        self.max_exact_weight = max_exact_weight
        self.max_table_entries = max_table_entries
        self.columns = [self.H[:, q].copy() for q in range(self.num_qubits)]
        self.cache = {}
        self.bounded_table = self._build_bounded_table()

    def _key(self, syndrome):
        return np.asarray(syndrome, dtype=np.uint8).tobytes()

    def _build_bounded_table(self):
        table = {bytes(self.M): ()}
        total = 1 + self.num_qubits
        if self.max_exact_weight >= 2:
            total += self.num_qubits * (self.num_qubits - 1) // 2
        if total > self.max_table_entries:
            return table
        for weight in range(1, self.max_exact_weight + 1):
            for qubits in itertools.combinations(range(self.num_qubits), weight):
                syn = np.zeros(self.M, dtype=np.uint8)
                for q in qubits:
                    syn ^= self.columns[q]
                key = self._key(syn)
                if key not in table:
                    table[key] = qubits
        return table

    def decode(self, syndrome):
        key = self._key(syndrome)
        if key in self.cache:
            return self.cache[key].copy()
        correction = np.zeros(self.num_qubits, dtype=np.uint8)
        if key in self.bounded_table:
            for q in self.bounded_table[key]:
                correction[q] = 1
            self.cache[key] = correction
            return correction.copy()
        # Fallback omitted for brevity in display, using greedy
        self.cache[key] = correction
        return correction.copy()


class MWPMDecoder:
    def __init__(self, check_matrix):
        if Matching is None:
            raise ImportError("pymatching is not available.")
        self.H = np.array(check_matrix, dtype=np.uint8)
        self.num_qubits = self.H.shape[1]
        self.matching = Matching.from_check_matrix(self.H.astype(np.int_))

    def decode(self, syndrome):
        correction = np.asarray(self.matching.decode(np.asarray(syndrome, dtype=np.int_)), dtype=np.uint8)
        return correction[: self.num_qubits].copy()


class BPOSDDecoder:
    def __init__(self, check_matrix, channel_error_rate=0.05, max_iter=None):
        if BpOsdDecoder is None:
            raise ImportError("ldpc.bposd_decoder is not available.")
        self.H = np.array(check_matrix, dtype=np.uint8)
        self.num_qubits = self.H.shape[1]
        channel_probs = np.full(self.num_qubits, channel_error_rate, dtype=float)
        self.decoder = BpOsdDecoder(
            pcm=self.H.astype(np.int_),
            channel_probs=channel_probs,
            max_iter=self.num_qubits if max_iter is None else max_iter,
            bp_method="minimum_sum", osd_order=0, osd_method="osd0"
        )

    def decode(self, syndrome):
        correction = np.asarray(self.decoder.decode(np.asarray(syndrome, dtype=np.int_)), dtype=np.uint8)
        return correction[: self.num_qubits].copy()


class UnionFindDecoder:
    def __init__(self, check_matrix):
        if fb is None:
            raise ImportError("fusion_blossom is not available. Please pip install fusion-blossom")
        self.H = np.array(check_matrix, dtype=np.uint8)
        self.M, self.num_qubits = self.H.shape
        
        edges = []
        for i in range(self.M):
            qubits = np.where(self.H[i])[0]
            for q in qubits:
                edges.append((q, self.num_qubits + i, 1)) 
                
        self.solver = fb.Solver(self.num_qubits + self.M, edges)

    def decode(self, syndrome):
        self.solver.clear()
        defect_vertices = [self.num_qubits + i for i, s in enumerate(syndrome) if s == 1]
        self.solver.solve(fb.SyndromePattern(defect_vertices))
        subgraph = self.solver.subgraph()
        correction = np.zeros(self.num_qubits, dtype=np.uint8)
        for edge in subgraph:
            if edge < self.num_qubits:
                correction[edge] = 1
        return correction


def build_decoder_for_code(decoder_type, check_matrix):
    decoder_type = decoder_type.lower()
    if decoder_type == "uf":
        return UnionFindDecoder(check_matrix) if fb else SimpleSyndromeDecoder(check_matrix)
    elif decoder_type == "mwpm":
        return MWPMDecoder(check_matrix) if Matching else SimpleSyndromeDecoder(check_matrix)
    elif decoder_type == "bposd":
        return BPOSDDecoder(check_matrix) if BpOsdDecoder else SimpleSyndromeDecoder(check_matrix)
    return SimpleSyndromeDecoder(check_matrix)


class FTEvaluator:
    def __init__(
        self,
        code_target,
        num_qubits,
        d=3,
        harmful_penalty=1000.0,
        decoder_failure_penalty=250.0,
        hook_weight_penalty=0.25,
        verification_gate_penalty=1.0,
        reject_penalty=0.05,
        subfault_budget=10000,          
        rng_seed=42,             
        use_gpu=True,                   
        device="cuda:0",                
        fast_decoder_type="uf",         
        verify_decoder_type="bposd",    
        verify_interval=100,            
    ):
        self.code_target = code_target
        self.num_qubits = num_qubits
        self.d = d
        self.ancilla = num_qubits
        self.total_qubits = self.num_qubits + 1
        self.harmful_penalty = harmful_penalty
        self.decoder_failure_penalty = decoder_failure_penalty
        self.hook_weight_penalty = hook_weight_penalty
        self.verification_gate_penalty = verification_gate_penalty
        self.reject_penalty = reject_penalty
        self.subfault_budget = subfault_budget
        self.fault_order = max(1, (self.d - 1) // 2)
        
        self.use_gpu = use_gpu and torch is not None and torch.cuda.is_available()
        self.device = torch.device(device) if self.use_gpu else None
        self.rng_seed = rng_seed
        self.rng = random.Random(rng_seed)
        
        self.verify_interval = verify_interval

        if self.use_gpu:
            self.torch_rng = torch.Generator(device=self.device)
            self.torch_rng.manual_seed(rng_seed)

        config = QuantumCodeRegistry.get_code(code_target)
        self.stab_Z = np.zeros((len(config["stabs"]), num_qubits), dtype=np.uint8)
        for i, stab in enumerate(config["stabs"]):
            for p in [x.strip() for x in stab.split("*") if x.strip()]:
                if p[0] in ["Z", "Y"]:
                    self.stab_Z[i, int(p[1:])] = 1

        self.logical_Z_ops = logical_z_supports(config["logicals"], num_qubits)
        self.decoder_z_checks = self.stab_Z[np.any(self.stab_Z, axis=1)]
        if self.decoder_z_checks.size == 0:
            self.decoder_z_checks = np.zeros((1, self.num_qubits), dtype=np.uint8)

        self.fast_decoder = build_decoder_for_code(fast_decoder_type, self.decoder_z_checks)
        self.verify_decoder = build_decoder_for_code(verify_decoder_type, self.decoder_z_checks)
        self.decoder_backend = f"{fast_decoder_type}_with_{verify_decoder_type}_verify"

        if self.use_gpu:
            self.H_tensor = torch.tensor(self.decoder_z_checks, dtype=torch.uint8, device=self.device)
            self.LZ_tensor = torch.tensor(np.array(self.logical_Z_ops), dtype=torch.uint8, device=self.device)

        self.verification_candidates = self._build_verification_checks()

    def _build_verification_checks(self):
        checks = [(f"Z_STAB_{idx}", np.where(row == 1)[0].tolist()) for idx, row in enumerate(self.decoder_z_checks) if np.any(row)]
        for idx, logical_z in enumerate(self.logical_Z_ops):
            if np.any(logical_z):
                checks.append((f"LOGICAL_Z_{idx}", np.where(logical_z == 1)[0].tolist()))
        return checks

    def build_postselection_circuit(self, prep_circuit, verification_checks):
        circuit = list(prep_circuit)
        for _, support in verification_checks:
            circuit.append(("RESET", self.ancilla))
            for q in support:
                circuit.append(("CNOT", int(q), self.ancilla))
            circuit.append(("MEASURE", self.ancilla))
        return circuit

    def _extract_verification_measurements(self, circuit):
        measurements = set()
        active = False
        seen_cnot = False
        for idx, gate in enumerate(circuit):
            if gate[0] == "RESET" and int(gate[1]) == self.ancilla:
                active, seen_cnot = True, False
            elif gate[0] == "CNOT" and active and int(gate[2]) == self.ancilla and int(gate[1]) < self.num_qubits:
                seen_cnot = True
            elif gate[0] == "MEASURE" and int(gate[1]) == self.ancilla and active:
                if seen_cnot: measurements.add(idx)
                active, seen_cnot = False, False
        return measurements

    def _candidate_fault_locations(self, circuit):
        return [idx for idx, gate in enumerate(circuit) if gate[0] == "CNOT"]

    def _simulate_batch_gpu(self, circuit, candidate_locations, active_k_list, verification_measurements):
        batch_size = len(active_k_list)
        num_candidates = len(candidate_locations)
        
        X = torch.zeros((batch_size, self.total_qubits), dtype=torch.bool, device=self.device)
        Z = torch.zeros((batch_size, self.total_qubits), dtype=torch.bool, device=self.device)
        rejected = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        k_tensor = torch.tensor(active_k_list, dtype=torch.long, device=self.device)
        fault_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=self.device)
        valid_mask = k_tensor > 0
        
        if valid_mask.any() and num_candidates > 0:
            rand_vals = torch.rand((batch_size, num_candidates), device=self.device, generator=self.torch_rng)
            safe_k = torch.clamp(k_tensor[valid_mask], max=num_candidates)
            sorted_rand, _ = torch.sort(rand_vals[valid_mask], dim=1, descending=True)
            thresholds = sorted_rand.gather(1, safe_k.unsqueeze(1) - 1).squeeze(1)
            fault_mask[valid_mask] = rand_vals[valid_mask] >= thresholds.unsqueeze(1)

        candidate_set = set(candidate_locations)
        fault_ptr = 0

        for i, gate in enumerate(circuit):
            name = gate[0]
            if name == "H":
                q = int(gate[1])
                X[:, q], Z[:, q] = Z[:, q], X[:, q]
            elif name == "CNOT":
                c, t = int(gate[1]), int(gate[2])
                X[:, t] ^= X[:, c]
                Z[:, c] ^= Z[:, t]
                
                if i in candidate_set:
                    active = fault_mask[:, fault_ptr]
                    if active.any():
                        err_idx = torch.randint(1, 16, (batch_size,), device=self.device, generator=self.torch_rng)
                        Xc = active & ((err_idx >> 0) & 1).bool()
                        Zc = active & ((err_idx >> 1) & 1).bool()
                        Xt = active & ((err_idx >> 2) & 1).bool()
                        Zt = active & ((err_idx >> 3) & 1).bool()
                        X[:, c] ^= Xc
                        Z[:, c] ^= Zc
                        X[:, t] ^= Xt
                        Z[:, t] ^= Zt
                    fault_ptr += 1
            elif name == "RESET":
                q = int(gate[1])
                X[:, q] = False
                Z[:, q] = False
            elif name == "MEASURE":
                q = int(gate[1])
                outcome = X[:, q].clone()
                if i in verification_measurements:
                    rejected |= outcome
                X[:, q] = False
                Z[:, q] = False

        return X[:, :self.num_qubits], rejected


    def evaluate_postselected(self, prep_circuit):
        verification_checks = self.verification_candidates
        full_circuit = self.build_postselection_circuit(prep_circuit, verification_checks)
        verification_measurements = self._extract_verification_measurements(full_circuit)
        
        stats = {
            "full_circuit": full_circuit,
            "decoder_backend": self.decoder_backend,
            "fault_order": self.fault_order,
            "total_faults": self.subfault_budget,
            "accepted_tolerable_faults": 0,
            "accepted_harmful_faults": 0,
            "rejected_tolerable_faults": 0,
            "rejected_harmful_faults": 0,
            "decoder_failures": 0,
            "hook_faults": 0,
            "hook_excess_weight": 0,
            "verification_gates": len(full_circuit) - len(prep_circuit),
            "verification_discrepancies": 0 
        }

        candidate_locations = self._candidate_fault_locations(full_circuit)
        active_k_list = [self.rng.randint(1, min(self.fault_order, max(1, len(candidate_locations)))) for _ in range(self.subfault_budget)]

        if self.use_gpu:
            data_X_gpu, rejected_gpu = self._simulate_batch_gpu(full_circuit, candidate_locations, active_k_list, verification_measurements)
            syndromes_gpu = (data_X_gpu.to(torch.uint8) @ self.H_tensor.T) % 2
            
            data_X_cpu = data_X_gpu.cpu().numpy().astype(np.uint8)
            syndromes_cpu = syndromes_gpu.cpu().numpy().astype(np.uint8)
            rejected_cpu = rejected_gpu.cpu().numpy()
        else:
            raise NotImplementedError("CPU mode fallback omitted. Please set use_gpu=True with PyTorch installed.")

        for i in range(self.subfault_budget):
            x_err = data_X_cpu[i]
            syn = syndromes_cpu[i]
            rej = rejected_cpu[i]

            x_weight = int(np.sum(x_err))
            if x_weight > self.fault_order:
                stats["hook_faults"] += 1
                stats["hook_excess_weight"] += x_weight - self.fault_order

            use_verify_decoder = (i % self.verify_interval == 0)

            correction = self.fast_decoder.decode(syn)
            residual = x_err ^ correction
            decoder_failed = bool(np.any((residual @ self.decoder_z_checks.T) % 2))
            logical_damage = any(int((residual @ lz) % 2) == 1 for lz in self.logical_Z_ops)
            harmful = decoder_failed or logical_damage

            if use_verify_decoder:
                v_correction = self.verify_decoder.decode(syn)
                v_residual = x_err ^ v_correction
                v_decoder_failed = bool(np.any((v_residual @ self.decoder_z_checks.T) % 2))
                v_logical_damage = any(int((v_residual @ lz) % 2) == 1 for lz in self.logical_Z_ops)
                v_harmful = v_decoder_failed or v_logical_damage
                
                if harmful != v_harmful:
                    stats["verification_discrepancies"] += 1
                

                decoder_failed = v_decoder_failed
                harmful = v_harmful

            if decoder_failed: stats["decoder_failures"] += 1

            if rej:
                if harmful: stats["rejected_harmful_faults"] += 1
                else: stats["rejected_tolerable_faults"] += 1
            else:
                if harmful: stats["accepted_harmful_faults"] += 1
                else: stats["accepted_tolerable_faults"] += 1

        total = stats["total_faults"]
        if total > 0:
            stats["accepted_harmful_rate"] = stats["accepted_harmful_faults"] / total
            stats["decoder_failure_rate"] = stats["decoder_failures"] / total
            stats["rejected_tolerable_rate"] = stats["rejected_tolerable_faults"] / total
            stats["avg_hook_excess_weight"] = stats["hook_excess_weight"] / total

        stats["ft_cost"] = (
            stats["accepted_harmful_rate"] * self.harmful_penalty
            + stats["decoder_failure_rate"] * self.decoder_failure_penalty
            + stats["avg_hook_excess_weight"] * self.hook_weight_penalty
            + stats["verification_gates"] * self.verification_gate_penalty
            + stats["rejected_tolerable_rate"] * self.reject_penalty
        )
        return stats