"""
quantum_registry.py — 量子纠错码注册表

调用 QuantumCodeRegistry.get_code(name) 即可获得指定码的 stabilizer 列表和逻辑算符列表。

返回格式：
    {
        "stabs":    List[str],  # stabilizer 字符串，如 "X0*X1*X3"
        "logicals": List[str],  # 逻辑 Z 算符字符串， 如 "Z0*Z1*Z2"
    }

Stabilizer 字符串格式：
    - 每个 Pauli 算符写成 <类型><qubit索引>，例如 X0、Z3
    - 多个算符用 * 连接，例如 "X0*X1*Z2*Z3"
    - 索引从 0 开始

CSS:
    CSS(Calderbank-Shor-Steane)码由两个经典线性码C1, C2构造
    满足 C2⊥ ⊆ C1。其校验矩阵满足 Hx·Hz^T ≡ 0 (mod 2)
    保证 X-type 和 Z-type stabilizer 两两对易。
    n个物理qubit rank(Hx) + rank(Hz) 个 stabilizer
    k = n - rank(Hx) - rank(Hz) 个逻辑 qubit。
"""

from pathlib import Path
import sys
import numpy as np

class QuantumCodeRegistry: # 所有方法均为 classmethod 不需要实例化。

    # 外部只需调用 get_code(name) 获取码参数。
    @classmethod
    def get_code(cls, name):
        """
        根据名称返回量子码的stabilizer列表和逻辑算符列表。

        Args:
            name (str): 码的名称字符串，规则如下：
                - "5_1_3_Perfect_Code"            → [[5,1,3]] 完美码
                - "7_1_3_Steane_Code"             → [[7,1,3]] Steane 码（最常用的 CSS 码）
                - "15_1_3_Reed_Muller_Code"       → [[15,1,3]] Reed-Muller 码
                - "15_1_3_Hamming_Code"           → 同上（别名）
                - "15_7_3_Hamming_Code"           → [[15,7,3]] Hamming CSS 码
                - "25_1_5_Rotated_Surface_Logical_0" → d=5 旋转表面码(格式：{n}_{k}_{d}_Rotated_Surface_Logical_0)
                - "72_12_6_BB_Code"               → [[72,12,6]] 双变量自行车码(格式：{n}_..._BB_Code)
                - "144_12_12_Gross_Code"          → [[144,12,12]] Gross 码

        Returns:
            dict: {"stabs": List[str], "logicals": List[str]}

        Raises:
            ValueError: 名称不在注册表中时返回
        """
    
        if name == "5_1_3_Perfect_Code": #非 CSS 码，stabilizer 中同时含 X 和 Z
            return {
                "stabs": ["X0*Z1*Z2*X3", "X1*Z2*Z3*X4", "X2*Z3*Z4*X0", "X3*Z4*Z0*X1"],
                "logicals": ["Z0*Z1*Z2*Z3*Z4"]
            }

        if name == "7_1_3_Steane_Code":
            return {
                "stabs": [
                    "X0*X1*X2*X3",   # X-stabilizer 对应经典 Hamming 码第 1 行
                    "X1*X2*X4*X5",   # X-stabilizer 对应经典 Hamming 码第 2 行
                    "X2*X3*X5*X6",   # X-stabilizer 对应经典 Hamming 码第 3 行
                    "Z0*Z1*Z2*Z3",   # Z-stabilizer（与 X 结构相同，CSS 对称）
                    "Z1*Z2*Z4*Z5",
                    "Z2*Z3*Z5*Z6",
                ],
                "logicals": ["Z0*Z1*Z2*Z3*Z4*Z5*Z6"]
            }

        if name in ["15_1_3_Reed_Muller_Code", "15_1_3_Hamming_Code"]:
            return cls._generate_15_1_3_reed_muller()
        
        if name == "15_7_3_Hamming_Code":
            return cls._generate_15_7_3_hamming()
        
         # 从名称中解析码距d，调用生成函数。
        if name.endswith("_Rotated_Surface_Logical_0"):
            try:
                parts = name.split("_")
                d = int(parts[2])  # 取第 3 个字段作为码距 d
                return cls._generate_rotated_surface_code(d)
            except Exception as e:
                raise ValueError(
                    f"无法解析表面码参数，请确保格式如 '25_1_5_Rotated_Surface_Logical_0'. 错误: {e}"
                )

        # 支持 n ∈ {72, 90, 108, 144, 288}（已预置各自的代数参数）。
        if name.endswith("_BB_Code"):
            try:
                n = int(name.split("_")[0])  # 取名称第一段作为物理 qubit 总数
                return cls._generate_bivariate_bicycle_code(n)
            except Exception as e:
                raise ValueError(
                    f"无法解析 BB 码参数，请确保格式如 '72_12_6_BB_Code'. 错误: {e}"
                )

        # Gross 码本质上是参数 n=144 的双变量自行车码。
        if name.endswith("_Gross_Code"):
            try:
                n = int(name.split("_")[0])
                return cls._generate_gross_code(n)
            except Exception as e:
                raise ValueError(
                    f"无法解析 Gross 码参数，请确保格式如 '144_12_12_Gross_Code'. 错误: {e}"
                )

        raise ValueError(f"Unknown code in registry: {name}")

    # 内部生成函数
    @classmethod
    def _generate_15_1_3_reed_muller(cls):
        """
          Reed-Muller 码 RM(1,4) 的校验矩阵作 Hx
          RM(1,4)^⊥ = RM(2,4) 的非零码字作 Z-stabilizer。
          满足 RM(1,4) ⊂ RM(2,4)，即 C2⊥ ⊆ C1 符合 CSS 构造条件。

        Returns:
            dict: {"stabs": List[str], "logicals": [str]}
        """
        stabs = []

        # 构造 4 个 X-stabilizers
        # H[i][j] = ((j+1) >> i) & 1：经典 [15,11,3] Hamming 码的校验矩阵
        # 第 i 行（i=0,1,2,3）覆盖所有 j+1 在二进制第 i 位为 1 的 qubit
        # 示例（i=0）：qubit 0,2,4,6,8,10,12,14（j+1 = 奇数）
        for i in range(4):
            row = [((j + 1) >> i) & 1 for j in range(15)]
            stab = "*".join([f"X{j}" for j in range(15) if row[j]])
            stabs.append(stab)

        # 构造 11 个 Z-stabilizers
        # 找到 Hamming 码的 "自由列"（信息位位置）
        #   [15,11,3] Hamming 码有 4 个校验位（j+1 是 2 的幂次：1,2,4,8）
        #   其余 11 列是信息位（"free_indices"）
        basis_vectors = []
        free_indices = [j for j in range(15) if (j + 1) not in (1, 2, 4, 8)]

        # 对每个自由列 f，构造对应的码字（codeword）
        #   - vec[f] = 1（自由变量为 1，其余自由变量为 0）
        #   - vec[校验位] = ((f+1) >> i) & 1（根据 Hamming 码校验规则反推校验位）
        for f in free_indices:
            vec = [0] * 15
            vec[f] = 1  # 自由变量本身
            for i in range(4):
                if ((f + 1) >> i) & 1:
                    vec[(1 << i) - 1] = 1  # 对应校验位置位：位置 2^i - 1
            basis_vectors.append(vec)

        # 按奇偶性分组
        #   偶重量码字可直接作 Z-stabilizer（与所有 X-stabilizer 对易）
        #   奇重量码字需要两两异或转换为偶重量
        even_vectors = []
        odd_vectors = []
        for vec in basis_vectors:
            if sum(vec) % 2 == 0:
                even_vectors.append(vec)
            else:
                odd_vectors.append(vec)

        # 将奇重量向量两两合并为偶重量（异或后重量变为偶数）
        #   取第一个奇重量向量作为基准，与其余每个异或
        base_odd = odd_vectors[0]
        for ov in odd_vectors[1:]:
            even_vectors.append([(a + b) % 2 for a, b in zip(ov, base_odd)])

        # 将偶重量向量转换为 Z-stabilizer 字符串
        for vec in even_vectors:
            stab_qubits = [f"Z{j}" for j in range(15) if vec[j]]
            stabs.append("*".join(stab_qubits))

        # 逻辑 Z 算符
        # 全体 15 个 qubit 的 Z 乘积，权重 15（最小逻辑算符权重 = 码距 3）
        logical_z = "*".join([f"Z{j}" for j in range(15)])

        return {
            "stabs": stabs,
            "logicals": [logical_z]
        }

    @classmethod
    def _generate_15_7_3_hamming(cls):
        """
        Returns:
            dict: {"stabs": List[str], "logicals": List[str]} 7 个逻辑Z算符
        """
        stabs = []

        # 构造 4 个 X-stabilizers
        # 与 Reed-Muller 码的 X-stabilizer 构造完全相同
        for i in range(4):
            row = [((j + 1) >> i) & 1 for j in range(15)]
            stab = "*".join([f"X{j}" for j in range(15) if row[j]])
            stabs.append(stab)

        # 构造 4 个 Z-stabilizers（CSS 对称结构：Hx = Hz）
        for i in range(4):
            row = [((j + 1) >> i) & 1 for j in range(15)]
            stab = "*".join([f"Z{j}" for j in range(15) if row[j]])
            stabs.append(stab)

        # 构造 Hx 和 Hz 的矩阵形式，供 _compute_css_logicals() 使用
        Hx = np.zeros((4, 15), dtype=np.int8)
        Hz = np.zeros((4, 15), dtype=np.int8)

        # 从已生成的 stabilizer 字符串反解矩阵（前 4 个是 X，后 4 个是 Z）
        for i, stab in enumerate(stabs[:4]):
            qubits = [int(q[1:]) for q in stab.split("*")]
            Hx[i, qubits] = 1

        for i, stab in enumerate(stabs[4:]):
            qubits = [int(q[1:]) for q in stab.split("*")]
            Hz[i, qubits] = 1

        # 使用通用算法计算7个真正独立的逻辑 Z 算符
        # _compute_css_logicals() 通过计算 ker(Hx) \ row(Hz) 的基，
        # 保证 rank([Hz; logicals]) = 11 = 15 - rank(Hx)
        logical_vectors = cls._compute_css_logicals(Hx, Hz)

        # 将向量转换为字符串格式
        logicals = []
        for vec in logical_vectors:
            qubits = [f"Z{i}" for i in range(15) if vec[i]]
            logicals.append("*".join(qubits))

        return {
            "stabs": stabs,
            "logicals": logicals
        }

    @classmethod
    def _generate_rotated_surface_code(cls, d):
        """
        Args:
            d (int): 码距，必须为奇数（如 3, 5, 7, 9...）

        Returns:
            dict: {"stabs": List[str], "logicals": [str]}
              逻辑 Z 为第一行所有 qubit 的 Z 链（权重 d，横向路径）

        Raises:
            ValueError: d 为偶数时抛出。
        """
        if d % 2 == 0:
            raise ValueError("Rotated Surface Code distance 'd' must be an odd integer.")

        stabs = []
        n = d * d  # 物理 qubit 总数

        # 内部方格 stabilizer（权重 4）
        # 遍历 (d-1)×(d-1) 个内部方格
        # qubit 编号：第 r 行第 c 列 → r*d + c
        for r in range(d - 1):
            for c in range(d - 1):
                # 获取 2×2 小方格的 4 个顶点 qubit 编号
                q_top_left = r * d + c
                q_top_right = r * d + c + 1
                q_bottom_left = (r + 1) * d + c
                q_bottom_right = (r + 1) * d + c + 1

                # 棋盘染色：(r+c) 为偶数 → Z-面，奇数 → X-面
                # 这种染色保证相邻方格类型不同，从而 X 和 Z stabilizer 不共享 qubit
                if (r + c) % 2 == 0:
                    stabs.append(
                        f"Z{q_top_left}*Z{q_top_right}*Z{q_bottom_left}*Z{q_bottom_right}"
                    )
                else:
                    stabs.append(
                        f"X{q_top_left}*X{q_top_right}*X{q_bottom_left}*X{q_bottom_right}"
                    )

        # 左右边界 Z-边（权重 2）
        # 旋转表面码的左右边界放置 Z-stabilizer（连接上下相邻 qubit）
        # 棋盘染色规则：奇数行在左边界，偶数行在右边界
        for r in range(d - 1):
            if r % 2 == 1:  # 左边界：奇数行
                q1, q2 = r * d, (r + 1) * d
                stabs.append(f"Z{q1}*Z{q2}")
            if r % 2 == 0:  # 右边界：偶数行
                q1, q2 = r * d + (d - 1), (r + 1) * d + (d - 1)
                stabs.append(f"Z{q1}*Z{q2}")

        # 上下边界 X-边（权重 2）
        # 旋转表面码的上下边界放置 X-stabilizer（连接左右相邻 qubit）
        # 棋盘染色规则：偶数列在上边界，奇数列在下边界
        for c in range(d - 1):
            if c % 2 == 0:  # 上边界：偶数列
                q1, q2 = c, c + 1
                stabs.append(f"X{q1}*X{q2}")
            if c % 2 == 1:  # 下边界：奇数列
                q1, q2 = (d - 1) * d + c, (d - 1) * d + c + 1
                stabs.append(f"X{q1}*X{q2}")

        # 逻辑 Z 算符
        # 第一行 qubit 的 Z 链（横向路径从左到右），权重 = d = 码距
        # 这是最小权重的非平凡逻辑 Z 算符之一
        logical_z = "*".join([f"Z{c}" for c in range(d)])

        return {
            "stabs": stabs,
            # 逻辑 Z 作为需要被稳定的目标，用于制备逻辑 |0⟩ 态
            "logicals": [logical_z]
        }

    @classmethod
    def _generate_bivariate_bicycle_code(cls, n):
        """
        支持的 n 值（物理 qubit 数）：
          n=72  → [[72,12,6]]   参数 m=6,  l=6
          n=90  → [[90,10,10]]  参数 m=15, l=3 
          n=108 → [[108,12,6]]  参数 m=9,  l=6
          n=144 → [[144,12,12]] 参数 m=12, l=6 Gross 码
          n=288 → [[288,12,12]] 参数 m=12, l=12

        Args:
            n (int): 物理 qubit 总数，必须是支持的值之一

        Returns:
            dict: {"stabs": List[str], "logicals": List[str]}（仅返回第一个逻辑 Z 算符）

        Raises:
            ValueError: n 不支持时，或未能生成有效逻辑算符时抛出。
        """
        # 构造 Hx 和 Hz 校验矩阵（numpy int8 数组）
        hx, hz = cls._construct_bb_hx_hz(n)

        # 使用通用算法计算独立逻辑 Z 算符
        lz = cls._compute_css_logicals(hx, hz)

        # 将 Hx 的每行转换为 X-stabilizer 字符串
        stabs = []
        for row in hx:
            idx = np.where(row == 1)[0]
            if idx.size > 0:
                stabs.append("*".join(f"X{i}" for i in idx))

        # 将 Hz 的每行转换为 Z-stabilizer 字符串
        for row in hz:
            idx = np.where(row == 1)[0]
            if idx.size > 0:
                stabs.append("*".join(f"Z{i}" for i in idx))

        # 取第一个逻辑 Z 算符（当前只使用一个）
        # BB 码一般有多个逻辑 qubit，此处只返回第一个供电路合成使用
        logicals = []
        if lz.shape[0] > 0:
            idx = np.where(lz[0] == 1)[0]
            if idx.size > 0:
                logicals.append("*".join(f"Z{i}" for i in idx))

        if not logicals:
            raise ValueError(f"BB code n={n} 未生成可用 Logical Z。")

        return {
            "stabs": stabs,
            "logicals": logicals
        }

    @classmethod
    def _generate_gross_code(cls, n):
        """
        Gross 码本质上是参数 n=144 的双变量自行车码 BB码

        Returns:
            dict: {"stabs": List[str], "logicals": List[str]}

        Raises:
            ValueError: n ≠ 144 时抛出。
        """
        # Common usage in recent qLDPC experiments refers to [[144,12,12]] Gross code.
        # We construct it through the same BB matrix recipe used for n=144.
        if n != 144:
            raise ValueError(
                "Currently supported Gross code size is n=144 (e.g., '144_12_12_Gross_Code')."
            )
        return cls._generate_bivariate_bicycle_code(144)

    # BB 码矩阵构造辅助函数
    @classmethod
    def _construct_bb_hx_hz(cls, n):
        """
        Args:
            n (int): 物理 qubit 总数，支持 {72, 90, 108, 144, 288}

        Returns:
            tuple: (hx, hz)，均为 numpy int8 数组，形状 (n//2, n)

        Raises:
            ValueError: n 不在支持列表中时抛出。
        """
        # 预置各 n 值对应的代数参数：(m, l, a_terms, b_terms)
        # a_terms = ([x 的幂次列表], [y 的幂次列表])
        # b_terms = ([x 的幂次列表], [y 的幂次列表])
        if n == 72:
            m, l_, a_terms, b_terms = 6, 6, ([3], [1, 2]), ([1, 2], [3])
        elif n == 90:
            m, l_, a_terms, b_terms = 15, 3, ([9], [1, 2]), ([2, 7], [0])
        elif n == 108:
            m, l_, a_terms, b_terms = 9, 6, ([3], [1, 2]), ([1, 2], [3])
        elif n == 144:
            m, l_, a_terms, b_terms = 12, 6, ([3], [1, 2]), ([1, 2], [3])
        elif n == 288:
            m, l_, a_terms, b_terms = 12, 12, ([3], [2, 7]), ([1, 2], [3])
        else:
            raise ValueError(
                f"No bb code with n = {n}. Supported: 72, 90, 108, 144, 288."
            )

        # 构造矩阵 A 和 B（在 GF(2) 上）
        a = cls._cyclic_matrix_sum(a_terms[0], a_terms[1], l_, m) % 2
        b = cls._cyclic_matrix_sum(b_terms[0], b_terms[1], l_, m) % 2

        # Hx = [A | B]，Hz = [B^T | A^T]
        hx = np.hstack((a, b)).astype(np.int8)
        hz = np.hstack((b.T, a.T)).astype(np.int8)
        return hx, hz

    @staticmethod
    def _shift_matrix(length):
        """
        构造 length*length 的循环移位矩阵S 右移1位。

        S[i, (i+1) % length] = 1 其余为 0。
        S 的幂次 S^k 表示循环右移 k 位的置换矩阵。

        Args:
            length (int): 矩阵维度

        Returns:
            numpy.ndarray: shape (length, length) 的 int8 矩阵
        """
        s = np.zeros((length, length), dtype=np.int8)
        for i in range(length):
            s[i, (i + 1) % length] = 1
        return s

    @classmethod
    def _x_matrix(cls, l_, m):
        """
        构造 BB 码中 x 方向的基础置换矩阵 X = S_l ⊗ I_m。

        X 对应 Z_l*Z_m 群中的 x 生成元（在 l 方向循环移位 m 方向不动）
        Kronecker 积 X = S_l ⊗ I_m 形状 (l*m, l*m)。

        Args:
            l_ (int): l 方向的循环群阶数
            m (int):  m 方向的循环群阶数

        Returns:
            numpy.ndarray: shape (l*m, l*m) 的 int8 矩阵
        """
        return np.kron(cls._shift_matrix(l_), np.eye(m, dtype=np.int8)).astype(np.int8)

    @classmethod
    def _y_matrix(cls, l_, m):
        """
        构造 BB 码中 y 方向的基础置换矩阵 Y = I_l ⊗ S_m。

        Y 对应 Z_l * Z_m 群中的 y 生成元 l 方向不动 m 方向循环移位。
        Kronecker 积 Y = I_l ⊗ S_m 形状 (l*m, l*m)。

        Args:
            l_ (int): l 方向的循环群阶数
            m (int):  m 方向的循环群阶数

        Returns:
            numpy.ndarray: shape (l*m, l*m) 的 int8 矩阵
        """
        return np.kron(np.eye(l_, dtype=np.int8), cls._shift_matrix(m)).astype(np.int8)

    @classmethod
    def _cyclic_matrix_sum(cls, x_terms, y_terms, l_, m):
        """
        计算 BB 码代数多项式在 GF(2) 上的矩阵求和。

        计算 ∑_{p ∈ x_terms} X^p + ∑_{q ∈ y_terms} Y^q 在 GF(2) 上，即 XOR

        其中 X = S_l ⊗ I_m Y = I_l ⊗ S_m 是两个正交方向的循环移位矩阵。

        Args:
            x_terms (List[int]): X 方向的幂次列表
            y_terms (List[int]): Y 方向的幂次列表
            l_ (int): l 方向的循环群阶数
            m (int):  m 方向的循环群阶数

        Returns:
            numpy.ndarray: shape (l*m, l*m) 的 int8 矩阵（GF(2) 结果）
        """
        x = cls._x_matrix(l_, m)
        y = cls._y_matrix(l_, m)
        acc = np.zeros(x.shape, dtype=np.int8)

        # XOR 各 x 方向幂次
        for p in x_terms:
            acc ^= np.linalg.matrix_power(x, p).astype(np.int8)

        # XOR 各 y 方向幂次
        for p in y_terms:
            acc ^= np.linalg.matrix_power(y, p).astype(np.int8)

        return acc

    
    # CSS 逻辑算符计算（GF(2) 线性代数）
    @classmethod
    def _compute_css_logicals(cls, hx, hz):
        """
        计算 CSS 码的独立逻辑 Z 算符集合。

        数学原理：
          逻辑Z算符需满足
            (1) 与所有 X-stabilizer 对易 l·Hx^T ≡ 0 (mod 2) → l ∈ ker(Hx)
            (2) 不等价于任意 Z-stabilizer l ∉ row(Hz)
          因此，逻辑 Z 算符空间 = ker(Hx) / row(Hz)（商空间），
          独立代表元的数量 = k = dim(ker(Hx)) - rank(Hz_restricted)
                                 = (n - rank(Hx)) - rank(Hz)

        Args:
            hx (numpy.ndarray): X-type 校验矩阵 shape (rx, n) dtype int8
            hz (numpy.ndarray): Z-type 校验矩阵 shape (rz, n) dtype int8

        Returns:
            numpy.ndarray: 逻辑 Z 算符矩阵 shape (k, n) dtype int8
              每行是一个独立逻辑 Z 算符的二进制向量 1 表示对应 qubit 上有 Z
              若 k=0 则返回 shape (0, n) 的空数组
        """
        # 步骤 1：计算 ker(Hx)（Hx 的零空间）
        ker_hx = cls._nullspace_mod2(hx)

        # 步骤 2：计算 row(Hz)（Hz 的行空间，即行基）
        row_hz = cls._row_basis_mod2(hz)

        if ker_hx.size == 0:
            # ker(Hx) 为空 → 没有逻辑算符
            return np.zeros((0, hx.shape[1]), dtype=np.int8)

        # 步骤 3：堆叠 row(Hz) 和 ker(Hx)
        # 顺序：row_hz 在前（行索引 0~start-1），ker_hx 在后（行索引 start~末尾）
        stack = np.vstack([row_hz, ker_hx]) if row_hz.size else ker_hx

        # 步骤 4：对转置后的矩阵做 RREF，找出主元列
        # 转置的目的：我们需要找"列独立"的行，等价于找转置后"行独立"的行
        rref, pivots = cls._rref_mod2(stack.T)

        # 步骤 5：只保留主元对应 ker(Hx) 部分的行（排除 row_hz 贡献的行）
        start = row_hz.shape[0] if row_hz.size else 0
        idx = [i for i in range(start, stack.shape[0]) if i in pivots]

        return stack[idx] if idx else np.zeros((0, hx.shape[1]), dtype=np.int8)

    @staticmethod
    def _rref_mod2(mat):
        """
        在 GF(2)模 2 域 上对矩阵做RREF。

        GF(2) 特点：加法 = XOR 乘法 = AND -1 = 1。
        所有行操作均通过 XOR ^= 实现。

        Args:
            mat (numpy.ndarray): 输入矩阵，任意形状

        Returns:
            tuple: (rref_mat, pivots)
              rref_mat (numpy.ndarray): 行简化阶梯形矩阵，与输入同形状
              pivots (List[int]): 主元所在的列索引列表（按顺序）
        """
        a = mat.copy().astype(np.int8) % 2
        rows, cols = a.shape
        pivots = []
        r = 0  # 当前处理的行号

        for c in range(cols):
            # 在当前列 c 中，从第 r 行往下找主元（第一个非零元素）
            pivot = None
            for rr in range(r, rows):
                if a[rr, c]:
                    pivot = rr
                    break

            if pivot is None:
                continue  # 当前列全为 0，跳过

            # 将主元行移到第 r 行（行交换）
            if pivot != r:
                a[[r, pivot]] = a[[pivot, r]]

            # 消元：让当前列所有其他行（含上方行）变为 0
            for rr in range(rows):
                if rr != r and a[rr, c]:
                    a[rr] ^= a[r]  # GF(2) 行加法 = XOR

            pivots.append(c)
            r += 1
            if r == rows:
                break

        return a, pivots

    @classmethod
    def _row_basis_mod2(cls, mat):
        """
        在 GF(2) 上计算矩阵的行空间基（即行基）。

        通过 RREF 变换后，非零行构成行空间的一组基。

        Args:
            mat (numpy.ndarray): 输入矩阵

        Returns:
            numpy.ndarray: 行基矩阵 shape (rank, n) dtype int8
              若矩阵为零矩阵则返回 shape (0, n) 的空数组
        """
        rref, _ = cls._rref_mod2(mat)

        # RREF 后非零行即为行基
        non_zero = np.where(rref.any(axis=1))[0]
        if non_zero.size == 0:
            return np.zeros((0, mat.shape[1]), dtype=np.int8)
        return rref[non_zero]

    @classmethod
    def _nullspace_mod2(cls, mat):
        """
        在 GF(2) 上计算矩阵的零空间（核空间）基。
        零空间 ker(mat) = {v : mat·v ≡ 0 (mod 2)}。

        Args:
            mat (numpy.ndarray): 输入矩阵，shape (m, n)

        Returns:
            numpy.ndarray: 零空间基矩阵，shape (nullity, n)，dtype int8
              nullity = n - rank(mat)
              若零空间为空（full rank）则返回 shape (0, n) 的空数组
        """
        rref, pivots = cls._rref_mod2(mat)
        rows, cols = rref.shape
        pivot_set = set(pivots)

        # 自由列（对应零空间维度的每一个方向）
        free_cols = [c for c in range(cols) if c not in pivot_set]
        if not free_cols:
            # 满秩矩阵，零空间只有零向量
            return np.zeros((0, cols), dtype=np.int8)

        basis = []
        for free in free_cols:
            # 构造对应自由列 free 的零空间基向量
            vec = np.zeros(cols, dtype=np.int8)
            vec[free] = 1  # 自由变量置为 1

            # 通过 RREF 的主元行反向替换，填充主元列的值
            pivot_row = 0
            for c in range(cols):
                if c in pivot_set:
                    # 主元列的值 = RREF 中第 pivot_row 行在 free 列处的值
                    vec[c] = rref[pivot_row, free]
                    pivot_row += 1

            basis.append(vec)

        return np.array(basis, dtype=np.int8)

# 快速验证入口
if __name__ == '__main__':
    # 测试几个典型码的生成是否正常
    targets = [
        "15_1_3_Reed_Muller_Code",
        "15_7_3_Hamming_Code",
        "25_1_5_Rotated_Surface_Logical_0",
    ]

    for target in targets:
        config = QuantumCodeRegistry.get_code(target)

        print(f"\n=== {target} 验证 ===")
        print(f"共生成 Stabilizer 数量: {len(config['stabs'])}")
        print(f"Logical Operator(s) 数量: {len(config['logicals'])}")
        print(f"Logical Operator 示例: {config['logicals'][0]}")

        print("Stabilizer 示例:")
        for stab in config['stabs'][:5]:
            print("  " + stab)
