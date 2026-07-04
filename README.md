# V-Net: Learning to Approximate Monte Carlo Rollout Evaluation

## 一句话
用神经网络（Value Network）近似 MCTS rollout 评估，加速 CSS 量子纠错码的逻辑态制备。

## 核心结论
在 Surface d=5 和 d=9 上，V-Net 未显著提升 MCTS CNOT/depth（p > 0.4），但所有产出电路 100% 满足 stabilizer 约束。

## 项目结构
- `experiment_14/17/20/22` — 核心实验（RowCol Transformer / Loss Ablation / Baseline vs V-Net）
- `utils/circuit_verifier.py` — CSS 电路正确性验证
- `quantum_registry.py` — 支持的量子码（Steane, Surface d=3/5/7/9, BB72 等）
- `value_network.py` — V-Net 架构（Flatten MLP + RowCol Transformer）
- `results/` — 所有实验的 CSV/summary 输出

## 快速复现
见 `results/surface_d9_preparation_report.md`
