# Final Validation Report

Date: 2026-07-04

Question: What does adding the Value Network to MCTS actually improve?

This report uses only existing artifacts plus static code audit. No experiment was run during report generation.

## 1. Correctness

Current verifier-backed evidence is sufficient only for Exp20 L0:

- `results/exp20_L0_mcts_raw.csv`: 20/20 valid circuits.
- `valid_rate = 1.0`
- `x_syndrome_error = 0.0` for every run.
- `z_syndrome_error = 0.0` for every run.
- `is_logical_zero = True` for every run.

For Exp14 and Exp17, verifier calls are present in code, but existing result files are stale/empty. For Exp11/12/13 and Exp20 non-L0 arms, current artifacts are insufficient or stale. Therefore, correctness beyond Exp20 L0: **无法证明**.

## 2. Quality

Verifier-backed Exp20 L0 quality:

- CNOT mean: 41.65
- CNOT std: 3.02
- CNOT min/max: 37 / 47
- Depth values in `exp20_L0_mcts_raw.csv`: mean 53.65, min/max 49 / 59
- All 20 runs are valid, so raw and valid-only CNOT means are identical for this arm.

However, there is no current same-budget, same-verifier, fully legal baseline in the artifacts. Because the required comparison set is incomplete, Value Network quality improvement over baseline: **无法证明**.

## 3. Efficiency

Efficiency fields have been added as optional diagnostics:

- `first_valid_iteration`
- `first_best_iteration`
- `search_time`
- `value_inference_time`
- `simulation_time`
- `node_count`
- per-100-iteration anytime CSV with best CNOT, best depth, and legality

These fields are only produced when `FINAL_VALIDATION_LOG_DIR` is set, so historical/default runs are not affected. Existing artifacts were produced before this logging was available, so search-time or convergence-speed improvement: **无法证明**.

## 4. Mechanism

Existing supporting diagnostics from Exp15/16/18/19 can help explain ranking behavior, distribution shift, interpretability, and prediction compression, but they do not by themselves prove MCTS benefit.

Mechanism claims require linking those diagnostics to valid-circuit MCTS outcomes under the same search budget and verifier. Current artifacts do not provide that link. Therefore, mechanism of Value Network contribution: **无法证明**.

## Conclusion

Under the stated criterion, Value Network has practical value only if, at the same search budget, same verifier, and 100% legal circuits, it significantly improves final CNOT, depth, search time, or convergence speed.

With current artifacts:

- Correctness is proven only for Exp20 L0.
- Quality improvement over a verified baseline is not proven.
- Efficiency improvement is not proven.
- Mechanism is not proven.

Final answer from current data: **无法证明 Value Network 对该任务具有实际收益**.
