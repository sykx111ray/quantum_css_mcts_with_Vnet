# Final Validation Audit Report

Date: 2026-07-04

Scope: Exp11 through Exp20. This is a static audit of the current repository and existing result artifacts. No training, MCTS, or verifier execution was performed.

## Summary

| Exp | init_matrix uses only X stabilizers | Verifier already applied in available results | Final circuit legal | Re-run needed |
| --- | --- | --- | --- | --- |
| Exp11 | Yes | No | Unknown | Yes |
| Exp12 | Yes | No | Unknown | Yes |
| Exp13 | Yes | No | Unknown | Yes |
| Exp14 | Yes | Code path yes; existing result file is stale/empty | Unknown from current artifacts | Yes |
| Exp15 | Yes | No final-circuit verifier output | Not applicable/unknown | No for final-circuit MCTS, unless used as supporting mechanism data |
| Exp16 | Yes | No final-circuit verifier output | Not applicable/unknown | No for final-circuit MCTS, unless used as supporting mechanism data |
| Exp17 | Yes | Code path yes; existing result file is stale/empty | Unknown from current artifacts | Yes |
| Exp18 | Yes | No final-circuit verifier output | Not applicable/unknown | No for final-circuit MCTS, unless used as supporting mechanism data |
| Exp19 | Yes | No final-circuit verifier output | Not applicable/unknown | No for final-circuit MCTS, unless used as supporting mechanism data |
| Exp20 | Yes | Partially: L0 current artifacts include verifier; L2/L4h summaries are stale | L0: legal in 20/20 runs; others unknown | Yes for non-L0 arms and for efficiency logging |

## Evidence

- Exp11: `experiment_11_representation_ablation.py` builds `x_stabs = [s for s in config["stabs"] if s.startswith("X")]` and fills `init_matrix` from those rows only. Existing `results/repr_ablation_mcts_raw.csv` has no verifier columns.
- Exp12: `experiment_12_representation_sweep.py` builds `x_stabs` from `startswith("X")` and uses those rows in `MCTSEnv`. Existing `results/repr_sweep_summary.csv` has no verifier columns.
- Exp13: `experiment_13_attribution.py` builds `xs = [s for s in cfg["stabs"] if s.startswith("X")]`. Existing `results/attr_results.csv` has no verifier columns.
- Exp14: `experiment_14_rowcol_baseline.py` calls `verify_css_circuit` in `eval_mcts`, and now writes raw plus summary verifier CSVs. Existing `results/exp14_summary.csv` contains only a header, so current artifacts cannot prove legality.
- Exp15: `experiment_15_diagnose_ranking.py` constructs `init_m` from X stabilizers only. It is a ranking diagnostic, not a final-circuit verifier result.
- Exp16: `experiment_16_diagnose_distribution.py` constructs `init_m` from X stabilizers only. It is a distribution diagnostic, not a final-circuit verifier result.
- Exp17: `experiment_17_rowcol_rank.py` calls `verify_css_circuit` in `eval_mcts`, and now writes raw verifier CSV. Existing `results/exp17_summary.csv` contains only a header, so current artifacts cannot prove legality.
- Exp18: `experiment_18_interpretability.py` constructs `init_m` from X stabilizers only. It is an interpretability/ranking comparison, not a final-circuit verifier result.
- Exp19: `experiment_19_prediction_compression.py` constructs `init_m` from X stabilizers only. It is a prediction-compression diagnostic, not a final-circuit verifier result.
- Exp20: `experiment_20_loss_ablation.py` uses X stabilizers only for both dataset and MCTS matrix. Current `results/exp20_L0_mcts_raw.csv` has verifier columns and shows 20/20 `is_valid=True`, `x_syndrome_error=0.0`, `z_syndrome_error=0.0`, and `is_logical_zero=True`.

## Required Re-runs

For final validation, re-run the experiments that will be compared under the same search budget and the same verifier:

- Exp14 and Exp17, because current result CSVs are stale/empty despite verifier code now being present.
- Exp20 non-L0 arms, because existing L2 and L4h summaries do not include verifier fields.
- Any Exp11/12/13 baseline that is used as a comparison point, because existing artifacts do not prove final-circuit legality.
- Re-run with `FINAL_VALIDATION_LOG_DIR` set if first-valid iteration, first-best iteration, value inference time, simulation time, search time, node count, or per-100-iteration anytime CSVs are needed.
