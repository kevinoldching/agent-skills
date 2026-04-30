# Skill Benchmark: llm-deploy-planner (Iteration 3)

**Model**: claude-opus-4-6
**Date**: 2026-03-22
**Evals**: 3 runs (with_skill vs without_skill)

## Summary

| Metric | With Skill | Without Skill | Delta |
|--------|------------|---------------|-------|
| Pass Rate | 100% | 100% | +0.00 |
| Avg Time | 148.6s | 98.5s | +50.1s |
| Avg Tokens | 35385 | 32653 | +2732 |

## Per-Eval Breakdown

| Eval | Config | Pass | Score | Time | Tokens |
|------|--------|------|-------|------|--------|
| eval-0 (Qwen-72B) | with_skill | ✓ | 5/5 | 245.9s | 35705 |
| eval-0 (Qwen-72B) | without_skill | ✓ | 5/5 | 110.5s | 37997 |
| eval-1 (Llama2-70B) | with_skill | ✓ | 5/5 | 96.0s | 33038 |
| eval-1 (Llama2-70B) | without_skill | ✓ | 5/5 | 101.0s | 28643 |
| eval-2 (Qwen-7B) | with_skill | ✓ | 4/4 | 103.8s | 37412 |
| eval-2 (Qwen-7B) | without_skill | ✓ | 4/4 | 83.9s | 31318 |

## Key Findings

### 1. All Assertions Passed
- Both configurations achieved 100% pass rate
- No regressions introduced by the skill

### 2. EP Annotation (Corrected)
| Eval | Model | Expected | with_skill | without_skill |
|------|-------|---------|------------|---------------|
| eval-0 | Qwen-72B (Dense) | EP=1 | EP=1 ✓ | EP=1 ✓ |
| eval-1 | Llama2-70B (Dense) | EP=1 | EP=1 ✓ | EP=1 ✓ |
| eval-2 | Qwen-7B (Dense) | EP=1 | EP=1 ✓ | EP=1 ✓ |

### 3. PD Strategy Decisions
| Eval | Scenario | Recommended | Correct? |
|------|----------|-------------|----------|
| eval-0 | Online inference, latency-sensitive | PD Separation | ✓ |
| eval-1 | High input variance (std=2000) | PD Separation | ✓ |
| eval-2 | Short I/O (512/128), batch focus | PD Mixed | ✓ |

### 4. Constraint: Total Cards Must Be Multiple of Single Machine Cards
| Eval | Total Cards | Single Machine | Valid? |
|------|------------|---------------|--------|
| eval-0 | 64 | 8 | 64 % 8 = 0 ✓ |
| eval-1 | 32 | 4 | 32 % 4 = 0 ✓ |
| eval-2 | 8 | 8 | 8 % 8 = 0 ✓ |

## Improvements in Iteration 3

1. **EP Annotation Fixed**: Dense models now correctly show EP=1 (vs previous incorrect EP>1)
2. **Total Cards Constraint**: Added validation that total_cards % single_machine_cards == 0
3. **llm_mem_estimator Verification**: Added validation of returned fields and fallback to manual estimation

## Observations

1. **with_skill runs took longer** (+50.1s avg) due to structured workflow and invoking llm_mem_estimator
2. **All constraints verified** in with_skill runs
3. **PD decisions align** with three-priority framework

## Analyst Notes

The skill successfully addresses all original issues:
- EP=1 correctly applied to dense models
- PD strategy decisions follow documented framework
- New constraints (total cards divisibility) properly verified
- llm_mem_estimator integration with verification and fallback
