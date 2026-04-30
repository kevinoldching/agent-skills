# Skill Benchmark: llm-deploy-planner

**Model**: claude-opus-4-6
**Date**: 2026-03-22
**Evals**: 3 runs (with_skill vs without_skill)

## Summary

| Metric | With Skill | Without Skill | Delta |
|--------|------------|---------------|-------|
| Pass Rate | 100% | 100% | +0.00 |
| Avg Time | 163.7s | 110.9s | +52.8s |
| Avg Tokens | 29921 | 29748 | +173 |

## Per-Eval Breakdown

| Eval | Config | Pass | Score | Time | Tokens |
|------|--------|------|-------|------|--------|
| eval-0 (Qwen-72B) | with_skill | ✓ | 5/5 | 177.0s | 31731 |
| eval-0 (Qwen-72B) | without_skill | ✓ | 5/5 | 148.7s | 33535 |
| eval-1 (Llama2-70B) | with_skill | ✓ | 5/5 | 161.4s | 30495 |
| eval-1 (Llama2-70B) | without_skill | ✓ | 5/5 | 90.3s | 31986 |
| eval-2 (Qwen-7B) | with_skill | ✓ | 4/4 | 152.8s | 27536 |
| eval-2 (Qwen-7B) | without_skill | ✓ | 4/4 | 93.8s | 23724 |

## Key Findings

### 1. All Assertions Passed
- Both with_skill and without_skill configurations achieved 100% pass rate
- The skill did not introduce regressions

### 2. EP Annotation Corrected (Key Fix)
| Eval | Model | Expected EP | with_skill | without_skill |
|------|-------|------------|------------|--------------|
| eval-0 | Qwen-72B (Dense) | EP=1 | EP=1 ✓ | EP=8 ✗ |
| eval-1 | Llama2-70B (Dense) | EP=1 | EP=1 ✓ | EP=8 ✗ |
| eval-2 | Qwen-7B (Dense) | EP=1 | EP=1 ✓ | N/A |

**Note**: without_skill baseline still shows incorrect EP=8 for dense models. The skill correctly fixes this issue.

### 3. PD Strategy Decisions
| Eval | Scenario | Recommended | Correct? |
|------|----------|-------------|----------|
| eval-0 | Online inference, latency-sensitive | PD Separation | ✓ |
| eval-1 | High input variance (std=2000) | PD Separation | ✓ |
| eval-2 | Short I/O (512/128), batch focus | PD Mixed | ✓ |

## Observations

1. **with_skill runs took longer** (+52.8s avg) - likely due to following the structured workflow and invoking llm_mem_estimator
2. **All critical fixes verified**:
   - Dense models correctly show EP=1
   - PD strategy decisions align with the three-priority framework
   - xPyD format correctly shown for separation cases
3. **No regressions** introduced by the skill

## Analyst Notes

The skill successfully addresses the original issues:
- EP annotation is now correct for dense models (EP=1 instead of EP>1)
- PD strategy decisions follow the documented three-priority framework
- Structured output format is consistently followed

The additional time in with_skill runs is expected given the structured workflow and memory estimation step.
