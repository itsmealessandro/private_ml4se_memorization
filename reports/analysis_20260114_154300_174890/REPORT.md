# LLM Memorization Probe Report

## Run Info
- Timestamp (UTC): 2026-01-14T15:43:54.118345Z
- Model: Qwen/Qwen2-0.5B-Instruct
- Dataset: Nan-Do/code-search-net-python / split=train
- Samples (n): 5
- Prompt template: prompts/code_gen_prompt.txt
- Docstring perturbation: True
- Seed: 42
- Max new tokens: 128

## Executive Summary (Human-readable)
- Verdict: LITTLE_EVIDENCE_MEMORIZATION
- One-line rationale: Similarity is close to background (margin=-0.371) and exact-match rate is low (0.0%).

## Key Metrics
- Avg MinHash similarity (original): 0.0691
- Exact match rate (original): 0.00%  (exact_matches=0/5)
- High similarity rate @0.7: 0.00%
- Background similarity (reference baseline): 0.4402
- Background similarity (generated baseline): 0.4422
- Similarity margin vs background (avg_original - background_ref): -0.3710
- Avg MinHash similarity (perturbed): 0.0543
- Avg robustness drop (original - perturbed): 0.0148

## Decision Criteria (What ‘verdict’ means)
We label the run as:
- STRONG_EVIDENCE_MEMORIZATION if:
  - exact_match_rate >= 0.05
  OR
  - (avg_original_minhash - background_ref_minhash) >= 0.20 AND high_similarity_rate@0.7 >= 0.20 AND avg_robustness_drop >= 0.10
- LITTLE_EVIDENCE_MEMORIZATION if all of:
  - (avg_original_minhash - background_ref_minhash) < 0.10
  - exact_match_rate < 0.01
  - avg_robustness_drop < 0.05
- otherwise INCONCLUSIVE.

## Interpretation Notes
- Exact match suggests verbatim reproduction (subject to code-extraction correctness).
- MinHash close to 1 indicates strong textual overlap; compare against background baselines.
- A large robustness drop after docstring perturbation suggests trigger sensitivity (memorization-like) vs robust generalization.

## Files Produced
- results.json: results.json (machine-readable)
- REPORT.md: this document
