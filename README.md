# qwen3-vl-efficiency

This repository studies inference efficiency for Qwen3-VL-4B-Instruct.

## Phase 1: Vanilla Baselines
We first run the model without optimization on:
- MMMU
- MathVista
- DocVQA
- RealWorldQA

We will measure:
- latency
- throughput
- peak GPU memory
- average GPU memory
- task performance

## Repository Structure
- `src/` : shared model loading and utility code
- `eval/` : benchmark-specific evaluation scripts
- `scripts/` : runnable shell scripts
- `results/baseline/` : saved outputs and summaries