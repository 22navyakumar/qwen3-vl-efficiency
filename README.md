# Qwen3-VL Efficiency

This project studies inference efficiency for **Qwen3-VL-4B-Instruct**, a multimodal vision-language model.

The goal is to understand and improve:
- latency (speed)
- GPU memory usage
- throughput

We begin by running **baseline (no optimization)** experiments and later apply techniques like:
- visual token reduction
- KV cache compression
- modality-aware optimization

---

## 📌 Phase 1: Baseline Evaluation

We first evaluate the model **as-is** to understand how slow and memory-heavy it is.

### Benchmarks
- RealWorldQA (real-world visual reasoning)
- MMMU (multimodal reasoning)
- MathVista (visual math reasoning)
- DocVQA (document understanding)

Currently implemented:
- ✅ RealWorldQA baseline

---

## 🚀 Running RealWorldQA Baseline

This runs Qwen3-VL on a small subset of RealWorldQA without any optimization.

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

```


### 2. Run Baseline 
```bash
bash scripts/run_realworldqa.sh
```


---

## 🚀 Running MathVista Baseline

This runs Qwen3-VL on MathVista (math reasoning from images).

### Run baseline

```bash
bash scripts/run_mathvista.sh