# RLM Learning Testbed

An educational testbed for learning [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) from scratch.

Based on the paper by Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT OASYS Lab).

## What Are RLMs?

Recursive Language Models let an LLM **call itself** through code. Instead of stuffing a massive context into one prompt, the model writes a program that:
- Examines pieces of the context
- Makes recursive sub-calls to itself (`llm_query()`)
- Aggregates results programmatically

This enables emergent strategies like peeking, grepping, partition+map, summarization, and long-I/O — all discovered by the model, not programmed by humans.

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or: source venv/bin/activate  # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start vLLM server (in a separate terminal)

**On Linux / WSL2:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B-AWQ \
    --quantization awq \
    --max-model-len 4096 \
    --port 8000
```

> **Note:** vLLM requires Linux. On Windows, use WSL2 to run the server. The Jupyter notebooks on Windows connect to `http://localhost:8000` which WSL2 forwards automatically.

### 4. Launch Jupyter

```bash
jupyter lab
```

### 5. Open notebooks in order, starting with `notebooks/01_llm_basics.ipynb`

All notebooks include a `SimulatedClient` for demos without a running vLLM server, plus instructions for switching to the real model.

## Notebook Guide

### Core Track — Build RLM From Scratch

| # | Notebook | What You Learn |
|---|----------|----------------|
| 1 | LLM Basics | vLLM setup, completions, tokens, temperature |
| 2 | REPL Sandbox | Safe code execution, variable injection |
| 3 | Recursive Calls | `llm_query()`, recursion trees, termination |
| 4 | Emergent Strategies | Peeking, grepping, partition+map |
| 5 | Full RLM | Complete pipeline, visualization, benchmarking |

### Foundations Track — Understand the Landscape

| # | Notebook | What You Learn |
|---|----------|----------------|
| 6 | CoT → ToT → RLM | Reasoning evolution: chain → tree → recursive |
| 7 | Quantization & KV Cache | Why long context is expensive, how quantization helps |
| 8 | Function Calling | Tool use patterns, how `llm_query()` fits in |

### Comparisons Track — RLM vs Alternatives

| # | Notebook | What You Learn |
|---|----------|----------------|
| 9 | RAG Comparison | Build RAG, compare with RLM, RAGAS evaluation |
| 10 | ReAct Agent | Build ReAct + CodeAct agents, compare with RLM |
| 11 | DSPy Connection | Programming-with-LLMs philosophy, DSPy basics |

## Project Structure

```
research/
├── notebooks/           # 11 Jupyter notebooks (start here)
├── rlm_core/            # Shared Python library
│   ├── llm_client.py    # vLLM client wrapper
│   ├── sandbox.py       # Safe code execution
│   ├── rlm.py           # Core recursive engine
│   └── visualizer.py    # Recursion tree visualization
├── tests/               # 25 unit tests
├── data/samples/        # Sample datasets
│   ├── needle_haystack.txt
│   ├── aggregation_items.json
│   └── multihop_docs/   # 10 documents for multi-hop QA
├── requirements.txt
└── docs/plans/          # Design and implementation documents
```

## Hardware Requirements

- NVIDIA GPU with 4GB+ VRAM (8GB recommended)
- Qwen3-1.7B-AWQ uses ~1.5GB VRAM
- Developed on NVIDIA RTX 5070 Laptop (8GB VRAM)

## Running Tests

```bash
pytest tests/ -v
```

All 25 tests run without a GPU or vLLM server — they use mock clients.

## Sample Tasks

Built into `data/samples/`:

1. **Needle-in-Haystack**: ~3 pages of text with one hidden fact. *"What is the secret code?"*
2. **Aggregation**: 100 JSON items. *"How many items are red and large?"* (Answer: 6)
3. **Multi-hop QA**: 10 documents requiring fact-chaining. *"Who is the lead engineer at the company that made the NovaPad?"*

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (2025)
- [DSPy](https://github.com/stanfordnlp/dspy) — Programming, not prompting
- [vLLM](https://github.com/vllm-project/vllm) — Fast LLM serving
