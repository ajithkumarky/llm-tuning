# RLM Learning Testbed — Design Document

**Date:** 2026-02-28
**Goal:** Build an educational testbed to learn Recursive Language Models (RLMs) from scratch, with comparison notebooks for related techniques.
**Paper:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Alex L. Zhang, Tim Kraska, Omar Khattab (MIT OASYS Lab)
**Hardware:** NVIDIA RTX 5070 Laptop (8GB VRAM)

## Overview

A series of 11 Jupyter notebooks that progressively build a minimal RLM implementation from scratch, then compare it with RAG, ReAct agents, and DSPy. Additional notebooks cover foundational concepts (Chain-of-Thought, quantization, tool use).

## Project Structure

```
research/
├── notebooks/
│   ├── 01_llm_basics.ipynb              # Core Track
│   ├── 02_repl_sandbox.ipynb
│   ├── 03_recursive_calls.ipynb
│   ├── 04_emergent_strategies.ipynb
│   ├── 05_full_rlm.ipynb
│   ├── 06_cot_tot_to_rlm.ipynb          # Foundations Track
│   ├── 07_quantization_kv_cache.ipynb
│   ├── 08_function_calling_tool_use.ipynb
│   ├── 09_rag_comparison.ipynb           # Comparisons Track
│   ├── 10_react_agent.ipynb
│   └── 11_dspy_connection.ipynb
├── rlm_core/
│   ├── __init__.py
│   ├── llm_client.py                    # vLLM client wrapper
│   ├── sandbox.py                       # Safe code execution
│   ├── rlm.py                           # Core recursive logic
│   └── visualizer.py                    # Recursion tree visualization
├── data/samples/
│   ├── needle_haystack.txt
│   ├── aggregation_items.json
│   └── multihop_docs/
├── requirements.txt
└── docs/plans/
```

## Notebook Details

### Core Track — Build RLM From Scratch

**Notebook 1: LLM Basics** (`01_llm_basics.ipynb`)
- Set up vLLM to serve Qwen3-1.7B (4-bit AWQ quantized)
- Make completion calls with system/user prompts
- Understand tokens, temperature, stop sequences
- Get structured output (JSON) from the model
- Exercises: first prompt, temperature experiments, JSON parsing

**Notebook 2: REPL Sandbox** (`02_repl_sandbox.ipynb`)
- Why RLMs need code execution
- Build a `Sandbox` class with safe `exec()` and restricted globals
- Capture `print()` output from executed code
- Inject variables (like `context`) into the sandbox namespace
- Error handling for bad code
- Exercises: build Sandbox class, test with code snippets, inject context variable

**Notebook 3: Recursive Self-Calls** (`03_recursive_calls.ipynb`)
- The key insight: LLM can call itself as a function
- Implement `llm_query()` injected into the sandbox
- Recursion flow: root LLM → code → `llm_query()` → sub-LLM → result → sandbox
- Track the recursion tree (parent/child relationships)
- Termination with `FINAL()` and `FINAL_VAR()`
- Exercises: wire llm_query, run decomposition task, visualize call tree

**Notebook 4: Emergent Strategies** (`04_emergent_strategies.ipynb`)
- The 5 strategies: peeking, grepping, partition+map, summarization, long-I/O
- Recognize each strategy in generated code
- Why context-as-variable enables these strategies
- Strategy classification and annotation
- Exercises: trigger different strategies, classify them, compare with prompt-stuffing

**Notebook 5: Full RLM Pipeline** (`05_full_rlm.ipynb`)
- Assemble all components into a complete `RLM` class
- Full execution loop: prompt → REPL → code → sub-calls → aggregation → answer
- Cost/depth analysis: tokens and calls per task
- Interactive recursion tree visualization (graphviz/networkx)
- Comparison with vanilla prompting
- Exercises: needle-in-haystack, aggregation task, visualize traces, measure token usage

### Foundations Track — Understand the Landscape

**Notebook 6: CoT → ToT → RLM** (`06_cot_tot_to_rlm.ipynb`)
- Chain-of-Thought: step-by-step reasoning
- Tree-of-Thought: branching reasoning paths
- RLM as the evolution: recursive decomposition over context
- Show the same task solved with each technique
- Exercises: implement simple CoT, implement simple ToT, compare with RLM

**Notebook 7: Quantization & KV Cache** (`07_quantization_kv_cache.ipynb`)
- Why long context is expensive (attention is O(n^2))
- KV cache: what it is, how it grows with context length
- Quantization: GPTQ, AWQ, GGUF — reducing model size
- How RLMs sidestep the long-context problem entirely
- Exercises: measure inference time vs. context length, compare quantization methods

**Notebook 8: Function Calling & Tool Use** (`08_function_calling_tool_use.ipynb`)
- LLM tool-use patterns: function calling, structured output
- How `llm_query()` is a form of self-tool-use
- Building a simple tool-using LLM
- Exercises: implement function calling, compare with RLM's approach

### Comparisons Track — RLM vs Alternatives

**Notebook 9: RAG Comparison** (`09_rag_comparison.ipynb`)
- Build simple RAG: chunk documents, embed, retrieve, generate
- Use chromadb + sentence-transformers (local, no API)
- Run the same task with RAG and RLM side by side
- Analyze: what does RAG miss that RLM catches?
- Exercises: build RAG pipeline, compare accuracy, analyze failure modes

**Notebook 10: ReAct Agent** (`10_react_agent.ipynb`)
- Build a simple ReAct agent: Thought → Action → Observation loop
- Implement CodeAct variant (actions are code)
- Run the same task with ReAct and RLM
- Compare: agent decides actions step-by-step vs. RLM writes decomposition code
- Exercises: build ReAct, add tools, compare reasoning traces

**Notebook 11: DSPy Connection** (`11_dspy_connection.ipynb`)
- DSPy overview: programming with LLMs, not prompting
- Signatures, modules, and optimizers
- Philosophical connection: DSPy and RLM both treat LLMs as programmable components
- Exercises: build a simple DSPy module, compare with RLM approach

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM serving | vLLM | Local, fast, supports quantized models |
| Model | Qwen3-1.7B (4-bit AWQ) | Fits 8GB VRAM, good instruction following |
| Notebooks | Jupyter Lab | Rich output, inline visualization |
| Visualization | graphviz + matplotlib | Recursion trees and diagrams |
| RAG (NB 9) | chromadb + sentence-transformers | Local, no API keys |
| DSPy (NB 11) | dspy | Official library |
| Package mgmt | pip + venv | Simplest for beginners |
| Python | 3.11+ | Required by vLLM |

## Sample Tasks

Built into `data/samples/`:

1. **Mini Needle-in-Haystack** (`needle_haystack.txt`): ~3 pages of filler text with one hidden fact. Task: "What is the secret code?"
2. **Mini Aggregation** (`aggregation_items.json`): 100 JSON items with properties (color, size, category). Task: "How many items are red and large?"
3. **Mini Multi-hop QA** (`multihop_docs/`): 10 short text files, answer requires connecting facts across 2-3 documents. Task: "Who works at the company that made product X?"

## Dependencies (requirements.txt)

```
vllm
jupyter
jupyterlab
graphviz
matplotlib
networkx
chromadb
sentence-transformers
dspy
openai  # for vLLM OpenAI-compatible API
```
