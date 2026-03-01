# RLM Learning Testbed — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an 11-notebook educational testbed that teaches Recursive Language Models from scratch, with comparisons to RAG, ReAct, and DSPy.

**Architecture:** Progressive Jupyter notebooks that incrementally build a minimal RLM library (`rlm_core/`). Each notebook introduces one concept, imports from the shared library, and includes hands-on exercises. Sample datasets in `data/samples/` provide consistent test material across notebooks.

**Tech Stack:** Python 3.11+, vLLM, Qwen3-1.7B (4-bit AWQ), Jupyter Lab, graphviz, matplotlib, chromadb, sentence-transformers, dspy

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `rlm_core/__init__.py`
- Create: `.gitignore`
- Create: `notebooks/` (directory)
- Create: `data/samples/multihop_docs/` (directory)

**Step 1: Create directory structure**

```bash
mkdir -p notebooks data/samples/multihop_docs rlm_core tests
```

**Step 2: Create requirements.txt**

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
openai
ipywidgets
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
.env
*.egg-info/
dist/
build/
.venv/
```

**Step 4: Create rlm_core/__init__.py**

```python
"""Minimal Recursive Language Model library — built incrementally across notebooks."""
```

**Step 5: Create virtual environment and install dependencies**

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

Note: vLLM requires CUDA. If installation fails, the user may need to install PyTorch with CUDA first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Step 6: Verify Jupyter works**

```bash
jupyter lab --version
```
Expected: version number prints without error.

**Step 7: Commit**

```bash
git add requirements.txt .gitignore rlm_core/__init__.py
git commit -m "feat: project scaffolding with dependencies and directory structure"
```

---

## Task 2: Create Sample Datasets

**Files:**
- Create: `data/samples/needle_haystack.txt`
- Create: `data/samples/aggregation_items.json`
- Create: `data/samples/multihop_docs/doc_01.txt` through `doc_10.txt`

**Step 1: Create needle-in-haystack document**

Create `data/samples/needle_haystack.txt` — approximately 3 pages (~2000 words) of plausible filler text about various topics (history, science, geography), with one hidden sentence buried in the middle: "The secret code is BLUE-FALCON-42."

The filler text should be coherent paragraphs (not lorem ipsum) so the model needs to actually read through it. Topics should vary paragraph to paragraph to make it realistic.

**Step 2: Create aggregation dataset**

Create `data/samples/aggregation_items.json` — a JSON array of 100 objects, each with:
```json
{
  "id": 1,
  "name": "Widget Alpha",
  "color": "red",
  "size": "large",
  "category": "electronics",
  "price": 29.99,
  "in_stock": true
}
```

Distribute properties so that:
- ~20 items are red, ~15 are large, ~6 are both red AND large (answer to the sample query)
- Colors: red, blue, green, yellow, black (~20 each)
- Sizes: small, medium, large (~33 each)
- Categories: electronics, clothing, food, tools (~25 each)

**Step 3: Create multi-hop QA documents**

Create 10 files in `data/samples/multihop_docs/`. Each is a short text (~200 words) about a fictional company, person, or product. The facts should chain together so answering "Who works at the company that made product X?" requires reading doc_03 (product X was made by AcmeCorp) and doc_07 (Jane Smith works at AcmeCorp).

Chain structure:
- doc_01: "TechVista Inc was founded in 2019 by Maria Chen in Austin, Texas..."
- doc_02: "The NovaPad tablet was developed by TechVista Inc in 2022..."
- doc_03: "Dr. James Park is the lead engineer at TechVista Inc..."
- doc_04: "BrightPath Solutions specializes in renewable energy consulting..."
- doc_05: "The SolarFlow battery was designed by BrightPath Solutions..."
- doc_06: "Sarah Martinez joined BrightPath Solutions as CTO in 2021..."
- doc_07: "DataForge Analytics provides AI-powered business intelligence..."
- doc_08: "The InsightEngine platform was built by DataForge Analytics..."
- doc_09: "Raj Patel is a senior data scientist at DataForge Analytics..."
- doc_10: "The annual Tech Innovation Summit is organized by TechVista Inc..."

Sample queries:
- "Who is the lead engineer at the company that made the NovaPad?" → Dr. James Park
- "Who is the CTO of the company that designed the SolarFlow battery?" → Sarah Martinez
- "Which company was founded by Maria Chen?" → TechVista Inc

**Step 4: Commit**

```bash
git add data/
git commit -m "feat: add sample datasets for needle-in-haystack, aggregation, and multi-hop QA"
```

---

## Task 3: Build LLM Client (`rlm_core/llm_client.py`)

**Files:**
- Create: `rlm_core/llm_client.py`
- Create: `tests/test_llm_client.py`
- Modify: `rlm_core/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_llm_client.py`:
```python
"""Tests for the LLM client wrapper."""
import pytest
from unittest.mock import patch, MagicMock
from rlm_core.llm_client import LLMClient


def test_client_init_with_defaults():
    client = LLMClient(base_url="http://localhost:8000/v1")
    assert client.base_url == "http://localhost:8000/v1"
    assert client.model == "default"


def test_client_init_with_model():
    client = LLMClient(base_url="http://localhost:8000/v1", model="qwen3-1.7b")
    assert client.model == "qwen3-1.7b"


@patch("rlm_core.llm_client.OpenAI")
def test_completion_returns_text(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello world"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.chat.completions.create.return_value = mock_response

    client = LLMClient(base_url="http://localhost:8000/v1", model="test-model")
    result = client.completion("Say hello")

    assert result.text == "Hello world"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


@patch("rlm_core.llm_client.OpenAI")
def test_completion_with_system_prompt(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hi"))]
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 3
    mock_client.chat.completions.create.return_value = mock_response

    client = LLMClient(base_url="http://localhost:8000/v1", model="test-model")
    result = client.completion("Say hello", system="You are helpful.")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."


@patch("rlm_core.llm_client.OpenAI")
def test_completion_tracks_total_tokens(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Yes"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.chat.completions.create.return_value = mock_response

    client = LLMClient(base_url="http://localhost:8000/v1", model="test-model")
    client.completion("Test")
    client.completion("Test2")

    assert client.total_prompt_tokens == 20
    assert client.total_completion_tokens == 10
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm_core.llm_client'`

**Step 3: Write the implementation**

Create `rlm_core/llm_client.py`:
```python
"""Thin wrapper around OpenAI-compatible API for local vLLM models."""
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class CompletionResult:
    """Result from an LLM completion call."""
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    """Client for calling an OpenAI-compatible LLM API (e.g., vLLM)."""

    def __init__(self, base_url: str, model: str = "default", api_key: str = "not-needed"):
        self.base_url = base_url
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def completion(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Send a completion request and return the result."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

        result = CompletionResult(
            text=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens
        self.call_count += 1

        return result
```

**Step 4: Update `rlm_core/__init__.py`**

```python
"""Minimal Recursive Language Model library — built incrementally across notebooks."""
from rlm_core.llm_client import LLMClient, CompletionResult
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_llm_client.py -v
```
Expected: All 5 tests PASS.

**Step 6: Commit**

```bash
git add rlm_core/ tests/
git commit -m "feat: add LLM client wrapper with token tracking"
```

---

## Task 4: Build Sandbox (`rlm_core/sandbox.py`)

**Files:**
- Create: `rlm_core/sandbox.py`
- Create: `tests/test_sandbox.py`
- Modify: `rlm_core/__init__.py`

**Step 1: Write the failing tests**

Create `tests/test_sandbox.py`:
```python
"""Tests for the Python code execution sandbox."""
import pytest
from rlm_core.sandbox import Sandbox


def test_sandbox_executes_simple_code():
    sb = Sandbox()
    output = sb.execute("print('hello')")
    assert output.stdout == "hello\n"
    assert output.error is None


def test_sandbox_captures_print_output():
    sb = Sandbox()
    output = sb.execute("for i in range(3): print(i)")
    assert output.stdout == "0\n1\n2\n"


def test_sandbox_persists_variables():
    sb = Sandbox()
    sb.execute("x = 42")
    output = sb.execute("print(x)")
    assert output.stdout == "42\n"


def test_sandbox_inject_variable():
    sb = Sandbox(variables={"context": "The sky is blue."})
    output = sb.execute("print(len(context))")
    assert output.stdout == "17\n"


def test_sandbox_catches_errors():
    sb = Sandbox()
    output = sb.execute("1/0")
    assert output.error is not None
    assert "ZeroDivisionError" in output.error


def test_sandbox_catches_syntax_errors():
    sb = Sandbox()
    output = sb.execute("def foo(")
    assert output.error is not None
    assert "SyntaxError" in output.error


def test_sandbox_inject_function():
    def my_func(x):
        return x * 2

    sb = Sandbox(variables={"double": my_func})
    output = sb.execute("print(double(21))")
    assert output.stdout == "42\n"


def test_sandbox_blocks_dangerous_builtins():
    sb = Sandbox()
    output = sb.execute("import os; os.system('echo hacked')")
    # Should either error or the import should be restricted
    # We allow imports but block os.system via restricted exec
    # For educational purposes, we restrict __import__ for os/sys/subprocess
    assert output.error is not None


def test_sandbox_get_variable():
    sb = Sandbox()
    sb.execute("result = 100")
    assert sb.get_variable("result") == 100


def test_sandbox_get_variable_missing():
    sb = Sandbox()
    assert sb.get_variable("missing") is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_sandbox.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm_core.sandbox'`

**Step 3: Write the implementation**

Create `rlm_core/sandbox.py`:
```python
"""Safe Python code execution sandbox for RLM REPL."""
import io
import contextlib
import traceback
from dataclasses import dataclass, field


BLOCKED_MODULES = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "http"}


@dataclass
class ExecutionResult:
    """Result from executing code in the sandbox."""
    stdout: str
    error: str | None = None


def _restricted_import(name, *args, **kwargs):
    """Import hook that blocks dangerous modules."""
    if name in BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)


class Sandbox:
    """A Python code execution sandbox with variable injection and output capture."""

    def __init__(self, variables: dict | None = None):
        self._globals = {"__builtins__": {"__import__": _restricted_import}}
        # Add safe builtins
        safe_builtins = [
            "print", "len", "range", "int", "float", "str", "bool", "list",
            "dict", "tuple", "set", "enumerate", "zip", "map", "filter",
            "sorted", "reversed", "min", "max", "sum", "abs", "round",
            "isinstance", "type", "hasattr", "getattr", "setattr",
            "True", "False", "None", "Exception", "ValueError", "TypeError",
            "KeyError", "IndexError", "RuntimeError", "StopIteration",
        ]
        import builtins
        for name in safe_builtins:
            if hasattr(builtins, name):
                self._globals["__builtins__"][name] = getattr(builtins, name)

        if variables:
            self._globals.update(variables)

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and return captured output."""
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, self._globals)
            return ExecutionResult(stdout=stdout_capture.getvalue())
        except Exception:
            return ExecutionResult(
                stdout=stdout_capture.getvalue(),
                error=traceback.format_exc(),
            )

    def get_variable(self, name: str):
        """Retrieve a variable from the sandbox namespace."""
        return self._globals.get(name)
```

**Step 4: Update `rlm_core/__init__.py`**

```python
"""Minimal Recursive Language Model library — built incrementally across notebooks."""
from rlm_core.llm_client import LLMClient, CompletionResult
from rlm_core.sandbox import Sandbox, ExecutionResult
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_sandbox.py -v
```
Expected: All 10 tests PASS.

**Step 6: Commit**

```bash
git add rlm_core/sandbox.py tests/test_sandbox.py rlm_core/__init__.py
git commit -m "feat: add Python sandbox with variable injection and safety restrictions"
```

---

## Task 5: Build RLM Core (`rlm_core/rlm.py`)

**Files:**
- Create: `rlm_core/rlm.py`
- Create: `tests/test_rlm.py`
- Modify: `rlm_core/__init__.py`

**Step 1: Write the failing tests**

Create `tests/test_rlm.py`:
```python
"""Tests for the core RLM recursive engine."""
import pytest
from unittest.mock import MagicMock, patch
from rlm_core.rlm import RLMEngine, RLMResult, RecursionNode


def make_mock_client(responses):
    """Create a mock LLM client that returns responses in sequence."""
    client = MagicMock()
    call_count = 0

    def mock_completion(prompt, **kwargs):
        nonlocal call_count
        result = MagicMock()
        result.text = responses[min(call_count, len(responses) - 1)]
        result.prompt_tokens = 10
        result.completion_tokens = 5
        call_count += 1
        return result

    client.completion = mock_completion
    return client


def test_rlm_engine_init():
    client = MagicMock()
    engine = RLMEngine(client=client, max_depth=5)
    assert engine.max_depth == 5


def test_rlm_direct_final_answer():
    """Model immediately returns FINAL() without writing code."""
    client = make_mock_client([
        'FINAL("The answer is 42")'
    ])
    engine = RLMEngine(client=client)
    result = engine.run("What is the answer?", context="The answer is 42.")

    assert result.answer == "The answer is 42"
    assert result.root_node is not None


def test_rlm_code_execution():
    """Model writes code that computes the answer."""
    client = make_mock_client([
        'lines = context.split("\\n")\ncount = len(lines)\nFINAL(f"{count} lines")',
    ])
    engine = RLMEngine(client=client)
    result = engine.run("How many lines?", context="line1\nline2\nline3")

    assert "3" in result.answer


def test_rlm_recursion_tree_tracking():
    """Verify the recursion tree is properly tracked."""
    client = make_mock_client([
        'FINAL("direct answer")'
    ])
    engine = RLMEngine(client=client)
    result = engine.run("Test", context="test context")

    assert isinstance(result.root_node, RecursionNode)
    assert result.root_node.depth == 0
    assert result.root_node.query == "Test"


def test_rlm_max_depth_enforced():
    """Engine should stop recursion at max_depth."""
    # Make client always try to recurse
    client = make_mock_client([
        'result = llm_query("go deeper")\nFINAL(result)',
    ] * 10)
    engine = RLMEngine(client=client, max_depth=2)
    result = engine.run("Test", context="test")

    # Should complete without infinite recursion
    assert result.answer is not None


def test_rlm_result_has_stats():
    client = make_mock_client(['FINAL("done")'])
    engine = RLMEngine(client=client)
    result = engine.run("Test", context="test")

    assert result.total_llm_calls >= 1
    assert result.max_depth_reached >= 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_rlm.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm_core.rlm'`

**Step 3: Write the implementation**

Create `rlm_core/rlm.py`:
```python
"""Core Recursive Language Model engine."""
import re
from dataclasses import dataclass, field
from rlm_core.sandbox import Sandbox
from rlm_core.llm_client import LLMClient


@dataclass
class RecursionNode:
    """A node in the recursion tree, tracking one LLM call."""
    query: str
    depth: int
    response: str = ""
    code_executed: str = ""
    result: str = ""
    children: list["RecursionNode"] = field(default_factory=list)
    error: str | None = None


@dataclass
class RLMResult:
    """Final result from an RLM execution."""
    answer: str
    root_node: RecursionNode
    total_llm_calls: int
    max_depth_reached: int


SYSTEM_PROMPT = """You are an RLM (Recursive Language Model). You have access to a Python REPL environment.

The user's context is stored in the variable `context` — do NOT ask for it, just use it in your code.
You can call `llm_query(question)` to recursively ask a sub-question to another LLM instance.
The sub-LLM will also have access to the same `context` variable.

When you have the final answer, output it as: FINAL("your answer here")
Or if your answer is stored in a variable: FINAL_VAR(variable_name)

Write Python code to examine the context, decompose the problem if needed, and find the answer.
Do NOT wrap your code in markdown code blocks. Just write raw Python."""


class RLMEngine:
    """The recursive LLM engine — orchestrates sandbox + LLM calls."""

    def __init__(self, client: LLMClient, max_depth: int = 5, max_iterations: int = 10):
        self.client = client
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self._call_count = 0
        self._max_depth_seen = 0

    def run(self, query: str, context: str) -> RLMResult:
        """Run an RLM query with the given context."""
        self._call_count = 0
        self._max_depth_seen = 0
        root_node = RecursionNode(query=query, depth=0)
        answer = self._execute_node(root_node, context)
        return RLMResult(
            answer=answer,
            root_node=root_node,
            total_llm_calls=self._call_count,
            max_depth_reached=self._max_depth_seen,
        )

    def _execute_node(self, node: RecursionNode, context: str) -> str:
        """Execute a single node in the recursion tree."""
        if node.depth > self.max_depth:
            node.error = f"Max depth {self.max_depth} exceeded"
            return f"[ERROR: max recursion depth {self.max_depth} reached]"

        self._max_depth_seen = max(self._max_depth_seen, node.depth)

        def llm_query(sub_question: str) -> str:
            """Callable injected into sandbox for recursive sub-calls."""
            child_node = RecursionNode(query=sub_question, depth=node.depth + 1)
            node.children.append(child_node)
            return self._execute_node(child_node, context)

        sandbox = Sandbox(variables={
            "context": context,
            "llm_query": llm_query,
        })

        prompt = f"Query: {node.query}\n\nThe context is available as the variable `context` (length: {len(context)} chars)."
        response = self.client.completion(prompt, system=SYSTEM_PROMPT)
        self._call_count += 1
        node.response = response.text

        # Check for direct FINAL() in the response (no code execution needed)
        final_match = re.search(r'FINAL\("([^"]*)"\)', response.text)
        if final_match and response.text.strip().startswith("FINAL("):
            node.result = final_match.group(1)
            return final_match.group(1)

        # Execute the response as code
        code = response.text
        node.code_executed = code
        exec_result = sandbox.execute(code)

        if exec_result.error:
            node.error = exec_result.error
            # Try once more with error feedback
            retry_prompt = f"Your code had an error:\n{exec_result.error}\n\nFix and try again. Query: {node.query}"
            retry_response = self.client.completion(retry_prompt, system=SYSTEM_PROMPT)
            self._call_count += 1
            retry_exec = sandbox.execute(retry_response.text)
            if retry_exec.error:
                return f"[ERROR: {retry_exec.error}]"

        # Check for FINAL() or FINAL_VAR() in executed code output or namespace
        # Check stdout for FINAL
        output = exec_result.stdout if not exec_result.error else (retry_exec.stdout if 'retry_exec' in dir() else "")

        # Check sandbox namespace for FINAL_VAR markers
        final_var = sandbox.get_variable("FINAL_ANSWER")
        if final_var is not None:
            node.result = str(final_var)
            return str(final_var)

        # Parse FINAL() from the code itself (it was executed as a function call)
        # We need to inject FINAL and FINAL_VAR as functions
        # Re-run with FINAL injected
        final_result = {"value": None}

        def final_fn(answer):
            final_result["value"] = str(answer)

        def final_var_fn(var_name):
            val = sandbox.get_variable(var_name)
            final_result["value"] = str(val) if val is not None else f"[variable '{var_name}' not found]"

        sandbox._globals["FINAL"] = final_fn
        sandbox._globals["FINAL_VAR"] = final_var_fn

        # Re-execute with FINAL available
        re_exec = sandbox.execute(code)
        if final_result["value"] is not None:
            node.result = final_result["value"]
            return final_result["value"]

        # Fallback: return stdout
        node.result = output.strip() if output else response.text
        return node.result
```

Note: The FINAL/FINAL_VAR injection logic above has a subtlety — on the first execution FINAL isn't defined, so code containing `FINAL(...)` will error. The re-execution approach handles this. A cleaner implementation (done in notebook 5) injects FINAL from the start.

**Step 4: Update `rlm_core/__init__.py`**

```python
"""Minimal Recursive Language Model library — built incrementally across notebooks."""
from rlm_core.llm_client import LLMClient, CompletionResult
from rlm_core.sandbox import Sandbox, ExecutionResult
from rlm_core.rlm import RLMEngine, RLMResult, RecursionNode
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_rlm.py -v
```
Expected: All 6 tests PASS.

**Step 6: Commit**

```bash
git add rlm_core/rlm.py tests/test_rlm.py rlm_core/__init__.py
git commit -m "feat: add core RLM recursive engine with recursion tree tracking"
```

---

## Task 6: Build Visualizer (`rlm_core/visualizer.py`)

**Files:**
- Create: `rlm_core/visualizer.py`
- Create: `tests/test_visualizer.py`
- Modify: `rlm_core/__init__.py`

**Step 1: Write the failing tests**

Create `tests/test_visualizer.py`:
```python
"""Tests for the recursion tree visualizer."""
import pytest
from rlm_core.rlm import RecursionNode
from rlm_core.visualizer import tree_to_text, tree_to_dict


def make_sample_tree():
    root = RecursionNode(query="Main question", depth=0, result="Final answer")
    child1 = RecursionNode(query="Sub-question 1", depth=1, result="Partial 1")
    child2 = RecursionNode(query="Sub-question 2", depth=1, result="Partial 2")
    grandchild = RecursionNode(query="Sub-sub-question", depth=2, result="Detail")
    child1.children.append(grandchild)
    root.children = [child1, child2]
    return root


def test_tree_to_text_single_node():
    node = RecursionNode(query="Simple question", depth=0, result="42")
    text = tree_to_text(node)
    assert "Simple question" in text
    assert "42" in text


def test_tree_to_text_nested():
    tree = make_sample_tree()
    text = tree_to_text(tree)
    assert "Main question" in text
    assert "Sub-question 1" in text
    assert "Sub-sub-question" in text
    # Indentation should increase with depth
    lines = text.split("\n")
    assert any("  " in line and "Sub-question" in line for line in lines)


def test_tree_to_dict():
    tree = make_sample_tree()
    d = tree_to_dict(tree)
    assert d["query"] == "Main question"
    assert len(d["children"]) == 2
    assert d["children"][0]["children"][0]["query"] == "Sub-sub-question"


def test_tree_to_dict_single_node():
    node = RecursionNode(query="Q", depth=0, result="A")
    d = tree_to_dict(node)
    assert d["query"] == "Q"
    assert d["result"] == "A"
    assert d["children"] == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_visualizer.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm_core.visualizer'`

**Step 3: Write the implementation**

Create `rlm_core/visualizer.py`:
```python
"""Visualize RLM recursion trees as text, dicts, and graphviz diagrams."""
from rlm_core.rlm import RecursionNode


def tree_to_text(node: RecursionNode, indent: int = 0) -> str:
    """Render a recursion tree as indented text."""
    prefix = "  " * indent
    marker = "+-" if indent > 0 else ""
    lines = [f"{prefix}{marker}[D{node.depth}] Q: {node.query}"]
    if node.result:
        lines.append(f"{prefix}  -> {node.result[:100]}")
    if node.error:
        lines.append(f"{prefix}  !! ERROR: {node.error[:100]}")
    for child in node.children:
        lines.append(tree_to_text(child, indent + 1))
    return "\n".join(lines)


def tree_to_dict(node: RecursionNode) -> dict:
    """Convert a recursion tree to a nested dictionary (for JSON serialization)."""
    return {
        "query": node.query,
        "depth": node.depth,
        "result": node.result,
        "code_executed": node.code_executed,
        "error": node.error,
        "children": [tree_to_dict(c) for c in node.children],
    }


def tree_to_graphviz(node: RecursionNode) -> str:
    """Generate a Graphviz DOT string for the recursion tree."""
    lines = ["digraph RLM {", '  node [shape=box, style=filled, fillcolor=lightyellow];']
    _counter = [0]

    def _add_node(n: RecursionNode, parent_id: str | None = None):
        node_id = f"n{_counter[0]}"
        _counter[0] += 1
        label = f"D{n.depth}: {n.query[:40]}"
        if n.result:
            label += f"\\n-> {n.result[:30]}"
        color = "lightcoral" if n.error else ("lightgreen" if n.result else "lightyellow")
        lines.append(f'  {node_id} [label="{label}", fillcolor={color}];')
        if parent_id:
            lines.append(f"  {parent_id} -> {node_id};")
        for child in n.children:
            _add_node(child, node_id)

    _add_node(node)
    lines.append("}")
    return "\n".join(lines)
```

**Step 4: Update `rlm_core/__init__.py`**

```python
"""Minimal Recursive Language Model library — built incrementally across notebooks."""
from rlm_core.llm_client import LLMClient, CompletionResult
from rlm_core.sandbox import Sandbox, ExecutionResult
from rlm_core.rlm import RLMEngine, RLMResult, RecursionNode
from rlm_core.visualizer import tree_to_text, tree_to_dict, tree_to_graphviz
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_visualizer.py -v
```
Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
git add rlm_core/visualizer.py tests/test_visualizer.py rlm_core/__init__.py
git commit -m "feat: add recursion tree visualizer (text, dict, graphviz)"
```

---

## Task 7: Notebook 01 — LLM Basics

**Files:**
- Create: `notebooks/01_llm_basics.ipynb`

**Step 1: Create the notebook**

Create `notebooks/01_llm_basics.ipynb` with these cells in order:

**Cell 1 (markdown):**
```markdown
# Notebook 1: What is an LLM Call?

In this notebook, you'll learn:
- What a Large Language Model (LLM) does: text in → text out
- How to run a model locally using **vLLM**
- How to send prompts and receive completions
- Key parameters: temperature, max_tokens, stop sequences
- How to get structured output (JSON) from a model

## Prerequisites
Before running this notebook, you need:
1. A running vLLM server (instructions below)
2. The `rlm_core` library installed (from this project)
```

**Cell 2 (markdown):**
```markdown
## Step 1: Starting a Local LLM Server

vLLM lets you serve any Hugging Face model as an OpenAI-compatible API.

Open a **separate terminal** and run:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B-AWQ \
    --quantization awq \
    --max-model-len 4096 \
    --port 8000
```

This downloads and serves Qwen3-1.7B (4-bit quantized, ~1.5GB). Wait until you see "Uvicorn running on http://0.0.0.0:8000".

**What just happened?**
- vLLM loaded a **language model** — a neural network trained to predict the next word
- It's now serving it as an API on your local machine
- We can talk to it just like OpenAI's API, but everything runs on YOUR GPU
```

**Cell 3 (markdown):**
```markdown
## Step 2: Your First LLM Call

An LLM is fundamentally simple: you give it text, it gives you text back.

Let's use our `LLMClient` wrapper to make this easy:
```

**Cell 4 (code):**
```python
import sys
sys.path.insert(0, "..")  # So we can import rlm_core

from rlm_core import LLMClient

# Connect to our local vLLM server
client = LLMClient(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-1.7B-AWQ"
)

# Make our first call!
result = client.completion("What is 2 + 2?")
print("Model says:", result.text)
print(f"\nTokens used: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
```

**Cell 5 (markdown):**
```markdown
## Understanding What Happened

1. We sent the text `"What is 2 + 2?"` to the model
2. The model **tokenized** it — broke it into small pieces called tokens
3. For each token position, the model predicted the most likely **next token**
4. It kept generating tokens until it decided to stop
5. We got back text + token counts

**Tokens** are the currency of LLMs. Every call costs tokens. This matters when we build RLMs later — recursive calls multiply token usage.
```

**Cell 6 (markdown):**
```markdown
## Step 3: System Prompts — Giving the Model a Role

A **system prompt** tells the model WHO it should be. The user prompt tells it WHAT to do.
```

**Cell 7 (code):**
```python
# Without a system prompt — the model decides its own personality
result1 = client.completion("Explain gravity in one sentence.")
print("Default:", result1.text)

print("\n---\n")

# With a system prompt — we control the style
result2 = client.completion(
    "Explain gravity in one sentence.",
    system="You are a pirate who explains science. Use pirate language."
)
print("Pirate:", result2.text)
```

**Cell 8 (markdown):**
```markdown
## Step 4: Temperature — Controlling Randomness

**Temperature** controls how "creative" vs "deterministic" the model is:
- `temperature=0.0` → Always picks the most likely token (deterministic)
- `temperature=0.7` → Balanced (default)
- `temperature=1.5` → Very creative/random

Let's see the difference:
```

**Cell 9 (code):**
```python
prompt = "Write a one-sentence story about a cat."

print("=== Temperature 0.0 (deterministic) ===")
for i in range(3):
    r = client.completion(prompt, temperature=0.0, max_tokens=50)
    print(f"  Run {i+1}: {r.text.strip()}")

print("\n=== Temperature 1.2 (creative) ===")
for i in range(3):
    r = client.completion(prompt, temperature=1.2, max_tokens=50)
    print(f"  Run {i+1}: {r.text.strip()}")
```

**Cell 10 (markdown):**
```markdown
## Step 5: Stop Sequences — Controlling When the Model Stops

By default, the model generates until `max_tokens`. But you can tell it to **stop early** when it produces certain text. This is critical for RLMs — we use stop sequences to detect when the model has finished writing code.
```

**Cell 11 (code):**
```python
# Without stop sequence — model rambles
result = client.completion(
    "List 3 colors, one per line.",
    max_tokens=100,
    temperature=0.0
)
print("Without stop:", repr(result.text))

print("\n---\n")

# With stop sequence — stops after 3 items
result = client.completion(
    "List 3 colors, one per line. Start with '1.'",
    max_tokens=100,
    temperature=0.0,
    stop=["\n4."]  # Stop before a 4th item
)
print("With stop:", repr(result.text))
```

**Cell 12 (markdown):**
```markdown
## Step 6: Structured Output — Getting JSON from the Model

For RLMs, the model needs to output **code**, not prose. Let's practice getting structured output:
```

**Cell 13 (code):**
```python
result = client.completion(
    "List 3 programming languages with their year of creation. "
    "Respond ONLY with a JSON array, no other text. Example format: "
    '[{"name": "Python", "year": 1991}]',
    temperature=0.0
)

print("Raw response:", result.text)

import json
try:
    data = json.loads(result.text)
    print("\nParsed successfully!")
    for lang in data:
        print(f"  {lang['name']}: {lang['year']}")
except json.JSONDecodeError as e:
    print(f"\nFailed to parse JSON: {e}")
    print("This is normal — models don't always produce valid JSON!")
```

**Cell 14 (markdown):**
```markdown
## Step 7: Token Tracking — Measuring Cost

Every LLM call costs tokens. Our `LLMClient` tracks cumulative usage:
```

**Cell 15 (code):**
```python
print(f"Total calls so far: {client.call_count}")
print(f"Total prompt tokens: {client.total_prompt_tokens}")
print(f"Total completion tokens: {client.total_completion_tokens}")
print(f"Total tokens: {client.total_prompt_tokens + client.total_completion_tokens}")
```

**Cell 16 (markdown):**
```markdown
## Key Takeaways

1. **LLMs are text-to-text functions** — they take text input and produce text output
2. **System prompts** control the model's behavior and personality
3. **Temperature** controls randomness — low = deterministic, high = creative
4. **Stop sequences** let you control when generation stops
5. **Tokens** are the unit of cost — every call uses them
6. **Structured output** (JSON, code) is possible but not guaranteed

## What's Next?

In Notebook 2, we'll build a **sandbox** that can execute Python code generated by the LLM. This is the foundation of how RLMs work — the model doesn't just generate text, it generates **programs** that manipulate data.
```

**Step 2: Verify notebook opens**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_llm_basics.ipynb --ExecutePreprocessor.timeout=60 2>&1 || echo "Note: execution requires vLLM server running"
```

This will fail without a running vLLM server, which is expected. Just verify the notebook structure is valid.

**Step 3: Commit**

```bash
git add notebooks/01_llm_basics.ipynb
git commit -m "feat: add Notebook 01 — LLM Basics (vLLM setup, completions, tokens)"
```

---

## Task 8: Notebook 02 — REPL Sandbox

**Files:**
- Create: `notebooks/02_repl_sandbox.ipynb`

**Step 1: Create the notebook**

Create `notebooks/02_repl_sandbox.ipynb` with cells covering:

1. **Introduction** (markdown): Why RLMs need code execution — the model writes Python to manipulate context instead of trying to process everything in its prompt
2. **Basic exec()** (code): Show raw `exec("print(2+2)")` to demonstrate Python can run code from strings
3. **The problem with raw exec** (markdown): Security risks, no output capture, no error handling
4. **Building the Sandbox** (code): Import and demonstrate the `Sandbox` class from `rlm_core.sandbox`
5. **Output capture** (code): Show how `print()` calls inside executed code are captured
6. **Variable persistence** (code): Execute `x = 42` then `print(x)` in separate calls to show state persists
7. **Variable injection** (code): Create a sandbox with `variables={"context": "The sky is blue."}` and execute code that reads it
8. **Error handling** (code): Show what happens when the model writes bad code (syntax errors, runtime errors)
9. **Safety restrictions** (code): Show that `import os` is blocked
10. **Injecting functions** (code): Show how `llm_query` will be injected as a callable function
11. **Exercise: Build a mini REPL** (code): Interactive loop where user types code and sees output
12. **Key takeaways** (markdown): The sandbox is how RLMs safely run LLM-generated code, with variable injection providing the bridge between the model and external data

**Step 2: Commit**

```bash
git add notebooks/02_repl_sandbox.ipynb
git commit -m "feat: add Notebook 02 — REPL Sandbox (safe code execution)"
```

---

## Task 9: Notebook 03 — Recursive Calls

**Files:**
- Create: `notebooks/03_recursive_calls.ipynb`

**Step 1: Create the notebook**

Create `notebooks/03_recursive_calls.ipynb` with cells covering:

1. **Introduction** (markdown): The key RLM insight — an LLM can call itself as a function inside its own generated code
2. **Diagram** (markdown): ASCII diagram showing: User Query → Root LLM → generates code → code calls `llm_query()` → Sub-LLM runs → result flows back to sandbox → Root LLM continues
3. **Manual recursion demo** (code): Manually simulate the process — call LLM, take its output, feed to sandbox, call LLM again with results
4. **The llm_query function** (code): Build the `llm_query()` function that gets injected into the sandbox
5. **Wiring it together** (code): Create sandbox with both `context` and `llm_query` injected, show the LLM using it
6. **Recursion tree** (code): Use `RecursionNode` to track parent/child relationships as the model makes sub-calls
7. **Visualizing the tree** (code): Use `tree_to_text()` to print the recursion tree after execution
8. **FINAL() and FINAL_VAR()** (code): Show how the model signals it's done — `FINAL("answer")` for inline answers, `FINAL_VAR(variable_name)` for answers stored in variables
9. **Depth limits** (code): Show what happens without depth limits (infinite recursion) and how `max_depth` prevents it
10. **Full demo with RLMEngine** (code): Use the `RLMEngine` class to run a complete recursive query on the needle-in-haystack data
11. **Exercise** (code): Run RLMEngine on the aggregation dataset and visualize the recursion tree
12. **Key takeaways** (markdown): Recursion lets the model decompose problems it can't solve in one pass

**Step 2: Commit**

```bash
git add notebooks/03_recursive_calls.ipynb
git commit -m "feat: add Notebook 03 — Recursive Calls (llm_query, recursion trees)"
```

---

## Task 10: Notebook 04 — Emergent Strategies

**Files:**
- Create: `notebooks/04_emergent_strategies.ipynb`

**Step 1: Create the notebook**

Create `notebooks/04_emergent_strategies.ipynb` with cells covering:

1. **Introduction** (markdown): The 5 emergent strategies from the paper — nobody programs these, the model discovers them naturally
2. **Strategy 1: Peeking** (code+markdown): Model samples the beginning of context to understand structure. Demo with a long document.
3. **Strategy 2: Grepping** (code+markdown): Model uses regex/keyword search to find relevant sections. Demo with needle-in-haystack.
4. **Strategy 3: Partition + Map** (code+markdown): Model chunks context and runs sub-calls on each chunk. Demo with aggregation task.
5. **Strategy 4: Summarization** (code+markdown): Model extracts key info from context subsets. Demo with multi-hop QA.
6. **Strategy 5: Long-input Long-output** (code+markdown): Model builds answer programmatically in a variable. Demo with a transformation task.
7. **Strategy classifier** (code): Simple function that examines generated code and labels which strategy it uses (by looking for regex patterns, llm_query calls, chunk/split operations)
8. **Side-by-side comparison** (code): Run the same task, classify the strategy, show that different queries trigger different strategies
9. **Why context-as-variable enables this** (markdown): If context were stuffed in the prompt, the model couldn't grep, chunk, or selectively examine it
10. **Exercise** (code): Run multiple queries, classify strategies, build a strategy frequency chart with matplotlib
11. **Key takeaways** (markdown): The model's ability to write code that manipulates context enables emergent problem-solving strategies

**Step 2: Commit**

```bash
git add notebooks/04_emergent_strategies.ipynb
git commit -m "feat: add Notebook 04 — Emergent Strategies (peeking, grepping, partition+map)"
```

---

## Task 11: Notebook 05 — Full RLM Pipeline

**Files:**
- Create: `notebooks/05_full_rlm.ipynb`

**Step 1: Create the notebook**

Create `notebooks/05_full_rlm.ipynb` with cells covering:

1. **Introduction** (markdown): Assembling everything — this notebook brings together LLM client, sandbox, recursive calls, and strategies into one complete pipeline
2. **Architecture recap** (markdown): Visual diagram of the full RLM flow with all components labeled
3. **Complete RLM class** (code): Import and demonstrate the full `RLMEngine` from `rlm_core`
4. **Task 1: Needle-in-haystack** (code): Load `needle_haystack.txt`, run RLM, show answer + recursion tree + graphviz visualization
5. **Task 2: Aggregation** (code): Load `aggregation_items.json`, run RLM, compare answer with ground truth
6. **Task 3: Multi-hop QA** (code): Load `multihop_docs/`, run RLM, show how it connects facts across documents
7. **Graphviz visualization** (code): Render the recursion tree as a proper graph using `graphviz` library + `tree_to_graphviz()`
8. **Cost analysis** (code): Bar chart comparing token usage across the 3 tasks — prompt tokens, completion tokens, total calls
9. **Vanilla vs RLM comparison** (code): Run the same queries with direct prompting (stuff context in prompt) vs RLM, compare accuracy
10. **Exercise** (code): Create your own custom task (a document + question), run it through the RLM, visualize and analyze
11. **Summary of the core track** (markdown): What you've built: a complete, working RLM from scratch. What's next: foundations and comparisons

**Step 2: Commit**

```bash
git add notebooks/05_full_rlm.ipynb
git commit -m "feat: add Notebook 05 — Full RLM Pipeline (complete system, visualization)"
```

---

## Task 12: Notebook 06 — CoT → ToT → RLM

**Files:**
- Create: `notebooks/06_cot_tot_to_rlm.ipynb`

**Step 1: Create the notebook**

Create `notebooks/06_cot_tot_to_rlm.ipynb` with cells covering:

1. **Introduction** (markdown): The evolution of LLM reasoning — from linear chains to trees to recursive programs
2. **Chain-of-Thought (CoT)** (markdown+code): Explain CoT, implement it (add "Let's think step by step" to prompts), run on a reasoning task, show the step-by-step output
3. **Tree-of-Thought (ToT)** (markdown+code): Explain ToT, implement a simple version (generate multiple reasoning paths, evaluate each, pick best), run on the same task
4. **RLM as the next evolution** (markdown+code): Show how RLM generalizes — instead of the *user* structuring the reasoning (CoT) or *search* structuring it (ToT), the *model itself* decides how to decompose via code
5. **Side-by-side comparison** (code): Same task solved with CoT, ToT, and RLM — compare accuracy, token usage, and reasoning traces
6. **Visualization** (code): Show CoT as a line, ToT as a tree, RLM as a program-generated tree — matplotlib diagrams
7. **Key insight** (markdown): CoT constrains reasoning to a chain, ToT explores breadth, RLM gives the model full programmatic control over its reasoning structure
8. **Exercise** (code): Pick a hard reasoning problem, solve with all three approaches, compare

**Step 2: Commit**

```bash
git add notebooks/06_cot_tot_to_rlm.ipynb
git commit -m "feat: add Notebook 06 — CoT to ToT to RLM reasoning evolution"
```

---

## Task 13: Notebook 07 — Quantization & KV Cache

**Files:**
- Create: `notebooks/07_quantization_kv_cache.ipynb`

**Step 1: Create the notebook**

Create `notebooks/07_quantization_kv_cache.ipynb` with cells covering:

1. **Introduction** (markdown): Why do RLMs exist? Because long context is expensive. This notebook explains *why* and how quantization helps.
2. **Attention is O(n^2)** (markdown+code): Simple visualization showing how compute grows quadratically with context length. matplotlib plot of n vs n^2.
3. **What is the KV cache?** (markdown): Explain that during generation, the model stores key-value pairs for all previous tokens. This uses VRAM proportional to context length.
4. **Measuring inference time vs context length** (code): Send prompts of increasing length (100, 500, 1000, 2000, 4000 tokens) and measure time-to-first-token. Plot the curve.
5. **What is quantization?** (markdown): Explain FP16 → INT8 → INT4. Each level halves the model size but slightly reduces quality.
6. **Quantization formats** (markdown): GPTQ (GPU-optimized), AWQ (activation-aware), GGUF (CPU/GPU flexible). We're using AWQ.
7. **How RLMs sidestep the problem** (markdown+code): Instead of processing 100K tokens in one pass, RLM processes small chunks recursively. Show token budget comparison: vanilla (all tokens in prompt) vs RLM (small prompts × many calls).
8. **VRAM budget calculator** (code): Simple calculator — enter model size, quantization level, context length → estimated VRAM usage
9. **Exercise** (code): Plot the theoretical cost of vanilla vs RLM for increasing context lengths
10. **Key takeaways** (markdown): Quantization makes models fit on consumer GPUs. RLMs make long-context tractable by avoiding the quadratic attention cost.

**Step 2: Commit**

```bash
git add notebooks/07_quantization_kv_cache.ipynb
git commit -m "feat: add Notebook 07 — Quantization and KV Cache explained"
```

---

## Task 14: Notebook 08 — Function Calling & Tool Use

**Files:**
- Create: `notebooks/08_function_calling_tool_use.ipynb`

**Step 1: Create the notebook**

Create `notebooks/08_function_calling_tool_use.ipynb` with cells covering:

1. **Introduction** (markdown): LLMs can use tools — functions, APIs, databases. `llm_query()` in RLMs is a special case: the tool is *the model itself*.
2. **Basic function calling** (code): Implement a simple tool-using LLM — give it a `calculator(expression)` function, have it call the function by outputting structured JSON, parse and execute
3. **Multiple tools** (code): Add `search(query)` and `calculator(expr)` tools, let the model choose which to use
4. **The tool-use loop** (markdown+code): Implement the classic loop: LLM outputs tool call → we execute → feed result back → LLM continues. This is the ReAct pattern (preview for notebook 10).
5. **llm_query as self-tool-use** (markdown+code): Show that RLM's `llm_query()` follows the exact same pattern — but the "tool" is another LLM call. The model is using itself as a tool.
6. **Comparison** (code): Side-by-side — tool-calling LLM vs RLM solving the same problem. Show that RLM is more flexible because it can write arbitrary code, not just call predefined tools.
7. **Exercise** (code): Build a tool-using LLM with 3 tools, then show the same task solved by RLM without predefined tools
8. **Key takeaways** (markdown): Function calling is LLMs using external capabilities. RLMs generalize this by letting the model write code that can call any function, including itself.

**Step 2: Commit**

```bash
git add notebooks/08_function_calling_tool_use.ipynb
git commit -m "feat: add Notebook 08 — Function Calling and Tool Use patterns"
```

---

## Task 15: Notebook 09 — RAG Comparison

**Files:**
- Create: `notebooks/09_rag_comparison.ipynb`

**Step 1: Create the notebook**

Create `notebooks/09_rag_comparison.ipynb` with cells covering:

1. **Introduction** (markdown): RAG (Retrieval-Augmented Generation) is the dominant approach for LLMs + external data. How does it compare to RLMs?
2. **What is RAG?** (markdown): Diagram — Document → Chunk → Embed → Store → Query → Retrieve → Generate
3. **Build a simple RAG** (code): Using chromadb + sentence-transformers:
   - Load the multi-hop documents
   - Chunk them (each doc is already a chunk)
   - Embed with sentence-transformers
   - Store in chromadb
   - Retrieve top-k relevant chunks for a query
   - Generate answer with LLM using retrieved context
4. **RAG on needle-in-haystack** (code): Run RAG on the needle task — does retrieval find the needle?
5. **RAG on aggregation** (code): Run RAG on the aggregation task — retrieval can't easily handle "count all red items" because it only retrieves top-k
6. **RAG on multi-hop QA** (code): Run RAG on multi-hop — does it retrieve both needed documents?
7. **RLM on the same tasks** (code): Run RLM on all three tasks for comparison
8. **Results comparison** (code): Table/chart comparing RAG vs RLM on accuracy, token usage, and failure modes
9. **When RAG wins, when RLM wins** (markdown): RAG is fast and cheap for simple retrieval. RLM wins when tasks require aggregation, complex reasoning, or connecting distant pieces of information.
10. **Exercise** (code): Create a custom task where RAG fails but RLM succeeds (e.g., "count all items matching X across the dataset")
11. **Key takeaways** (markdown): RAG retrieves, RLM reasons. They solve different problems.

**Step 2: Commit**

```bash
git add notebooks/09_rag_comparison.ipynb
git commit -m "feat: add Notebook 09 — RAG vs RLM comparison"
```

---

## Task 16: Notebook 10 — ReAct Agent

**Files:**
- Create: `notebooks/10_react_agent.ipynb`

**Step 1: Create the notebook**

Create `notebooks/10_react_agent.ipynb` with cells covering:

1. **Introduction** (markdown): ReAct (Reasoning + Acting) — the dominant agent paradigm. How does RLM compare?
2. **What is ReAct?** (markdown): The loop: Thought → Action → Observation → Thought → ... → Final Answer
3. **Build a ReAct agent** (code): Implement a simple ReAct agent with:
   - `search(text, query)` — search within text
   - `read_section(text, start, end)` — read a portion of text
   - Parse the model's output for "Thought:", "Action:", and "Final Answer:"
   - Execute actions and feed observations back
4. **CodeAct variant** (code): Instead of predefined actions, the agent writes Python code (closer to RLM)
5. **ReAct on our tasks** (code): Run ReAct on needle-in-haystack, aggregation, multi-hop
6. **RLM on the same tasks** (code): Side-by-side comparison
7. **Key difference** (markdown): ReAct is step-by-step (one action at a time). RLM writes a complete program that may include loops, recursion, and multiple sub-calls planned together.
8. **Visualization** (code): Show ReAct trace (linear sequence) vs RLM trace (tree) for the same task
9. **Exercise** (code): Modify the ReAct agent to handle a task that requires 5+ steps, compare with RLM
10. **Key takeaways** (markdown): ReAct is sequential and human-readable. RLM is parallel and programmatic. RLM can plan multiple sub-calls in a single code block.

**Step 2: Commit**

```bash
git add notebooks/10_react_agent.ipynb
git commit -m "feat: add Notebook 10 — ReAct Agent comparison with RLM"
```

---

## Task 17: Notebook 11 — DSPy Connection

**Files:**
- Create: `notebooks/11_dspy_connection.ipynb`

**Step 1: Create the notebook**

Create `notebooks/11_dspy_connection.ipynb` with cells covering:

1. **Introduction** (markdown): DSPy — "programming, not prompting." Created by Omar Khattab (RLM co-author). Both DSPy and RLM share the philosophy of treating LLMs as programmable components.
2. **What is DSPy?** (markdown): Signatures (input/output specs), Modules (composable LLM programs), Optimizers (auto-tune prompts)
3. **Install and setup** (code): `import dspy`, configure with local vLLM model
4. **Simple DSPy module** (code): Build a `dspy.ChainOfThought` module for QA, run it on a sample question
5. **DSPy signatures** (code): Define custom signatures — show how DSPy structures LLM I/O
6. **DSPy vs RLM philosophy** (markdown): Both treat LLMs as functions. DSPy composes predefined modules. RLM lets the model write its own composition at runtime.
7. **Same task, both approaches** (code): Solve a multi-hop QA with DSPy (using `dspy.ChainOfThought` + retrieval) and RLM
8. **The spectrum** (markdown): Prompting → DSPy (structured modules) → RLM (self-programming). Each gives the model more autonomy.
9. **Exercise** (code): Build a DSPy pipeline for one of our tasks, compare with RLM
10. **Key takeaways** (markdown): DSPy and RLM are complementary — DSPy optimizes structured LLM programs, RLM lets models write their own programs at inference time.

**Step 2: Commit**

```bash
git add notebooks/11_dspy_connection.ipynb
git commit -m "feat: add Notebook 11 — DSPy Connection and philosophy comparison"
```

---

## Task 18: Final Integration & README

**Files:**
- Create: `README.md` (user requested documentation as part of the project)

**Step 1: Create README.md**

```markdown
# RLM Learning Testbed

An educational testbed for learning [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) from scratch.

Based on the paper by Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT OASYS Lab).

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows Git Bash
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start vLLM server** (in a separate terminal):
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3-1.7B-AWQ \
       --quantization awq \
       --max-model-len 4096 \
       --port 8000
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```

5. **Open notebooks in order**, starting with `notebooks/01_llm_basics.ipynb`.

## Notebook Guide

### Core Track — Build RLM From Scratch
| # | Notebook | What You Learn |
|---|----------|----------------|
| 1 | LLM Basics | vLLM setup, completions, tokens, temperature |
| 2 | REPL Sandbox | Safe code execution, variable injection |
| 3 | Recursive Calls | llm_query(), recursion trees, termination |
| 4 | Emergent Strategies | Peeking, grepping, partition+map |
| 5 | Full RLM | Complete pipeline, visualization, benchmarking |

### Foundations Track — Understand the Landscape
| # | Notebook | What You Learn |
|---|----------|----------------|
| 6 | CoT → ToT → RLM | Reasoning evolution: chain → tree → recursive |
| 7 | Quantization & KV Cache | Why long context is expensive, how quantization helps |
| 8 | Function Calling | Tool use patterns, how llm_query() fits in |

### Comparisons Track — RLM vs Alternatives
| # | Notebook | What You Learn |
|---|----------|----------------|
| 9 | RAG Comparison | Build RAG, compare with RLM on same task |
| 10 | ReAct Agent | Build an agent, compare reasoning with RLM |
| 11 | DSPy Connection | Programming-with-LLMs philosophy, DSPy basics |

## Hardware Requirements

- NVIDIA GPU with 4GB+ VRAM (8GB recommended)
- Qwen3-1.7B-AWQ uses ~1.5GB VRAM

## Running Tests

```bash
pytest tests/ -v
```
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```
Expected: All tests pass.

**Step 3: Commit**

```bash
git add README.md
git commit -m "feat: add README with setup instructions and notebook guide"
```

**Step 4: Final commit — push to remote**

```bash
git push -u origin master
```

---

## Execution Order Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Project scaffolding | — |
| 2 | Sample datasets | 1 |
| 3 | LLM client library | 1 |
| 4 | Sandbox library | 1 |
| 5 | RLM core engine | 3, 4 |
| 6 | Visualizer | 5 |
| 7 | Notebook 01 (LLM basics) | 3 |
| 8 | Notebook 02 (Sandbox) | 4 |
| 9 | Notebook 03 (Recursive calls) | 5, 6 |
| 10 | Notebook 04 (Strategies) | 9 |
| 11 | Notebook 05 (Full RLM) | 6, 2 |
| 12 | Notebook 06 (CoT/ToT) | 11 |
| 13 | Notebook 07 (Quantization) | 7 |
| 14 | Notebook 08 (Tool use) | 11 |
| 15 | Notebook 09 (RAG) | 11 |
| 16 | Notebook 10 (ReAct) | 11 |
| 17 | Notebook 11 (DSPy) | 11 |
| 18 | README & integration | All |

**Parallelizable tasks:** 2, 3, 4 can run in parallel. 7 and 8 can run in parallel. 12-17 can largely run in parallel.
