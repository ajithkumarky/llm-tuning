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

        # Container for FINAL result
        final_result = {"value": None}

        def final_fn(answer):
            final_result["value"] = str(answer)

        def final_var_fn(var_name):
            val = sandbox.get_variable(var_name)
            final_result["value"] = str(val) if val is not None else f"[variable '{var_name}' not found]"

        def llm_query(sub_question: str) -> str:
            """Callable injected into sandbox for recursive sub-calls."""
            child_node = RecursionNode(query=sub_question, depth=node.depth + 1)
            node.children.append(child_node)
            return self._execute_node(child_node, context)

        sandbox = Sandbox(variables={
            "context": context,
            "llm_query": llm_query,
            "FINAL": final_fn,
            "FINAL_VAR": final_var_fn,
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

        if final_result["value"] is not None:
            node.result = final_result["value"]
            return final_result["value"]

        if exec_result.error:
            node.error = exec_result.error
            # Try once more with error feedback
            retry_prompt = f"Your code had an error:\n{exec_result.error}\n\nFix and try again. Query: {node.query}"
            retry_response = self.client.completion(retry_prompt, system=SYSTEM_PROMPT)
            self._call_count += 1
            retry_exec = sandbox.execute(retry_response.text)

            if final_result["value"] is not None:
                node.result = final_result["value"]
                return final_result["value"]

            if retry_exec.error:
                return f"[ERROR: {retry_exec.error}]"
            output = retry_exec.stdout
        else:
            output = exec_result.stdout

        # Fallback: return stdout
        node.result = output.strip() if output else response.text
        return node.result
