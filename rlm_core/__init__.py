"""Minimal Recursive Language Model library — built incrementally across notebooks."""
from rlm_core.llm_client import LLMClient, CompletionResult, strip_thinking_tags
from rlm_core.sandbox import Sandbox, ExecutionResult
from rlm_core.rlm import RLMEngine, RLMResult, RecursionNode
from rlm_core.visualizer import tree_to_text, tree_to_dict, tree_to_graphviz
