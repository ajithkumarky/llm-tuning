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
    client = make_mock_client([
        'result = llm_query("go deeper")\nFINAL(result)',
    ] * 10)
    engine = RLMEngine(client=client, max_depth=2)
    result = engine.run("Test", context="test")

    assert result.answer is not None


def test_rlm_result_has_stats():
    client = make_mock_client(['FINAL("done")'])
    engine = RLMEngine(client=client)
    result = engine.run("Test", context="test")

    assert result.total_llm_calls >= 1
    assert result.max_depth_reached >= 0
