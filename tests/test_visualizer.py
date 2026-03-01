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
