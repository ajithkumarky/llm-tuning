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
