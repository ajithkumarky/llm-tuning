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
