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
    assert output.stdout == "16\n"


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
    assert output.error is not None


def test_sandbox_get_variable():
    sb = Sandbox()
    sb.execute("result = 100")
    assert sb.get_variable("result") == 100


def test_sandbox_get_variable_missing():
    sb = Sandbox()
    assert sb.get_variable("missing") is None
