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
