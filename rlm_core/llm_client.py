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
