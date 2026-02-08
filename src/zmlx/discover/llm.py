"""LLM backends for kernel candidate generation."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM backend."""

    candidates: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    tokens_used: int = 0
    model_id: str = ""


class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends."""

    @abc.abstractmethod
    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        n_candidates: int = 8,
        temperature: float = 0.8,
    ) -> LLMResponse:
        """Generate kernel candidate variants.

        Args:
            system_prompt: System prompt with Metal expertise context.
            user_prompt: User prompt with target info and current source.
            n_candidates: Requested number of variants.
            temperature: Sampling temperature.

        Returns:
            An :class:`LLMResponse` with parsed candidates.
        """


class ClaudeBackend(LLMBackend):
    """Claude API backend using the ``anthropic`` SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        n_candidates: int = 8,
        temperature: float = 0.8,
    ) -> LLMResponse:
        from .prompts import parse_llm_response

        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text
        tokens = getattr(message.usage, "input_tokens", 0) + getattr(
            message.usage, "output_tokens", 0
        )

        candidates = parse_llm_response(raw)
        return LLMResponse(
            candidates=candidates,
            raw_response=raw,
            tokens_used=tokens,
            model_id=self.model,
        )


class OpenAIBackend(LLMBackend):
    """OpenAI API backend using the ``openai`` SDK."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        n_candidates: int = 8,
        temperature: float = 0.8,
    ) -> LLMResponse:
        from .prompts import parse_llm_response

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0

        candidates = parse_llm_response(raw)
        return LLMResponse(
            candidates=candidates,
            raw_response=raw,
            tokens_used=tokens,
            model_id=self.model,
        )


class MockBackend(LLMBackend):
    """Mock backend for testing — returns trivial mutations of the parent."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self._responses = list(responses) if responses else []
        self._call_count = 0

    @staticmethod
    def _extract_parent_source(user_prompt: str) -> str | None:
        """Extract the parent kernel source from a ```metal code block in the prompt."""
        import re

        m = re.search(r"```metal\s*\n(.*?)```", user_prompt, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        n_candidates: int = 8,
        temperature: float = 0.8,
    ) -> LLMResponse:
        self._call_count += 1

        if self._responses:
            return self._responses.pop(0)

        # Extract parent source from the prompt to produce compilable mutations
        parent_source = self._extract_parent_source(user_prompt)

        candidates = []
        for i in range(n_candidates):
            if parent_source:
                # Produce a compilable mutation: add a comment to the parent source
                source = (
                    f"// Mock variant {self._call_count}_{i}\n"
                    f"{parent_source}"
                )
            else:
                source = (
                    f"// Mock variant {self._call_count}_{i}\n"
                    "uint idx = thread_position_in_grid.x;\n"
                    "out[idx] = (T)((float)inp[idx]);\n"
                )
            candidates.append({
                "source": source,
                "reasoning": f"Mock variant {i}: trivial mutation for testing",
                "threadgroup": (256, 1, 1),
            })

        return LLMResponse(
            candidates=candidates,
            raw_response=f"Mock response #{self._call_count}",
            tokens_used=0,
            model_id="mock",
        )


class ClaudeCodeBackend(LLMBackend):
    """Claude Code CLI backend — shells out to ``claude`` with --print mode.

    No API key needed; uses whatever auth the local ``claude`` CLI has.
    """

    def __init__(self, model: str = "sonnet"):
        self.model = model

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        n_candidates: int = 8,
        temperature: float = 0.8,
    ) -> LLMResponse:
        import subprocess

        from .prompts import parse_llm_response

        # Combine system + user into a single prompt for --print mode
        combined = f"{system_prompt}\n\n---\n\n{user_prompt}"

        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",  # print mode (no interactive)
                    "--model", self.model,
                    "--dangerously-skip-permissions",
                ],
                input=combined,
                capture_output=True,
                text=True,
                timeout=120,
            )
            raw = result.stdout
            if result.returncode != 0:
                raw = result.stderr or "claude CLI failed"
        except FileNotFoundError:
            raw = "Error: 'claude' CLI not found in PATH"
        except subprocess.TimeoutExpired:
            raw = "Error: claude CLI timed out after 120s"

        candidates = parse_llm_response(raw)
        return LLMResponse(
            candidates=candidates,
            raw_response=raw,
            tokens_used=0,
            model_id=f"claude-code:{self.model}",
        )


def get_backend(name: str, model: str | None = None) -> LLMBackend:
    """Factory function to get an LLM backend by name.

    Args:
        name: One of ``"claude"``, ``"openai"``, ``"mock"``.
        model: Optional model override.

    Returns:
        An :class:`LLMBackend` instance.
    """
    if name == "claude":
        return ClaudeBackend(model=model or "claude-sonnet-4-20250514")
    elif name == "claude-code":
        return ClaudeCodeBackend(model=model or "sonnet")
    elif name == "openai":
        return OpenAIBackend(model=model or "gpt-4o")
    elif name == "mock":
        return MockBackend()
    else:
        raise ValueError(
            f"Unknown LLM backend: {name!r}. "
            "Choose from: claude, claude-code, openai, mock"
        )
