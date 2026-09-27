"""Model providers behind one small interface.

- AnthropicProvider: Claude, through the official SDK.
- OpenAICompatibleProvider: any server speaking the OpenAI chat-completions format. That covers
  Ollama and other local or self-hosted open-weight model servers (vLLM, LM Studio...), so data
  can stay inside the organisation.
- RecordingProvider: wraps any provider with a disk recorder, so evaluations can be replayed
  offline with identical answers and recorded latencies.

Messages use the OpenAI shape ({"role": "system" | "user" | "assistant", "content": str}); each
provider converts as needed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx


class ProviderError(RuntimeError):
    """The model call failed; the gateway may fall back to another model."""


class ReplayMiss(RuntimeError):
    """Offline replay has no recording for this call."""


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int = 0
    replayed: bool = False


class Provider(Protocol):
    def complete(
        self, model: str, messages: list[dict], max_tokens: int, temperature: float
    ) -> Completion: ...


class AnthropicProvider:
    def __init__(self, client=None):
        import anthropic

        self.client = client or anthropic.Anthropic()

    def complete(self, model, messages, max_tokens, temperature) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        # The current Messages API takes no sampling temperature, so it is not forwarded; the
        # gateway still treats temperature 0 as "a repeatable answer is wanted" (cacheable).
        kwargs = {"model": model, "messages": turns, "max_tokens": max_tokens}
        if system:
            kwargs["system"] = system
        started = time.perf_counter()
        try:
            response = self.client.messages.create(**kwargs)
        except Exception as exc:  # SDK errors: network, rate limit, overload, bad request
            raise ProviderError(f"anthropic: {exc}") from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class OpenAICompatibleProvider:
    def __init__(
        self, base_url: str, api_key: str | None = None, client: httpx.Client | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=120.0)

    def complete(self, model, messages, max_tokens, temperature) -> Completion:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self.headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"{self.base_url}: HTTP {response.status_code}")
        data = response.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"].get("content") or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class RecordingProvider:
    """Record every call to disk; in offline mode, replay only (and fail on a gap)."""

    def __init__(self, name: str, inner: Provider | None, root: Path, offline: bool = False):
        self.name = name
        self.inner = inner
        self.root = root
        self.offline = offline

    def complete(self, model, messages, max_tokens, temperature) -> Completion:
        request = [self.name, model, messages, max_tokens, temperature]
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
        path = self.root / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if "error" in record:  # failures are replayed too, so fallbacks behave the same
                raise ProviderError(record["error"])
            return Completion(**{**record, "replayed": True})
        if self.offline or self.inner is None:
            raise ReplayMiss(f"No recording for a call to {self.name}/{model}.")
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            completion = self.inner.complete(model, messages, max_tokens, temperature)
        except ProviderError as exc:
            path.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
            raise
        path.write_text(json.dumps(asdict(completion)), encoding="utf-8")
        return completion
