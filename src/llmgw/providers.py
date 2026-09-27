"""Model providers behind one small interface.

- AnthropicProvider: Claude, through the official SDK.
- OpenAICompatibleProvider: any server speaking the OpenAI chat-completions format. That covers
  Ollama and other local or self-hosted open-weight model servers (vLLM, LM Studio...), so data
  can stay inside the organisation. Also embeddings (/embeddings).
- AzureOpenAIProvider: Azure OpenAI in Microsoft Foundry through its v1 API
  (https://RESOURCE.openai.azure.com/openai/v1), authenticated with an API key or with a
  Microsoft Entra ID token (managed identity, no key to rotate). The model is the deployment name.
- RecordingProvider: wraps any provider with a disk recorder, so evaluations can be replayed
  offline with identical answers and recorded latencies.

Messages use the OpenAI shape ({"role": "system" | "user" | "assistant", "content": str}); each
provider converts as needed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
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


@dataclass
class Embeddings:
    vectors: list[list[float]]
    input_tokens: int
    latency_ms: int = 0


class Provider(Protocol):
    def complete(
        self, model: str, messages: list[dict], max_tokens: int, temperature: float
    ) -> Completion: ...


class EmbeddingProvider(Protocol):
    def embed(self, model: str, texts: list[str]) -> Embeddings: ...


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
        self,
        base_url: str,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=120.0)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _chat_payload(self, model, messages, max_tokens, temperature) -> dict:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self.client.post(
                f"{self.base_url}{path}", json=payload, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"{self.base_url}: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"{self.base_url}: the reply is not JSON") from exc

    def complete(self, model, messages, max_tokens, temperature) -> Completion:
        started = time.perf_counter()
        data = self._post(
            "/chat/completions", self._chat_payload(model, messages, max_tokens, temperature)
        )
        usage = data.get("usage") or {}
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"{self.base_url}: unexpected reply format") from exc
        return Completion(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def embed(self, model: str, texts: list[str]) -> Embeddings:
        started = time.perf_counter()
        data = self._post("/embeddings", {"model": model, "input": texts})
        try:
            rows = sorted(data.get("data") or [], key=lambda row: row.get("index", 0))
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"{self.base_url}: unexpected reply format") from exc
        if len(vectors) != len(texts):
            raise ProviderError(f"{self.base_url}: expected {len(texts)} vectors, got {len(rows)}")
        usage = data.get("usage") or {}
        return Embeddings(
            vectors=vectors,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Azure OpenAI v1 API. Pass an API key, or a token provider for Microsoft Entra ID.

    Differences from the plain OpenAI format handled here: the key goes in an "api-key" header;
    current models take max_completion_tokens; and the sampling temperature is not forwarded,
    because reasoning models reject it (as with Claude, temperature 0 only marks a request as
    cacheable in the gateway).
    """

    ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        token_provider: Callable[[], str] | None = None,
        client: httpx.Client | None = None,
    ):
        if not (api_key or token_provider):
            raise ValueError("AzureOpenAIProvider needs an API key or a token provider.")
        super().__init__(base_url, api_key=api_key, client=client)
        self.token_provider = token_provider

    def _headers(self) -> dict:
        if self.token_provider is not None:
            try:
                token = self.token_provider()
            except Exception as exc:  # no managed identity, expired login, network...
                first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                raise ProviderError(
                    f"could not get a Microsoft Entra ID token: {first_line}"
                ) from exc
            return {"Authorization": f"Bearer {token}"}
        return {"api-key": self.api_key}

    def _chat_payload(self, model, messages, max_tokens, temperature) -> dict:
        return {"model": model, "messages": messages, "max_completion_tokens": max_tokens}


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

    def embed(self, model: str, texts: list[str]) -> Embeddings:
        request = [self.name, "embed", model, texts]
        key = hashlib.sha256(json.dumps(request).encode("utf-8")).hexdigest()
        path = self.root / f"{key}.json"
        if path.exists():
            return Embeddings(**json.loads(path.read_text(encoding="utf-8")))
        if self.offline or self.inner is None:
            raise ReplayMiss(f"No recording for embeddings from {self.name}/{model}.")
        self.root.mkdir(parents=True, exist_ok=True)
        result = self.inner.embed(model, texts)
        path.write_text(json.dumps(asdict(result)), encoding="utf-8")
        return result
