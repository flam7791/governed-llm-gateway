import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic
import httpx
import pytest

from llmgw.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    ProviderError,
    RecordingProvider,
    ReplayMiss,
)

from .conftest import FakeProvider


def test_openai_compatible_provider_speaks_the_chat_completions_format():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Lisbon"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            },
        )

    provider = OpenAICompatibleProvider(
        "http://localhost:11434/v1", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = provider.complete("llama3.1:8b", [{"role": "user", "content": "Capital?"}], 50, 0.0)
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["model"] == "llama3.1:8b" and seen["body"]["stream"] is False
    assert (result.text, result.input_tokens, result.output_tokens) == ("Lisbon", 12, 2)


def test_openai_compatible_errors_become_provider_errors():
    provider = OpenAICompatibleProvider(
        "http://localhost:11434/v1",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
    )
    with pytest.raises(ProviderError):
        provider.complete("m", [{"role": "user", "content": "x"}], 10, 0.0)


class FakeMessagesApi(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeMessagesApi.received.append(body)
        payload = json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "Lisbon"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 3},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_anthropic_provider_moves_system_messages_and_reads_usage():
    server = HTTPServer(("127.0.0.1", 0), FakeMessagesApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        sdk = anthropic.Anthropic(
            api_key="test", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
        )
        result = AnthropicProvider(client=sdk).complete(
            "claude-test",
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital?"}],
            50,
            0.0,
        )
    finally:
        server.shutdown()
    sent = FakeMessagesApi.received[-1]
    assert sent["system"] == "Be brief."
    assert sent["messages"] == [{"role": "user", "content": "Capital?"}]
    assert "temperature" not in sent  # not part of the current Messages API
    assert (result.text, result.input_tokens, result.output_tokens) == ("Lisbon", 20, 3)


def test_recording_provider_replays_offline(tmp_path):
    inner = FakeProvider("x")
    recorder = RecordingProvider("x", inner, tmp_path)
    args = ("m", [{"role": "user", "content": "q"}], 10, 0.0)
    first = recorder.complete(*args)
    replay = RecordingProvider("x", None, tmp_path, offline=True).complete(*args)
    assert replay.replayed and replay.text == first.text and replay.latency_ms == 42
    with pytest.raises(ReplayMiss):
        RecordingProvider("x", None, tmp_path, offline=True).complete("m", [], 10, 0.0)


def test_recorded_failures_replay_as_failures(tmp_path):
    args = ("m", [{"role": "user", "content": "q"}], 10, 0.0)
    with pytest.raises(ProviderError):
        RecordingProvider("x", FakeProvider("x", fail=True), tmp_path).complete(*args)
    with pytest.raises(ProviderError):  # replayed offline, not a ReplayMiss
        RecordingProvider("x", None, tmp_path, offline=True).complete(*args)
