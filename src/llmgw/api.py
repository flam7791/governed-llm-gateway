"""HTTP API: a subset of the OpenAI chat-completions format, so existing clients and SDKs can
point at the gateway by changing only their base URL and API key.

    POST /v1/chat/completions   model: "auto" | "fast" | "strong" | "local" | a model alias
    POST /v1/embeddings         model: "auto" (the configured default) | an embedding model alias
    GET  /v1/models             the models this team may use
    GET  /v1/usage              this month's spend against the team's budget
    GET  /metrics               Prometheus metrics (bearer token if LLMGW_METRICS_TOKEN is set)
    GET  /healthz

Every answer carries the routing decision in headers (x-gateway-*) and in a "gateway" field.
Streaming is not supported in this version.
"""

import hmac
import time
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from . import __version__, tracing
from .config import TIERS, TeamPolicy
from .gateway import AuthError, BadRequest, ChatRequest, EmbeddingRequest, Gateway, GatewayError
from .metrics import Metrics


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[dict]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")


class ChatCompletionBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "auto"
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    stream: bool = False


class EmbeddingsBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "auto"
    input: str | list[str]


def create_app(gateway: Gateway, metrics_token: str | None = None) -> FastAPI:
    app = FastAPI(title="governed-llm-gateway", version=__version__)
    metrics = Metrics(gateway)

    def team(authorization: Annotated[str | None, Header()] = None) -> TeamPolicy:
        key = None
        if authorization and authorization.lower().startswith("bearer "):
            key = authorization[7:].strip()
        return gateway.authenticate(key)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError):
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"message": str(exc), "type": exc.code, "code": exc.code}},
        )

    @app.post("/v1/chat/completions")
    def chat_completions(
        body: ChatCompletionBody, caller: Annotated[TeamPolicy, Depends(team)], request: Request
    ):
        if body.stream:
            raise BadRequest("Streaming is not supported by this gateway version.")
        with tracing.incoming(request.headers):
            result = _chat(body, caller)
        return _chat_response(body, result)

    def _chat(body: ChatCompletionBody, caller: TeamPolicy):
        return gateway.chat(
            caller,
            ChatRequest(
                messages=[{"role": m.role, "content": m.text()} for m in body.messages],
                model=body.model,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            ),
        )

    def _chat_response(body: ChatCompletionBody, result):
        content = {
            "id": f"chatcmpl-{result.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
            "gateway": {
                "route_reason": result.route_reason,
                "cost_usd": result.cost_usd,
                "cache_hit": result.cache_hit,
                "pii_values_masked": result.pii_masked,
                "fallbacks_tried": result.fallbacks_tried,
            },
        }
        headers = {
            "x-gateway-model": result.model,
            "x-gateway-cost-usd": f"{result.cost_usd:.6f}",
            "x-gateway-cache": "hit" if result.cache_hit else "miss",
            "x-gateway-pii-masked": str(result.pii_masked),
        }
        return JSONResponse(content=content, headers=headers)

    @app.post("/v1/embeddings")
    def embeddings(
        body: EmbeddingsBody, caller: Annotated[TeamPolicy, Depends(team)], request: Request
    ):
        texts = [body.input] if isinstance(body.input, str) else body.input
        with tracing.incoming(request.headers):
            result = gateway.embed(caller, EmbeddingRequest(texts=texts, model=body.model))
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i, vector in enumerate(result.vectors)
                ],
                "model": result.model,
                "usage": {
                    "prompt_tokens": result.input_tokens,
                    "total_tokens": result.input_tokens,
                },
                "gateway": {
                    "route_reason": result.route_reason,
                    "cost_usd": result.cost_usd,
                    "pii_values_masked": result.pii_masked,
                },
            },
            headers={
                "x-gateway-model": result.model,
                "x-gateway-cost-usd": f"{result.cost_usd:.8f}",
            },
        )

    @app.get("/v1/models")
    def models(caller: Annotated[TeamPolicy, Depends(team)]):
        allowed = [a for a in gateway.config.models if caller.may_use(a)]
        names = list(dict.fromkeys(["auto", *TIERS, *allowed]))  # a tier and alias may share a name
        return {"object": "list", "data": [{"id": n, "object": "model"} for n in names]}

    @app.get("/v1/usage")
    def usage(caller: Annotated[TeamPolicy, Depends(team)]):
        return gateway.budget_status(caller)

    @app.get("/metrics")
    def prometheus_metrics(authorization: Annotated[str | None, Header()] = None):
        if metrics_token and not hmac.compare_digest(
            authorization or "", f"Bearer {metrics_token}"
        ):
            raise AuthError("Metrics need the monitoring token.")
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "version": __version__}

    return app
