"""HTTP API: a subset of the OpenAI chat-completions format, so existing clients and SDKs can
point at the gateway by changing only their base URL and API key.

    POST /v1/chat/completions   model: "auto" | "fast" | "strong" | "local" | a model alias
    GET  /v1/models             the models this team may use
    GET  /v1/usage              this month's spend against the team's budget
    GET  /healthz

Every answer carries the routing decision in headers (x-gateway-*) and in a "gateway" field.
Streaming is not supported in this version.
"""

import time
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from . import __version__
from .config import TIERS, TeamPolicy
from .gateway import BadRequest, ChatRequest, Gateway, GatewayError


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


def create_app(gateway: Gateway) -> FastAPI:
    app = FastAPI(title="governed-llm-gateway", version=__version__)

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
    def chat_completions(body: ChatCompletionBody, caller: Annotated[TeamPolicy, Depends(team)]):
        if body.stream:
            raise BadRequest("Streaming is not supported by this gateway version.")
        result = gateway.chat(
            caller,
            ChatRequest(
                messages=[{"role": m.role, "content": m.text()} for m in body.messages],
                model=body.model,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            ),
        )
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

    @app.get("/v1/models")
    def models(caller: Annotated[TeamPolicy, Depends(team)]):
        allowed = [a for a in gateway.config.models if caller.may_use(a)]
        names = ["auto", *TIERS, *allowed]
        return {"object": "list", "data": [{"id": n, "object": "model"} for n in names]}

    @app.get("/v1/usage")
    def usage(caller: Annotated[TeamPolicy, Depends(team)]):
        return gateway.budget_status(caller)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    return app
