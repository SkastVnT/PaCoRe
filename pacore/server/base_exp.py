"""Minimal PaCoRe server.

This module provides a small OpenAI-compatible `/v1/chat/completions` server that:
- Accepts standard `chat.completions` requests (`messages`, `model`, etc.).
- Calls an *upstream* OpenAI-compatible chat completions endpoint (e.g. vLLM).
- Runs a PaCoRe-style multi-round parallel sampling + synthesis loop.

OpenAI compatibility:
- Returns the final upstream response object as-is (single JSON response).

Research/debug extras:
- Adds `pacore_round_responses` containing raw upstream responses for each round.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import time
import uuid
from typing import Any, Optional

import jinja2
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from pacore.utils import async_chat_completion


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = None  # ignored; we return a single JSON response
    tools: Optional[list[dict[str, Any]]] = None

    class Config:
        extra = "allow"


class Exp:
    # Upstream OpenAI-compatible endpoint (e.g. vLLM, OpenRouter)
    upstream_api_base: str = os.getenv(
        "PACORE_UPSTREAM_API_BASE", "http://localhost:8000/v1/chat/completions"
    )
    host: str = os.getenv("PACORE_HOST", "0.0.0.0")
    port: int = int(os.getenv("PACORE_PORT", "8000"))

    # PaCoRe breadth schedule (final `[1]` is appended automatically)
    num_responses_per_round: list[int] = [4]

    # Upstream generation defaults (client can override per-request)
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0

    # Retry behavior (delegated to pacore.utils.async_chat_completion)
    timeout_seconds: float = float(os.getenv("PACORE_UPSTREAM_TIMEOUT_SECONDS", "7200"))
    retry_times: int = int(os.getenv("PACORE_UPSTREAM_RETRY_TIMES", "5"))
    upstream_stream: bool = os.getenv("PACORE_UPSTREAM_STREAM", "0") == "1"

    # Shuffle to avoid ordering bias
    random_seed: int = int(os.getenv("PACORE_RANDOM_SEED", "42"))

    user_template = """\
You are given a problem and a list of reference responses. Your job is to analyze these
references and provide your own response.

Original Problem:
{{ original_content }}

Reference Responses:
Note: Some references may contain <tool_call> tags indicating tool calls the reference intended to make. These tool calls have NOT been executed - they are shown only as reference for your analysis.
{% for response in ref_responses %}
Reference {{ loop.index }}:
{{ response }}
{% endfor %}

Now, based on the original problem and reference responses above, please provide your own
comprehensive solution.
"""

    tool_template = """\
You are given a tool response and a list of reference responses analyzing it. Your job is
to analyze these references and provide your own response.

Original Tool Response:
{{ original_content }}

Reference Responses:
Note: Some references may contain <tool_call> tags indicating tool calls the reference intended to make. These tool calls have NOT been executed - they are shown only as reference for your analysis.
{% for response in ref_responses %}
Reference {{ loop.index }}:
{{ response }}
{% endfor %}

Now, based on the original tool response and reference responses above, please provide your
own comprehensive analysis and next steps.
"""

    def _resolve_model(self, request: ChatCompletionRequest) -> str:
        if request.model and request.model != "default":
            return request.model
        raise ValueError("model is required!!")

    def get_upstream_extra_headers(self, request: ChatCompletionRequest) -> dict[str, str]:
        """Extra headers forwarded to the upstream endpoint.

        Subclasses can override this for providers that require auth headers.
        """
        _ = request
        return {}

    def get_upstream_extra_body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Extra JSON fields merged into the upstream request body.

        Subclasses can override this to enable provider-specific features.
        """
        _ = request
        return {}

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
        """Serialize tool_calls into a readable, non-standard XML-ish format.

        This is meant to be fed back as reference text in PaCoRe rounds. We do not
        execute tool calls in this minimal server.
        """
        parts: list[str] = []
        for tc in tool_calls:
            func = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(func, dict):
                func = tc if isinstance(tc, dict) else {}
            func_name = func.get("name", "unknown")
            args: Any = func.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            lines = ["<tool_call>", f"<function={func_name}>"]
            if isinstance(args, dict):
                for k, v in args.items():
                    val = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    lines.extend([f"<parameter={k}>", val, "</parameter>"])
            lines.extend(["</function>", "</tool_call>"])
            parts.append("\n".join(lines))
        return "\n".join(parts)

    @staticmethod
    def _extract_answer(upstream_response: dict[str, Any]) -> str:
        choice = (upstream_response.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []

        parts: list[str] = []
        if content:
            parts.append(content)
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            parts.append(Exp._serialize_tool_calls(tool_calls))
        return "\n".join(parts).strip()

    async def _call_upstream(
        self,
        request: ChatCompletionRequest,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await async_chat_completion(
            messages=messages,
            model=self._resolve_model(request),
            api_base=self.upstream_api_base,
            max_tokens=request.max_tokens or self.max_tokens,
            temperature=request.temperature or self.temperature,
            top_p=request.top_p or self.top_p,
            timeout_seconds=self.timeout_seconds,
            retry_times=self.retry_times,
            stream=self.upstream_stream,
            tools=request.tools,
            extra_headers=self.get_upstream_extra_headers(request),
            extra_body=self.get_upstream_extra_body(request),
        )

    async def handle_chat_completions(self, request: ChatCompletionRequest) -> dict[str, Any]:
        logger.debug(f"[chat_completions] request: {request}")
        t0 = time.time()
        rounds = self.num_responses_per_round + [1]
        request_id = uuid.uuid4().hex[:12]

        logger.info(
            f"[chat_completions] model={request.model!r} -> upstream_model={self._resolve_model(request)!r}, "
            f"rounds={rounds}, messages={len(request.messages)}"
        )

        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        last_msg = request.messages[-1]
        is_tool = last_msg.role == "tool"
        original_content = last_msg.content or ""

        all_rounds: list[list[dict[str, Any]]] = []
        for round_idx, num_calls in enumerate(rounds):
            if round_idx > 0:
                prev_answers = [self._extract_answer(r) for r in all_rounds[-1]]
                prev_answers = [a for a in prev_answers if a]

                template = jinja2.Template(self.tool_template if is_tool else self.user_template)
                prompt = template.render(
                    original_content=original_content,
                    ref_responses=prev_answers,
                )
                messages[-1] = {"role": last_msg.role, "content": prompt}

            async def _call_with_log(call_idx: int) -> dict[str, Any]:
                upstream_model = self._resolve_model(request)
                t_call = time.time()
                log_dict = {
                    "upstream_api_base": self.upstream_api_base,
                    "req_id": request_id,
                    "round": round_idx + 1,
                    "call": f"{call_idx}/{num_calls}",
                    "model": upstream_model,
                    "len(messages)": len(messages),
                    "temperature": request.temperature or self.temperature,
                    "top_p": request.top_p or self.top_p,
                    "max_tokens": request.max_tokens or self.max_tokens,
                    "tools": request.tools,
                    "stream": self.upstream_stream,
                }
                logger.info(
                    f"[upstream_call] " + " ".join([f"{k}={v}" for k, v in log_dict.items()]),
                )
                logger.debug(f"messages:\n{messages}")
                try:
                    resp = await self._call_upstream(request, messages.copy())
                except Exception as exc:
                    logger.warning(
                        "[upstream_call] "
                        f"req={request_id} round={round_idx + 1}/{len(rounds)} call={call_idx}/{num_calls} "
                        f"failed after {time.time() - t_call:.2f}s: {type(exc).__name__}: {exc}"
                    )
                    raise

                usage = resp.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                total_tokens = usage.get("total_tokens")
                if (
                    total_tokens is None
                    and isinstance(prompt_tokens, int)
                    and isinstance(completion_tokens, int)
                ):
                    total_tokens = prompt_tokens + completion_tokens
                logger.info(
                    "[upstream_call] "
                    f"req={request_id} round={round_idx + 1}/{len(rounds)} call={call_idx}/{num_calls} "
                    f"done in {time.time() - t_call:.2f}s "
                    f"tokens(prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens})"
                )
                return resp

            tasks = [asyncio.create_task(_call_with_log(i + 1)) for i in range(num_calls)]
            responses: list[dict[str, Any]] = []
            for coro in asyncio.as_completed(tasks):
                responses.append(await coro)

            responses.sort(key=lambda r: len(self._extract_answer(r)))
            random.Random(self.random_seed).shuffle(responses)
            all_rounds.append(responses)

        # Copy the final dict before attaching `pacore_round_responses` to avoid
        # introducing a self-referential object graph (FastAPI JSON encoding recursion).
        final = copy.deepcopy(all_rounds[-1][0]) if all_rounds[-1] else {"choices": []}
        final["pacore_round_responses"] = all_rounds

        dt = time.time() - t0
        logger.info(f"[chat_completions] done in {dt:.2f}s")
        return final

    def run(self):
        app = FastAPI(title="PaCoRe Server")

        # For demos / local usage. For production, consider restricting origins.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        def health():
            return Response(content="ok", status_code=200)

        @app.get("/")
        def root():
            return {
                "message": "PaCoRe Server",
                "upstream_api_base": self.upstream_api_base,
                "num_responses_per_round": self.num_responses_per_round,
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self.handle_chat_completions(request)

        logger.info(f"Starting PaCoRe Server on port {self.port}")
        logger.info(f"Upstream: {self.upstream_api_base}")
        uvicorn.run(app, host=self.host, port=self.port)


if __name__ == "__main__":
    Exp().run()
