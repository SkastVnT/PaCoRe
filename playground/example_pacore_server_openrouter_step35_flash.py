"""Example PaCoRe server using OpenRouter as the upstream provider.

Purpose:
    Runs a PaCoRe-compatible server that routes requests to OpenRouter's API,
    specifically configured for models with reasoning capabilities (e.g., Step 3.5 Flash).

Server Usage:
    1. Set your API key:  export OPENROUTER_API_KEY='sk-or-...'
    2. Run the server:    python example_server_openrouter_step35_flash.py
    3. Send requests to:  http://localhost:8000/v1/chat/completions

Client Example for Step 3.5 Flash:
```python
import requests
import json
test_messages_0 = [
    {'role': 'user', 'content': "在三角形ABC中,\sin^2 A + \sin^2 B = \sin(A+B) 是 C 为直角的什么条件"},
]
response = requests.post(
    url="http://localhost:8000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "stepfun/step-3.5-flash:free",
        "messages": test_messages_0,
        "reasoning": {"enabled": True}
    })
    )
    # Extract the assistant message with reasoning_details
    response = response.json()
    print(test_messages_0)
    print(response)
```
"""

import os
from typing import Any
from loguru import logger

from pacore.server.base_exp import ChatCompletionRequest, Exp


class OpenRouterStep35FlashServer(Exp):
    # Upstream OpenRouter endpoint (OpenAI-compatible)
    upstream_api_base = "https://openrouter.ai/api/v1/chat/completions"

    # Optional: keep it small by default for demos; override via env if desired.
    num_responses_per_round = [4,]

    def get_upstream_extra_headers(self, request: ChatCompletionRequest) -> dict[str, str]:
        _ = request
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY. Export it first:\n"
                "  export OPENROUTER_API_KEY='...'\n"
                "Then re-run the server."
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        x_title = os.getenv("OPENROUTER_X_TITLE")
        if x_title:
            headers["X-Title"] = x_title
        logger.info(f"headers: {headers}")
        return headers

    def get_upstream_extra_body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        _ = request
        # OpenRouter extension: enable reasoning. If unsupported, upstream will ignore it.
        return {"reasoning": {"enabled": True}}


if __name__ == "__main__":
    OpenRouterStep35FlashServer().run()

