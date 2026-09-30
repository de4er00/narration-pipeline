"""OpenRouter client: strict-JSON calls for writing, plain-text calls for search.

There is deliberately no fallback model. A refusal, a content filter or a
timeout stops the run with a clear error; silently switching to another model
would change what is being measured and billed.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import httpx

logger = logging.getLogger(__name__)

API = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.6-luna-pro"
RESEARCH_MODEL = "perplexity/sonar"

# USD per million tokens (input, output). Used for the estimate before a run
# and as a fallback when the response does not report its own cost. The
# default model has no temperature parameter; it takes a reasoning effort.
PRICES: dict[str, tuple[float, float]] = {
    DEFAULT_MODEL: (0.20, 1.20),
}

# Measured: one Sonar query with facts and citations. Search happens on the
# provider side, so the bill is for the summary, not for the pages read.
RESEARCH_USD_PER_QUERY = 0.006

# Tokens per call for the estimate. Output includes reasoning tokens, which
# dominate; the numbers are generous so the estimate errs high.
EXPECTED_TOKENS: dict[str, tuple[int, int]] = {
    "script": (4_500, 20_000),
    "repair": (7_500, 20_000),
    "visuals": (6_000, 30_000),
    "baseline": (3_000, 25_000),
}


class LLMError(RuntimeError):
    """A message that can be shown to the operator as is."""


@dataclass
class Reply:
    text: str
    data: dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    citations: list[str] = field(default_factory=list)


class ChatClient(Protocol):
    """What the pipeline needs from a model client. Tests pass a fake."""
    model: str

    def ask_json(self, system: str, user: str, schema: dict, *,
                 name: str = "result", max_tokens: int = 32000) -> Reply: ...

    def ask_text(self, system: str, user: str, *, model: str | None = None,
                 max_tokens: int = 4000) -> Reply: ...


def price_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def estimate_usd(model: str, calls: dict[str, float],
                 research_queries: int = 0) -> float | None:
    """Estimated spend for `calls` = {call kind: count}."""
    total = research_queries * RESEARCH_USD_PER_QUERY
    for kind, count in calls.items():
        tin, tout = EXPECTED_TOKENS[kind]
        usd = price_usd(model, tin, tout)
        if usd is None:
            return None
        total += usd * count
    return total


class _Retry(Exception):
    pass


class LLMClient:
    def __init__(self, api_key: str | None = None, *,
                 model: str = DEFAULT_MODEL, base_url: str = API,
                 timeout: float = 600.0, effort: str = "high",
                 http: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self._key = key
        self.model = model
        self._base = base_url.rstrip("/")
        self._effort = effort
        self._http = http or httpx.Client(timeout=timeout)
        self._sleep = sleep

    def ask_json(self, system: str, user: str, schema: dict, *,
                 name: str = "result", max_tokens: int = 32000,
                 retries: int = 2) -> Reply:
        """A call with a strict response schema.

        Structured outputs move shape validation to the provider, so there is
        no "find the JSON in the text" parsing on this path.
        """
        body = {
            "model": self.model,
            "messages": _messages(system, user),
            "max_tokens": max_tokens,
            "reasoning": {"effort": self._effort},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
            "usage": {"include": True},
        }
        return self._call(body, max_tokens, retries, want_json=True)

    def ask_text(self, system: str, user: str, *, model: str | None = None,
                 max_tokens: int = 4000, retries: int = 2) -> Reply:
        body = {
            "model": model or self.model,
            "messages": _messages(system, user),
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        return self._call(body, max_tokens, retries, want_json=False)

    def _call(self, body: dict, max_tokens: int, retries: int,
              want_json: bool) -> Reply:
        model = body["model"]
        last = ""
        # Tokens of every attempt: the provider bills a reply that failed to
        # parse just the same.
        spent = {"in": 0, "out": 0, "usd": 0.0}
        t0 = time.monotonic()
        for attempt in range(1, retries + 2):
            try:
                reply = self._attempt(body, max_tokens, want_json, spent)
            except _Retry as e:
                last = str(e)
                logger.warning("llm: %s, attempt %d", last, attempt)
                self._sleep(3.0 * attempt)
                continue
            reply.seconds = round(time.monotonic() - t0, 1)
            logger.info("llm: %s %.0fs, %d->%d tokens, $%.4f", model,
                        reply.seconds, reply.tokens_in, reply.tokens_out, reply.usd)
            return reply
        raise LLMError(f"{model} failed after {retries + 1} attempts: {last}. "
                       f"No fallback model is used.")

    def _attempt(self, body: dict, max_tokens: int, want_json: bool,
                 spent: dict) -> Reply:
        try:
            r = self._http.post(
                f"{self._base}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {self._key}",
                         "Content-Type": "application/json"})
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise _Retry(f"network: {e}") from e
        if r.status_code != 200:
            # Any 4xx but 429 is a bad request: a retry only burns money.
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise LLMError(f"OpenRouter refused: HTTP {r.status_code} "
                               f"{r.text[:300]}")
            raise _Retry(f"HTTP {r.status_code}: {r.text[:200]}")
        # A 200 is not a promise of a JSON body: gateways return HTML pages
        # with status 200, and losing a paid video to that is worse than a retry.
        try:
            payload = r.json()
        except ValueError as e:
            raise _Retry(f"response body is not JSON ({e}): {r.text[:200]}") from e
        if "error" in payload:
            raise LLMError(f"OpenRouter returned an error: {payload['error']}")

        usage = payload.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        spent["in"] += tin
        spent["out"] += tout
        cost = usage.get("cost")
        if cost is None:
            cost = price_usd(body["model"], tin, tout) or 0.0
        spent["usd"] += float(cost)

        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response: {str(payload)[:300]}") from e
        # A reply cut at max_tokens comes back cut again on retry, for the
        # same money.
        if choice.get("finish_reason") == "length":
            raise LLMError(f"reply truncated at max_tokens={max_tokens}; a retry "
                           f"would be truncated the same way")
        if content is None:
            raise LLMError(f"the model returned no text: {str(payload)[:300]}")

        data: dict = {}
        if want_json:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise _Retry(f"reply is not valid JSON ({e})") from e
        return Reply(text=content, data=data, tokens_in=spent["in"],
                     tokens_out=spent["out"], usd=round(spent["usd"], 5),
                     citations=_citations(payload, message))


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _citations(payload: dict, message: dict) -> list[str]:
    """Sonar returns links in `citations`; web-plugin models in annotations."""
    out = list(payload.get("citations") or [])
    out += [(a.get("url_citation") or {}).get("url", "")
            for a in (message.get("annotations") or [])]
    return [u for u in out if u]
