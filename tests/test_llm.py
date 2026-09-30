from __future__ import annotations

import json

import httpx
import pytest

from narration.llm import (CALL_TOKENS, DEFAULT_MODEL, RESEARCH_USD_PER_QUERY, LLMClient,
                           LLMError, estimate_usd, price_usd)


def ok(content: str, *, tin: int = 100, tout: int = 50, cost: float | None = None,
       finish: str = "stop", **extra) -> dict:
    usage = {"prompt_tokens": tin, "completion_tokens": tout}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}],
            "usage": usage, **extra}


def client(responses: list, seen: list | None = None) -> LLMClient:
    """A client whose HTTP layer replays `responses` in order."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json=item)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return LLMClient(api_key="test", model=DEFAULT_MODEL, http=http, sleep=lambda s: None)


def test_the_key_comes_from_the_environment_only(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        LLMClient()
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert LLMClient().model == DEFAULT_MODEL


def test_json_call_sends_a_strict_schema_and_reasoning_effort():
    seen: list = []
    reply = client([ok('{"a": 1}')], seen).ask_json("sys", "user", {"type": "object"},
                                                    name="script")
    assert reply.data == {"a": 1}
    body = seen[0]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["name"] == "script"
    assert body["reasoning"] == {"effort": "high"}
    assert "temperature" not in body


def test_text_call_can_use_another_model_and_returns_citations():
    seen: list = []
    reply = client([ok("plain", citations=["https://a.example"])], seen).ask_text(
        "sys", "user", model="perplexity/sonar")
    assert seen[0]["model"] == "perplexity/sonar"
    assert "response_format" not in seen[0]
    assert reply.text == "plain" and reply.citations == ["https://a.example"]


def test_annotation_links_are_read_as_citations():
    msg = {"content": "x", "annotations": [{"url_citation": {"url": "https://b.example"}}]}
    payload = {"choices": [{"message": msg}], "usage": {}}
    assert client([payload]).ask_text("s", "u").citations == ["https://b.example"]


def test_a_200_with_a_non_json_body_is_retried_not_fatal():
    c = client([httpx.Response(200, text="<html>bad gateway</html>"), ok('{"ok": true}')])
    assert c.ask_json("s", "u", {}).data == {"ok": True}


def test_server_errors_and_rate_limits_are_retried():
    c = client([httpx.Response(502, text="bad"), httpx.Response(429, text="slow"),
                ok('{"ok": 1}')])
    assert c.ask_json("s", "u", {}).data == {"ok": 1}


def test_network_errors_are_retried():
    c = client([httpx.ConnectError("down"), ok('{"ok": 1}')])
    assert c.ask_json("s", "u", {}).data == {"ok": 1}


def test_a_bad_request_is_not_retried():
    seen: list = []
    c = client([httpx.Response(400, text="bad schema"), ok("{}")], seen)
    with pytest.raises(LLMError, match="HTTP 400"):
        c.ask_json("s", "u", {})
    assert len(seen) == 1


def test_gives_up_after_the_retries_and_names_no_fallback():
    c = client([httpx.Response(503, text="x")] * 3)
    with pytest.raises(LLMError, match="No fallback"):
        c.ask_json("s", "u", {})


def test_a_reply_cut_by_max_tokens_is_not_paid_for_three_times():
    seen: list = []
    c = client([ok('{"ok": tr', finish="length")] * 3, seen)
    with pytest.raises(LLMError, match="max_tokens"):
        c.ask_json("s", "u", {})
    assert len(seen) == 1


def test_an_empty_content_is_a_clear_error():
    with pytest.raises(LLMError, match="no text"):
        client([{"choices": [{"message": {"content": None}}]}]).ask_json("s", "u", {})


def test_a_provider_error_in_the_body_is_raised():
    with pytest.raises(LLMError, match="error"):
        client([{"error": {"message": "moderation"}}]).ask_json("s", "u", {})


def test_tokens_of_a_rejected_attempt_are_counted():
    reply = client([ok("not json"), ok('{"ok": true}')]).ask_json("s", "u", {})
    assert (reply.tokens_in, reply.tokens_out) == (200, 100)


def test_reported_cost_wins_over_the_price_table():
    reply = client([ok('{"a": 1}', tin=1_000_000, tout=0, cost=0.5)]).ask_json("s", "u", {})
    assert reply.usd == pytest.approx(0.5)


def test_cost_falls_back_to_the_price_table():
    reply = client([ok('{"a": 1}', tin=1_000_000, tout=1_000_000)]).ask_json("s", "u", {})
    assert reply.usd == pytest.approx(0.20 + 1.20)


def test_estimates():
    assert price_usd("unknown/model", 1, 1) is None
    assert estimate_usd("unknown/model", {"writing": 1}) is None
    (tin, tout), (hin, hout) = CALL_TOKENS["writing"]
    one = estimate_usd(DEFAULT_MODEL, {"writing": 1}, research_queries=1)
    assert one == pytest.approx(tin / 1e6 * 0.20 + tout / 1e6 * 1.20 + RESEARCH_USD_PER_QUERY)
    high = estimate_usd(DEFAULT_MODEL, {"writing": 1}, high=True)
    assert high == pytest.approx(hin / 1e6 * 0.20 + hout / 1e6 * 1.20)
    assert estimate_usd(DEFAULT_MODEL, {"baseline": 1}, scale=2) == pytest.approx(
        2 * estimate_usd(DEFAULT_MODEL, {"baseline": 1}), rel=1e-3)


def test_call_sizes_reproduce_the_first_eval_run():
    # 6 topics: 6 fact queries, 6 baseline calls and 11 writing calls billed
    # $0.975 in total; per call, writing cost $0.058-0.067 and baseline
    # $0.040-0.047.
    run = estimate_usd(DEFAULT_MODEL, {"writing": 11, "baseline": 6}, research_queries=6)
    assert run == pytest.approx(0.975, rel=0.03)
    assert 0.058 <= estimate_usd(DEFAULT_MODEL, {"writing": 1}) <= 0.067
    assert 0.040 <= estimate_usd(DEFAULT_MODEL, {"baseline": 1}) <= 0.047
    assert estimate_usd(DEFAULT_MODEL, {"writing": 1}, high=True) >= 0.067
