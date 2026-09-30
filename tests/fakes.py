"""Scripted model replies for offline tests."""
from __future__ import annotations

import json

from conftest import CLEAN_CHAPTERS, CLEAN_CTA
from narration.llm import Reply
from narration.research import Fact, Research

FACTS = [
    Fact("Winds in the upper atmosphere reach about 400 miles per hour.",
         "confirmed", "NASA Juno mission overview"),
    Fact("Jupiter may have a diluted, fuzzy core rather than a compact one.",
         "disputed", "Juno gravity science papers"),
]


def script_reply(chapters: list[dict], cta: str = CLEAN_CTA, next_video: int = 0) -> dict:
    return {"main_idea": "Falling into Jupiter never ends on a floor.",
            "key_facts": ["winds of about 400 miles per hour"],
            "chapters": chapters, "final_cta": cta, "next_video": next_video}


def dirty_chapters() -> list[dict]:
    """The clean script with a repeated sentence and a banned filler phrase."""
    out = [dict(c) for c in CLEAN_CHAPTERS]
    out[1]["narration"] = ("This matters because the gas just gets thicker as you "
                           "drop. " + out[1]["narration"])
    out[2]["narration"] += " Jupiter has no surface to land on."
    return out


BASELINE_FRAMES = [
    "You are falling into the planet Jupiter right now.",
    "The clouds are thick and they are all around you.",
    "This matters because the pressure is enormous down there.",
    "The clouds are thick and they are all around you.",
    "The temperature reaches 3,000 degrees in the deep layers.",
    "The clouds are thick and they are all around you.",
    "The key point is that the pressure is enormous down there.",
    "Jupiter has a diluted, fuzzy core instead of a compact one.",
    "The clouds are thick and they are all around you.",
    "Tell us what you think in the comments below.",
]


def research_text(facts: list[Fact] = FACTS) -> str:
    return json.dumps({"facts": [
        {"claim": f.claim, "status": f.status, "source": f.source,
         "why_it_lands": "it can be felt"} for f in facts]})


class FakeLLM:
    """Scripted stand-in for LLMClient. Replies are queued per call name.

    A queued value may be a dict (JSON reply), a str (text reply) or a
    callable taking the user prompt.
    """
    model = "openai/gpt-5.6-luna-pro"

    def __init__(self, **queues: list) -> None:
        self.queues = {k: list(v) for k, v in queues.items()}
        self.calls: list[tuple[str, str]] = []

    def _next(self, name: str, user: str):
        self.calls.append((name, user))
        queue = self.queues.get(name)
        if not queue:
            raise AssertionError(f"unexpected call {name!r}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return item(user) if callable(item) else item

    def ask_json(self, system: str, user: str, schema: dict, *,
                 name: str = "result", max_tokens: int = 32000) -> Reply:
        data = self._next(name, user)
        return Reply(text=json.dumps(data), data=data, tokens_in=1000,
                     tokens_out=5000, usd=0.0062, seconds=0.0)

    def ask_text(self, system: str, user: str, *, model: str | None = None,
                 max_tokens: int = 4000) -> Reply:
        text = self._next("research", user)
        return Reply(text=text, tokens_in=180, tokens_out=600, usd=0.006,
                     citations=["https://example.org/a", "https://example.org/b"])

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


def make_research() -> Research:
    return Research(topic="What would happen if you fell into Jupiter?",
                    facts=list(FACTS))
