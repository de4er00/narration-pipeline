"""Stage 0: facts for the topic, each with a status, from a search model.

Same video, same writing model: 0 concrete numbers without facts on input, 15
with them. The status sets how firmly the script may state a fact, and that is
checked in code. Sonar searches on the provider side, so one query cost $0.006
against $0.337 for the writing model with a web plugin; it has no structured
outputs, so the JSON is parsed from text.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cards import ChannelCard
from .llm import RESEARCH_MODEL, ChatClient, LLMError

logger = logging.getLogger(__name__)

# Six, not eight: eight facts once gave 32 numbers in 823 words. A fact is
# there to hold the story up, not to be the story.
DEFAULT_FACTS = 6

STATUSES = ("confirmed", "reported", "disputed", "speculation")

STATUS_LANGUAGE = {
    "confirmed": "state it plainly as fact",
    "reported": "attribute it — say who reported or stated it",
    "disputed": "name the disagreement; never pick a side the sources have not",
    "speculation": "mark it as a hypothesis or an open question, never as fact",
}

_SYSTEM = (
    "You are a research assistant for a documentary video channel. You "
    "search the web, read primary sources and report what they actually say. "
    "Respond with a JSON object only — no prose, no markdown fences."
)


@dataclass
class Fact:
    claim: str
    status: str
    source: str
    why_it_lands: str = ""

    def __post_init__(self) -> None:
        # Unknown status goes to the weakest one: understating confidence is
        # safe, overstating it is not.
        if self.status not in STATUSES:
            self.status = "speculation"

    def as_prompt_line(self) -> str:
        return (f"- [{self.status}] {self.claim} (source: {self.source}) — "
                f"{STATUS_LANGUAGE[self.status]}")


@dataclass
class Research:
    topic: str
    facts: list[Fact] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    usd: float = 0.0
    seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.facts:
            out[f.status] = out.get(f.status, 0) + 1
        return out

    @property
    def claims(self) -> list[str]:
        return [f.claim for f in self.facts]

    def as_prompt_block(self) -> str:
        if not self.facts:
            return "- none supplied"
        return "\n".join(f.as_prompt_line() for f in self.facts)

    def to_dict(self) -> dict:
        return {**asdict(self), "facts": [asdict(f) for f in self.facts]}

    @classmethod
    def from_dict(cls, raw: dict) -> "Research":
        return cls(topic=raw["topic"],
                   facts=[Fact(**f) for f in raw.get("facts", [])],
                   sources=list(raw.get("sources", [])),
                   usd=float(raw.get("usd", 0.0)),
                   seconds=float(raw.get("seconds", 0.0)),
                   tokens_in=int(raw.get("tokens_in", 0)),
                   tokens_out=int(raw.get("tokens_out", 0)))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Research":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def research_prompt(topic: str, card: ChannelCard, count: int) -> str:
    return f"""Collect {count} concrete, checkable pieces of material for a
video titled "{topic}".

THE CHANNEL
{card.role}

WHAT COUNTS AS USEFUL MATERIAL
Each item must give the viewer something to hold onto — but that does NOT mean
a date. Rank the kinds of detail in this order:

1. a mechanism: WHY something happens, in one causal sentence;
2. a comparison a person can picture: as heavy as, in the time it takes to;
3. a hard quantity, where the quantity itself is the surprise;
4. one vivid named example.

A figure is welcome when the figure itself is the surprise, and nothing is
wrong with an item that has no figure at all. Do not reach for a number to
look better sourced. A script stuffed with dates and organisation names reads
like a report being read aloud, and viewers leave.

"Scientists believe the universe is large" is useless. "The Milky Way is about
100,000 light years across" is useful. "Light from the far side left before
humans existed" is better, because it can be pictured.

Prefer primary sources: official reports, published studies, government
records, museum and institutional publications. Say which one.

BE HONEST ABOUT STATUS — this matters more than looking authoritative
- confirmed: an official record or peer-reviewed finding says exactly this.
- reported: a body, witness or publication stated it; nobody independently
  verified it.
- disputed: credible sources contradict each other.
- speculation: a hypothesis nobody has established.

A striking claim honestly marked speculation is MORE useful to us than a
confirmed-looking claim that is actually contested. We label evidence in the
video, so a mislabelled fact costs us credibility; a hypothesis labelled as a
hypothesis costs us nothing.

Do not invent a source. If you cannot name where something comes from, mark it
speculation and say so in the source field.

FOR EACH ITEM
- claim: one sentence containing the specific detail.
- status: one of the four above.
- source: the organisation, document or publication, named precisely.
- why_it_lands: one clause on why a viewer would find it striking.

Return {{"facts": [...]}}.

WRITE PLAIN PROSE INSIDE THE JSON STRINGS. No LaTeX, no markdown, no
backslashes, no footnote markers. Write "an effective population of about 200",
never "\\(N_e \\sim 200\\)". A backslash makes the whole response unparseable
and the material is lost."""


def collect(topic: str, card: ChannelCard, client: ChatClient, *,
            count: int = DEFAULT_FACTS, model: str = RESEARCH_MODEL) -> Research:
    """Topic to facts with statuses. One search-model call."""
    reply = client.ask_text(_SYSTEM, research_prompt(topic, card, count),
                            model=model, max_tokens=4000)
    facts = [_fact(f) for f in extract_facts(reply.text) if isinstance(f, dict)]
    res = Research(
        topic=topic, facts=[f for f in facts if f.claim],
        sources=reply.citations, usd=reply.usd, seconds=reply.seconds,
        tokens_in=reply.tokens_in, tokens_out=reply.tokens_out)
    # An empty fact list is never returned quietly: the script would go ahead
    # without grounding and look like a success.
    if not res.facts:
        raise LLMError(f"research on {topic!r} produced no facts although it "
                       f"found {len(res.sources)} sources")
    logger.info("research: %d facts, %d sources, $%.3f, %s", len(res.facts),
                len(res.sources), res.usd, res.by_status)
    return res


def _fact(raw: dict) -> Fact:
    return Fact(claim=str(raw.get("claim", "")).strip(),
                status=str(raw.get("status", "speculation")).strip().lower(),
                source=str(raw.get("source", "")).strip(),
                why_it_lands=str(raw.get("why_it_lands", "")).strip())


_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
# JSON knows eight escapes. LaTeX in a string ("\(N_e \sim 200\)") makes
# json.loads reject the whole reply, which once lost two videos' facts. A lone
# backslash is doubled: that restores exactly the text the model meant.
_LONE_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def extract_facts(text: str) -> list[dict]:
    """Facts from a free-text reply, whatever its shape.

    The reply may be an object, a bare array, or either inside prose or a code
    fence. A parser that only looked for "{" once returned zero facts from a
    bare array; a silent zero is worse than an error, so this raises.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^json\s*", "", raw.strip("`"), flags=re.IGNORECASE).strip()

    candidates: list = []
    for rx in (_OBJECT_RE, _ARRAY_RE):
        m = rx.search(raw)
        if not m:
            continue
        for attempt in (m.group(0), _LONE_BACKSLASH_RE.sub(r"\\\\", m.group(0))):
            try:
                candidates.append(json.loads(attempt))
                break
            except json.JSONDecodeError:
                continue
    if not candidates:
        raise LLMError(f"research reply is not JSON: {raw[:300]!r}")

    for data in candidates:
        if isinstance(data, dict):
            for key in ("facts", "items", "material", "results"):
                if isinstance(data.get(key), list) and data[key]:
                    return data[key]
            if "claim" in data:
                return [data]
        elif isinstance(data, list) and data:
            return data
    raise LLMError(f"research reply has no facts; it starts with {raw[:300]!r}")


# --- verification ------------------------------------------------------------

# A hedge in the sentence means the claim is not stated as fact. False
# positives are costlier than misses here: each one buys a paid rewrite of a
# correct sentence, so the list grew from real misfires ("proposed",
# "questioned whether", "no settled answer").
_HEDGES = ("may ", "might ", "could ", "appears", "suggests", "reportedly",
           "according to", "claims", "alleged", "disputed", "unconfirmed",
           "hypothes", "one idea", "some argue", "no one has", "unproven",
           "propos", "settled", "question", "doubt", "uncertain", "unclear",
           "debat", "contested", "disagree", "whether", "unknown",
           "not known", "remains open")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def overstated(narration: str, facts: list[Fact]) -> list[str]:
    """Disputed or speculative facts that the script states as plain fact.

    A sentence is matched to a fact by its significant words, because the
    script retells the claim rather than quoting it.
    """
    sentences = [s for s in _SENTENCE_END.split(narration) if s.strip()]
    out: list[str] = []
    for fact in facts:
        if fact.status in ("confirmed", "reported"):
            continue
        words = set(re.findall(r"[a-z]{5,}", fact.claim.lower()))
        if len(words) < 3:
            continue
        for s in sentences:
            hit = words & set(re.findall(r"[a-z]{5,}", s.lower()))
            if len(hit) / len(words) >= 0.5 and not any(h in s.lower() for h in _HEDGES):
                out.append(f'a "{fact.status}" fact is stated as fact: '
                           f'"{s.strip()[:110]}"')
                break
    return out


# Numbers that say nothing about the world ("three ways", "one gate").
_HARMLESS_NUMBERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                     "100", "1000"}


def _digits(text: str) -> list[str]:
    # "1,519" in the script and "1519" in the source are the same number.
    return re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))


def _significant(n: str) -> int:
    return max(1, len(n.replace(".", "").strip("0")))


def _rounds_known(n: str, known: set[str]) -> bool:
    """A coarser figure within 5% of a known one is a rounding, not an
    invention. Years must match exactly: 5% of 1947 lets 1900 through."""
    if len(n) == 4 and n.isdigit() and 1000 <= int(n) <= 2100:
        return False
    v = float(n)
    for k in known:
        kv = float(k)
        if kv and abs(v - kv) / abs(kv) <= 0.05 and _significant(n) <= _significant(k):
            return True
    return False


def unsourced_numbers(narration: str, facts: list[Fact]) -> list[str]:
    """Numbers in the script that appear nowhere in the collected facts.

    With little material the model topped up from memory: probably true, and
    untraceable. Removing a figure is a repair the model does reliably.
    """
    known: set[str] = set()
    for f in facts:
        known.update(_digits(f.claim))
        known.update(_digits(f.source or ""))
    out: list[str] = []
    seen: set[str] = set()
    for s in (s for s in _SENTENCE_END.split(narration) if s.strip()):
        for n in _digits(s):
            if n in known or n in _HARMLESS_NUMBERS or n in seen:
                continue
            if _rounds_known(n, known):
                continue
            seen.add(n)
            out.append(f'the number {n} is not in the facts: "{s.strip()[:110]}". '
                       f"Remove it or make the point without the figure")
    return out
