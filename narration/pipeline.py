"""Stage 1 and the cut: prose under a budget, checks, repairs, then frames.

Given a form with N frame slots, a model pads it: measured on 36 production
scripts, slot-filling produced 15-19 verbatim-duplicate frames out of 80.
Here the model writes continuous prose and code places the frame boundaries.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from .cards import ChannelCard, VideoCard
from .checks import (Problem, check_narration, cta_follows_card,
                     missing_grounding, sentences_of)
from .llm import ChatClient, LLMError
from .research import Research, overstated, unsourced_numbers
from .script import (SCRIPT_SCHEMA, SYSTEM, repair_prompt, resolve_next_video,
                     script_prompt)
from .timing import Segmentation, segment_sections

logger = logging.getLogger(__name__)

MAX_REPAIRS = 2


@dataclass
class Script:
    main_idea: str
    key_facts: list[str]
    chapters: list[dict]
    final_cta: str
    next_video: str = ""
    usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    repairs: int = 0
    problems: list[Problem] = field(default_factory=list)
    history: list[int] = field(default_factory=list)

    @property
    def sections(self) -> list[tuple[str, str]]:
        return [(c["title"], c["narration"]) for c in self.chapters]

    @property
    def text(self) -> str:
        return " ".join(c["narration"] for c in self.chapters)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Script":
        raw = dict(raw)
        raw["problems"] = [Problem(**p) for p in raw.get("problems") or []]
        return cls(**raw)


def script_problems(chapters: list[dict], card: ChannelCard,
                    research: Research | None, *, key_facts: list[str] | None = None,
                    final_cta: str | None = None) -> list[Problem]:
    """Every finding about a script, checked sentence by sentence.

    Numbers are demanded only as far as the material goes. With no facts the
    number checks stay silent and the operator gets a note instead: a repair
    cannot conjure figures that no source provided, it can only invent them.
    """
    texts = sentences_of(chapters)
    out = check_narration(texts, card)
    if final_cta is not None:
        out += cta_follows_card(final_cta, card)
    if research is not None and research.facts:
        joined = " ".join(texts)
        out += [Problem("status", m) for m in overstated(joined, research.facts)]
        out += [Problem("source", m) for m in unsourced_numbers(joined, research.facts)]
    else:
        out += missing_grounding(key_facts or [])
    return out


def write_script(video: VideoCard, client: ChatClient, *,
                 research: Research | None = None,
                 siblings: list[str] | None = None,
                 max_repairs: int = MAX_REPAIRS) -> Script:
    """One writing call plus up to `max_repairs` repair calls.

    The best version ships, not the last: a repair for length introduced new
    problems in five cases out of five. Checks are free, so every version is
    scored; a tie keeps the earlier one, and each repair starts from the best.
    """
    card = video.channel
    logger.info("script: %s, budget %d words (%s, %.2f syll/s)", video.video_id,
                card.words, card.voice.label, card.voice.syllables_per_second)

    def blocking(data: dict) -> list[Problem]:
        return [p for p in script_problems(
                    data.get("chapters") or [], card, research,
                    key_facts=video.key_facts,
                    final_cta=str(data.get("final_cta", "")))
                if not p.advisory]

    reply = client.ask_json(SYSTEM, script_prompt(video, research, siblings),
                            SCRIPT_SCHEMA, name="script")
    replies = [reply]
    data = reply.data
    if not data.get("chapters"):
        raise LLMError("the model returned a script without chapters")
    best_data, best_problems = data, blocking(data)
    history = [len(best_problems)]

    attempt = 0
    while best_problems and attempt < max_repairs:
        attempt += 1
        logger.info("script: repair %d/%d, %d problems: %s", attempt, max_repairs,
                    len(best_problems), ", ".join(sorted({p.kind for p in best_problems})))
        reply = client.ask_json(
            SYSTEM,
            repair_prompt(video, best_data["chapters"], best_problems,
                          final_cta=str(best_data.get("final_cta", "")),
                          research=research, siblings=siblings),
            SCRIPT_SCHEMA, name="script")
        replies.append(reply)
        data = reply.data
        if not data.get("chapters"):
            logger.warning("script: the repair returned no chapters; keeping the "
                           "previous version")
            history.append(-1)
            break
        problems = blocking(data)
        history.append(len(problems))
        if len(problems) < len(best_problems):
            best_data, best_problems = data, problems

    chapters = best_data["chapters"]
    final_cta = str(best_data.get("final_cta", "")).strip()
    return Script(
        main_idea=str(best_data.get("main_idea", "")).strip(),
        key_facts=[str(x) for x in (best_data.get("key_facts") or [])],
        chapters=chapters,
        final_cta=final_cta,
        next_video=resolve_next_video(best_data.get("next_video"), siblings or []),
        usd=round(sum(r.usd for r in replies), 5),
        tokens_in=sum(r.tokens_in for r in replies),
        tokens_out=sum(r.tokens_out for r in replies),
        seconds=round(sum(r.seconds for r in replies), 1),
        repairs=attempt,
        problems=script_problems(chapters, card, research,
                                 key_facts=video.key_facts, final_cta=final_cta),
        history=history,
    )


def cut(script: Script, card: ChannelCard, *, uniform: bool = False,
        ) -> tuple[Segmentation, list[int]]:
    """Chapters to frames. No frame crosses a chapter boundary."""
    return segment_sections(script.sections, card.frames,
                            frame_seconds=card.frame_seconds,
                            rate=card.voice.syllables_per_second,
                            pause=card.voice.sentence_pause_s,
                            uniform=uniform)
